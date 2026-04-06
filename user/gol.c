//
// gol.c – Conway's Game of Life demo. Supports two display modes:
//
//   flip (default): zero-copy page flip via flip_display().
//     Each frame the child:
//       1. Computes the next GOL generation into an off-screen buffer.
//       2. Renders cells into a user-space pixel buffer (back-buffer).
//       3. Calls flip_display(back_buf) – the kernel changes the GPU
//          resource's backing pages to point at back_buf's physical
//          pages; NO pixel data is copied.
//       4. The two pixel buffers are ping-ponged so the GPU is never
//          reading from the same buffer we are currently drawing into.
//
//   map: direct-mapped framebuffer via map_display().
//     Each frame the child renders straight into the mapped fb. The
//     display_daemon flushes the display ~every tick, so user code does not
//     need to flush explicitly. No double-buffering.
//
// Invoke as:   gol            (flip mode, default)
//              gol flip
//              gol map
//
// Grid  : 80 × 60 cells  (each cell = 8×8 pixels → 640×480 screen)
// Memory: 2 × (640×480×4) = 2 × 1,228,800 bytes of page-aligned heap
//         per child process.
//

#include "kernel/types.h"
#include "user/user.h"

// ── Screen / grid constants ──────────────────────────────────────────
#define SCREEN_W 640
#define SCREEN_H 480
#define CELL_W 8                           // pixels per cell horizontally
#define CELL_H 8                           // pixels per cell vertically
#define GRID_W (SCREEN_W / CELL_W)         // 80
#define GRID_H (SCREEN_H / CELL_H)         // 60
#define FB_BYTES (SCREEN_W * SCREEN_H * 4) // 1,228,800 (300 × PGSIZE)

// ── BGRX pixel helpers ───────────────────────────────────────────────
// uint32 layout on little-endian RISC-V: bits[7:0]=B, [15:8]=G, [23:16]=R
static uint32
rgb(int r, int g, int b)
{
    return ((uint32)r << 16) | ((uint32)g << 8) | (uint32)b;
}

// ── GOL state (static – each forked child gets its own copy; xv6 fork
//    copies the parent's memory eagerly, no COW) ─────────────────────
static uint8 grid[2][GRID_W * GRID_H];

// ── Helpers ──────────────────────────────────────────────────────────
static void
grid_clear(uint8 *g)
{
    memset(g, 0, GRID_W * GRID_H);
}

// Toroidal (wrapping) get/set
static int
cell_get(const uint8 *g, int x, int y)
{
    x = (x % GRID_W + GRID_W) % GRID_W;
    y = (y % GRID_H + GRID_H) % GRID_H;
    return g[y * GRID_W + x];
}

static void
cell_set(uint8 *g, int x, int y)
{
    x = (x % GRID_W + GRID_W) % GRID_W;
    y = (y % GRID_H + GRID_H) % GRID_H;
    g[y * GRID_W + x] = 1;
}

// Advance one GOL generation: cur → nxt
static void
gol_step(const uint8 *cur, uint8 *nxt)
{
    for (int y = 0; y < GRID_H; y++)
    {
        for (int x = 0; x < GRID_W; x++)
        {
            int n = cell_get(cur, x - 1, y - 1) + cell_get(cur, x, y - 1) + cell_get(cur, x + 1, y - 1) + cell_get(cur, x - 1, y) + cell_get(cur, x + 1, y) + cell_get(cur, x - 1, y + 1) + cell_get(cur, x, y + 1) + cell_get(cur, x + 1, y + 1);
            int alive = cur[y * GRID_W + x];
            nxt[y * GRID_W + x] = alive ? (n == 2 || n == 3) : (n == 3);
        }
    }
}

// Render grid into a 32-bit BGRX pixel buffer
static void
render(const uint8 *g, uint32 *fb, uint32 alive_color, uint32 dead_color)
{
    for (int cy = 0; cy < GRID_H; cy++)
    {
        uint32 c = g[cy * GRID_W] ? alive_color : dead_color; // dummy init
        for (int cx = 0; cx < GRID_W; cx++)
        {
            c = g[cy * GRID_W + cx] ? alive_color : dead_color;
            for (int py = 0; py < CELL_H; py++)
            {
                int base = ((cy * CELL_H + py) * SCREEN_W) + cx * CELL_W;
                for (int px = 0; px < CELL_W; px++)
                    fb[base + px] = c;
            }
        }
    }
}

// Allocate page-aligned framebuffer.
// FB_BYTES = 300 * PGSIZE so as long as p->sz is page-aligned when we
// first call sbrk (guaranteed after exec), the returned pointer is also
// page-aligned.
static uint32 *
alloc_fb(void)
{
    char *p = sbrk(FB_BYTES);
    if (p == (char *)-1)
    {
        fprintf(2, "gol: sbrk failed\n");
        exit(1);
    }
    return (uint32 *)p;
}

// Run one GOL pattern in flip mode (zero-copy flip_display, double-buffered).
static void
run_pattern_flip(void (*init_fn)(uint8 *), uint32 alive_color, uint32 dead_color, int generations)
{
    // Allocate two page-aligned framebuffers for double-buffering.
    uint32 *buf[2];
    buf[0] = alloc_fb();
    buf[1] = alloc_fb();

    // Initialise grid.
    grid_clear(grid[0]);
    grid_clear(grid[1]);
    init_fn(grid[0]);

    int cur = 0;  // index into grid[] for current generation
    int draw = 0; // index into buf[] for the next draw target

    for (int gen = 0; gen < generations; gen++)
    {
        // Draw current generation into back-buffer.
        render(grid[cur], buf[draw], alive_color, dead_color);

        // Zero-copy flip: kernel re-points GPU resource to buf[draw]'s
        // physical pages by walking the calling process's page table.
        if (flip_display(buf[draw]) < 0)
        {
            fprintf(2, "gol: flip_display failed\n");
            exit(1);
        }

        // Pace the animation (~30 ms per frame at 100 Hz ticks).
        sleep(1);

        // Advance GOL: compute next generation into the other grid slot.
        gol_step(grid[cur], grid[cur ^ 1]);
        cur ^= 1;

        // Ping-pong buffers: next draw goes to the buffer the GPU is
        // no longer reading (the previous front-buffer).
        draw ^= 1;
    }

    // Clear the display to black when this pattern finishes.
    memset(buf[draw], 0, FB_BYTES);
    flip_display(buf[draw]);
}

// Run one GOL pattern in map mode (direct writes into the mapped fb).
static void
run_pattern_map(void (*init_fn)(uint8 *), uint32 alive_color, uint32 dead_color, int generations)
{
    // Map the kernel framebuffer into our address space. The returned
    // pointer refers to the exact pages the GPU is scanning out, so we
    // draw straight into it; the display_daemon flushes the display on its
    // own tick.
    uint32 *fb = (uint32 *)map_display(0);
    if (fb == (uint32 *)-1)
    {
        fprintf(2, "gol: map_display failed\n");
        exit(1);
    }

    grid_clear(grid[0]);
    grid_clear(grid[1]);
    init_fn(grid[0]);

    int cur = 0;
    for (int gen = 0; gen < generations; gen++)
    {
        render(grid[cur], fb, alive_color, dead_color);
        sleep(1); // wait for the daemon tick to flush
        gol_step(grid[cur], grid[cur ^ 1]);
        cur ^= 1;
    }

    // Clear the display to black when this pattern finishes.
    memset(fb, 0, FB_BYTES);
}

// ── Pattern 1: scattered gliders ─────────────────────────────────────
// Classic NE-moving glider:  . X .
//                            . . X
//                            X X X
static void
init_gliders(uint8 *g)
{
    // 9 gliders spread across the grid
    int origins[][2] = {
        {3, 3},
        {30, 3},
        {57, 3},
        {3, 25},
        {30, 25},
        {57, 25},
        {3, 48},
        {30, 48},
        {57, 48},
    };
    for (int k = 0; k < 9; k++)
    {
        int cx = origins[k][0], cy = origins[k][1];
        cell_set(g, cx + 1, cy + 0);
        cell_set(g, cx + 2, cy + 1);
        cell_set(g, cx + 0, cy + 2);
        cell_set(g, cx + 1, cy + 2);
        cell_set(g, cx + 2, cy + 2);
    }
}

// ── Pattern 2: five R-pentominoes ────────────────────────────────────
// R-pentomino:  . X X
//               X X .
//               . X .
// Five copies scattered — each evolves chaotically for ~1100 steps.
static void
init_rpentominos(uint8 *g)
{
    int origins[][2] = {{15, 10}, {55, 10}, {35, 30}, {15, 48}, {55, 48}};
    for (int k = 0; k < 5; k++)
    {
        int cx = origins[k][0], cy = origins[k][1];
        cell_set(g, cx + 1, cy + 0);
        cell_set(g, cx + 2, cy + 0);
        cell_set(g, cx + 0, cy + 1);
        cell_set(g, cx + 1, cy + 1);
        cell_set(g, cx + 1, cy + 2);
    }
}

// ── Pattern 3: Gosper Glider Gun ─────────────────────────────────────
// Emits a new NE-glider every 30 generations.
// Placed in the upper-left quadrant; gliders travel SE.
static void
init_glider_gun(uint8 *g)
{
    // Gosper Glider Gun cells (relative coordinates, from Wikipedia)
    static const int cells[][2] = {
        {24, 0}, {25, 0}, {22, 1}, {26, 1}, {12, 2}, {13, 2}, {20, 2}, {21, 2}, {34, 2}, {35, 2}, {11, 3}, {15, 3}, {20, 3}, {21, 3}, {34, 3}, {35, 3}, {0, 4}, {1, 4}, {10, 4}, {16, 4}, {20, 4}, {21, 4}, {0, 5}, {1, 5}, {10, 5}, {14, 5}, {15, 5}, {22, 5}, {26, 5}, {10, 6}, {16, 6}, {23, 6}, {24, 6}, {25, 6}, {11, 7}, {15, 7}, {26, 7}, {12, 8}, {13, 8}, {24, 9}, {25, 9}, {-1, -1} // sentinel
    };
    // Offset so the gun sits in the upper-left with a small margin.
    int ox = 2, oy = 12;
    for (int i = 0; cells[i][0] != -1; i++)
        cell_set(g, ox + cells[i][0], oy + cells[i][1]);
}

// ── main ──────────────────────────────────────────────────────────────
int main(int argc, char *argv[])
{
    // Pick display mode: default "flip", or "map" if requested.
    int use_map = 0;
    if (argc > 1)
    {
        if (strcmp(argv[1], "map") == 0)
            use_map = 1;
        else if (strcmp(argv[1], "flip") == 0)
            use_map = 0;
        else
        {
            fprintf(2, "usage: %s [flip|map]\n", argv[0]);
            exit(1);
        }
    }

    // Pattern colours (BGRX little-endian: value = R<<16 | G<<8 | B)
    uint32 colors_alive[3] = {
        rgb(0, 255, 0),   // green  – gliders
        rgb(255, 200, 0), // amber  – R-pentominoes
        rgb(0, 200, 255), // cyan   – glider gun
    };
    uint32 colors_dead[3] = {
        rgb(10, 20, 10), // dark green tint background
        rgb(20, 15, 5),  // dark amber tint
        rgb(5, 15, 25),  // dark cyan tint
    };
    void (*inits[3])(uint8 *) = {
        init_gliders,
        init_rpentominos,
        init_glider_gun,
    };
    int gens[3] = {100, 100, 100};

    for (int i = 0; i < 3; i++)
    {
        int pid = fork();
        if (pid < 0)
        {
            fprintf(2, "gol: fork failed\n");
            exit(1);
        }
        if (pid == 0)
        {
            // Child: run one pattern then exit.
            if (use_map)
                run_pattern_map(inits[i], colors_alive[i], colors_dead[i], gens[i]);
            else
                run_pattern_flip(inits[i], colors_alive[i], colors_dead[i], gens[i]);
            exit(0);
        }
        // Parent: wait for child before starting the next pattern.
        wait(0);
    }

    exit(0);
}
