#include "types.h"
#include "riscv.h"
#include "defs.h"
#include "param.h"
#include "memlayout.h"
#include "spinlock.h"
#include "proc.h"

extern uint64 get_fb_page(int index); //helper function to get the physical address of fb[index]

uint64
sys_exit(void)
{
  int n;
  argint(0, &n);
  exit(n);
  return 0;  // not reached
}

uint64
sys_getpid(void)
{
  return myproc()->pid;
}

uint64
sys_fork(void)
{
  return fork();
}

uint64
sys_wait(void)
{
  uint64 p;
  argaddr(0, &p);
  return wait(p);
}

uint64
sys_sbrk(void)
{
  uint64 addr;
  int n;

  argint(0, &n);
  addr = myproc()->sz;
  if(growproc(n) < 0)
    return -1;
  return addr;
}

uint64
sys_sleep(void)
{
  int n;
  uint ticks0;

  argint(0, &n);
  acquire(&tickslock);
  ticks0 = ticks;
  while(ticks - ticks0 < n){
    if(killed(myproc())){
      release(&tickslock);
      return -1;
    }
    sleep(&ticks, &tickslock);
  }
  release(&tickslock);
  return 0;
}

uint64
sys_kill(void)
{
  int pid;

  argint(0, &pid);
  return kill(pid);
}

// return how many clock tick interrupts have occurred
// since start.
uint64
sys_uptime(void)
{
  uint xticks;

  acquire(&tickslock);
  xticks = ticks;
  release(&tickslock);
  return xticks;
}

// sys_flip_display: zero-copy page flip.
//
// Syscall argument 0: user virtual address of a page-aligned buffer
// that is exactly GPU_FB_PAGES (300) * PGSIZE bytes (i.e. 640x480x4 =
// 1,228,800 bytes).  The buffer must already be fully mapped in the
// calling process's address space.
//
// TODO: Students implement this syscall.
uint64
sys_flip_display(void)
{
  return -1;
}

// sys_map_display: map the GPU's kernel framebuffer pages (fb[]) directly
// into the calling process's address space with PTE_U|PTE_R|PTE_W.
//
// Syscall argument 0: desired user virtual address (must be page-aligned).
//   Pass 0 to let the kernel auto-select the next available VA above p->sz.
//
// Returns the mapped virtual address on success, (uint64)-1 on failure.
//
// TODO: Students implement this syscall.
uint64
sys_map_display(void)  //task 1
{
  struct proc *p = myproc();
  uint64 display_size = 300 * PGSIZE; 
  uint64 addr;

  //getting the user virtual address from the syscall argument
  //user_addr will be 0 if the user passed 0, which means the kernel should auto-select the VA
  uint64 user_addr;
  argaddr(0, &user_addr); 

  //if the user gave a non-zero address, we need to validate it
  //1.it must be page-aligned (checked by user_addr % PGSIZE == 0)
  //2.it must be above the current process size (p->sz) to avoid overwriting existing memory
  //3.it must not cause overflow beyond MAXVA
  if (user_addr != 0) {
    if ((user_addr % PGSIZE) != 0 || user_addr < p->sz || user_addr + display_size >= MAXVA)
      return -1;
    addr = user_addr; //if the address is valid, we will use it for mapping
  } 
  //if the user gave 0, we need to find the next available virtual address above p->sz that can fit the display_size
  else {
    addr = PGROUNDUP(p->sz);
    
    //going through the virtual address space starting from addr, checking for a contiguous range of free pages that can fit display_size
    while (addr + display_size < MAXVA) {
      int found_space = 1;
      
      //checking the next 300 pages (display_size) to see if they are all free (not mapped) in the current process's page table 
      for (uint64 curr = addr; curr < addr + display_size; curr += PGSIZE) {
        pte_t *pte = walk(p->pagetable, curr, 0); //returns the column in the page table
        if (pte != 0 && (*pte & PTE_V)) { //if the pte is valid and the page is mapped, then this range is not free
          found_space = 0;
          break; 
        }
      }
      
      if (found_space) { //if we found a contiguous range of free pages that can fit the display, we will use this addr for mapping
        break; 
      }
      
      addr += PGSIZE;  //if not found, move to the next page and check again
    }
    
    if (addr + display_size >= MAXVA) //if we exit the loop without finding a suitable range, it means there is not enough virtual address space to map the display, so we return -1
      return -1; 
  }

  uint64 current = addr;
  //after we foind a valid address space to contain the display, we start the actual mapping
  for(int i = 0; i < 300; i++) {
    uint64 pa = get_fb_page(i); //get the physical address of fb[i] using the helper function get_fb_page
    int error = mappages(p->pagetable, current, PGSIZE, pa, PTE_U|PTE_R|PTE_W); //read, write, user premmissions
    if (error < 0) { //if mappages returns -1 , mapping falied
      if (current > addr) { //if we have mapped some pages successfully before the failure, we need to unmap those pages to clean up
        uvmunmap(p->pagetable, addr, (current - addr) / PGSIZE, 0);
      }
      return -1;
    }
    current += PGSIZE;
  }

  if (addr + display_size > p->sz) { //if the mapped display continues beyond the current process size, we need to update p->dz, so that the process knows that it's memory grew
    p->sz = addr + display_size;
  }

  return addr; 
}