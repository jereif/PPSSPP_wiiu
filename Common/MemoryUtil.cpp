// Copyright (C) 2003 Dolphin Project.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2.0 or later versions.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License 2.0 for more details.

// A copy of the GPL 2.0 should have been included with the program.
// If not, see http://www.gnu.org/licenses/

// Official SVN repository and contact information can be found at
// http://code.google.com/p/dolphin-emu/

#include "ppsspp_config.h"

#include <cstring>

#include "Common/Common.h"
#include "Common/Log.h"
#include "Common/MemoryUtil.h"
#include "Common/StringUtils.h"
#include "Common/SysError.h"

#ifdef _WIN32
#include "Common/CommonWindows.h"
#else
#include <errno.h>
#include <stdio.h>
#endif

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/mman.h>
#include <mach/vm_param.h>
#endif

#ifdef __wiiu__
#include <malloc.h>
#include <stdarg.h>
#include <wiiu/os/memory.h>
#include <wiiu/mem/expandedheap.h>
#include <wiiu/mem/frameheap.h>
static MEMExpandedHeap *rwx_heap = nullptr;
static bool rwx_heap_shared = false;

static void WiiU_LogJit(const char *fmt, ...) {
	char buf[512];
	va_list va;
	va_start(va, fmt);
	vsnprintf(buf, sizeof(buf), fmt, va);
	va_end(va);
	NOTICE_LOG(MEMMAP, "%s", buf);
}

static void InitWiiUExecutableHeap() {
	if (rwx_heap)
		return;

	{
		u32 cgAddr = 0, cgSize = 0, pAddr = 0, pSize = 0;
		BOOL cgOk = OSGetCodegenVirtAddrRange(&cgAddr, &cgSize);
		int qCg = cgAddr ? OSQueryVirtAddr((void *)(uintptr_t)cgAddr) : -1;
		OSGetAvailPhysAddrRange(&pAddr, &pSize);
		WiiU_LogJit("jit: OSGetCodegenVirtAddrRange ok=%d addr=0x%08X size=%u query=%d availPhys=0x%08X(%u)",
			(int)cgOk, (unsigned)cgAddr, (unsigned)cgSize, qCg, (unsigned)pAddr, (unsigned)pSize);
	}

	const uintptr_t rwxStart = 0x00800000;
	const uintptr_t rwxEnd   = 0x01000000;

	MEMExpandedHeap *bestExpHeap = nullptr;
	uint32_t bestExpAvail = 0;
	MEMFrameHeap *bestFrmHeap = nullptr;
	uint32_t bestFrmAvail = 0;

	// 1. Scan 0x00800000 - 0x01000000 for all Expanded or Frame Heaps (including inner heaps nested inside outer heaps).
	for (uintptr_t addr = rwxStart; addr <= rwxEnd - sizeof(MEMExpandedHeap); addr += 4) {
		MEMHeapHeader *hdr = (MEMHeapHeader *)addr;
		if (hdr->tag == MEM_EXPANDED_HEAP_TAG) {
			uintptr_t dStart = (uintptr_t)hdr->dataStart;
			uintptr_t dEnd   = (uintptr_t)hdr->dataEnd;
			if (dStart >= addr + sizeof(MEMExpandedHeap) && dStart < rwxEnd &&
			    dEnd > dStart && dEnd <= rwxEnd) {
				MEMExpandedHeap *exp = (MEMExpandedHeap *)hdr;
				uint32_t avail = MEMGetAllocatableSizeForExpHeapEx(exp, 0x100);
				uint32_t totalFree = MEMGetTotalFreeSizeForExpHeap(exp);
				WiiU_LogJit("jit: Found RWX ExpHeap at 0x%08X (data=0x%08X..0x%08X avail=%u totalFree=%u)",
				            (unsigned)addr, (unsigned)dStart, (unsigned)dEnd, (unsigned)avail, (unsigned)totalFree);
				if (avail > bestExpAvail) {
					bestExpAvail = avail;
					bestExpHeap = exp;
				}
			}
		} else if (hdr->tag == MEM_FRAME_HEAP_TAG) {
			uintptr_t dStart = (uintptr_t)hdr->dataStart;
			uintptr_t dEnd   = (uintptr_t)hdr->dataEnd;
			if (dStart >= addr + sizeof(MEMFrameHeap) && dStart < rwxEnd &&
			    dEnd > dStart && dEnd <= rwxEnd) {
				MEMFrameHeap *frm = (MEMFrameHeap *)hdr;
				uint32_t avail = MEMGetAllocatableSizeForFrmHeapEx(frm, 0x100);
				WiiU_LogJit("jit: Found RWX FrmHeap at 0x%08X (data=0x%08X..0x%08X head=0x%08X tail=0x%08X avail=%u)",
				            (unsigned)addr, (unsigned)dStart, (unsigned)dEnd,
				            (unsigned)(uintptr_t)frm->head, (unsigned)(uintptr_t)frm->tail, (unsigned)avail);
				if (avail > bestFrmAvail) {
					bestFrmAvail = avail;
					bestFrmHeap = frm;
				}
			}
		}
	}

	// Diagnostic: list the biggest used blocks of the largest shared ExpHeap so we know who owns the RWX space.
	if (bestExpHeap) {
		uint32_t usedTotal = 0, usedCount = 0;
		for (MEMExpandedHeapBlock *b = bestExpHeap->usedList.head; b && usedCount < 4096; b = b->next, usedCount++) {
			usedTotal += b->blockSize;
			if (b->blockSize >= 0x8000) {
				const uint8_t *d = (const uint8_t *)(b + 1);
				WiiU_LogJit("jit: used block %p size=%u attribs=%08x tag=%04x data=%02x%02x%02x%02x %.8s", (void *)b, (unsigned)b->blockSize, (unsigned)b->attribs, (unsigned)b->tag, d[0], d[1], d[2], d[3], (const char *)d + 4);
			}
		}
		WiiU_LogJit("jit: ExpHeap used blocks=%u total=%u", (unsigned)usedCount, (unsigned)usedTotal);
	}

	// 2. Also scan 4 KB pages in 0x00809000..0x00FFF000 for the largest contiguous all-zero range.
	const uintptr_t scanStart = 0x00809000;
	const uintptr_t scanEnd   = 0x00FFF000;
	const uintptr_t pageSize  = 0x1000;

	uintptr_t bestZeroStart = 0;
	uintptr_t bestZeroEnd   = 0;
	uintptr_t curZeroStart  = 0;

	for (uintptr_t page = scanStart; page < scanEnd; page += pageSize) {
		const uint32_t *words = (const uint32_t *)page;
		bool allZero = true;
		for (size_t i = 0; i < pageSize / sizeof(uint32_t); ++i) {
			if (words[i] != 0) {
				allZero = false;
				break;
			}
		}
		if (allZero) {
			if (!curZeroStart)
				curZeroStart = page;
			uintptr_t curZeroEnd = page + pageSize;
			if (curZeroEnd - curZeroStart > bestZeroEnd - bestZeroStart) {
				bestZeroStart = curZeroStart;
				bestZeroEnd   = curZeroEnd;
			}
		} else {
			curZeroStart = 0;
		}
	}

	const uintptr_t guardBottom = 0x20000;
	const uintptr_t guardTop    = 0x4000;
	uint32_t zeroUsable = 0;
	if (bestZeroEnd > bestZeroStart && (bestZeroEnd - bestZeroStart) > (guardBottom + guardTop)) {
		zeroUsable = (uint32_t)((bestZeroEnd - bestZeroStart) - (guardBottom + guardTop));
	}
	uint32_t expUsable = (bestExpHeap && bestExpAvail > 64 * 1024) ? (bestExpAvail - 64 * 1024) : 0;
	uint32_t frmUsable = (bestFrmHeap && bestFrmAvail > 64 * 1024) ? (bestFrmAvail - 64 * 1024) : 0;

	WiiU_LogJit("jit: RWX candidates -> expUsable=%u frmUsable=%u zeroUsable=%u (0x%08X..0x%08X)",
	            (unsigned)expUsable, (unsigned)frmUsable, (unsigned)zeroUsable,
	            (unsigned)bestZeroStart, (unsigned)bestZeroEnd);

	// If the all-zero range overlaps the free space of an existing Aroma ExpHeap, we must NOT just
	// create a heap on top of it (Aroma could later allocate the same memory). Instead, properly
	// allocate the free tail from that heap, and extend it with the unclaimed zero pages after the heap end.
	if (bestExpHeap && expUsable >= 256 * 1024) {
		MEMHeapHeader *hh = (MEMHeapHeader *)bestExpHeap;
		uintptr_t hStart = (uintptr_t)hh->dataStart;
		uintptr_t hEnd   = (uintptr_t)hh->dataEnd;
		if (bestZeroStart < hEnd && bestZeroEnd > hStart) {
			uint32_t sz = expUsable & ~0xFFFu;
			void *p = MEMAllocFromExpHeapEx(bestExpHeap, sz, -0x1000);
			if (p) {
				uintptr_t start = (uintptr_t)p;
				uintptr_t end = start + sz;
				if (end == hEnd && bestZeroEnd > hEnd + guardTop) {
					end = bestZeroEnd - guardTop;
				}
				rwx_heap = MEMCreateExpHeapEx(p, (uint32_t)(end - start), MEM_HEAP_FLAG_ZERO_ALLOCATED | MEM_HEAP_FLAG_USE_LOCK);
				rwx_heap_shared = false;
				WiiU_LogJit("jit: Claimed RWX block from Aroma ExpHeap: 0x%08X..0x%08X (%u bytes, heapEnd=0x%08X) handle=%p",
				            (unsigned)start, (unsigned)end, (unsigned)(end - start), (unsigned)hEnd, rwx_heap);
				if (rwx_heap)
					return;
			}
		}
	}

	if (expUsable >= frmUsable && expUsable >= zeroUsable && expUsable >= 1024 * 1024) {
		rwx_heap = bestExpHeap;
		rwx_heap_shared = true;
		WiiU_LogJit("jit: Using shared RWX ExpHeap at %p (avail=%u)", rwx_heap, (unsigned)bestExpAvail);
		return;
	}

	if (frmUsable >= zeroUsable && frmUsable >= 1024 * 1024) {
		void *subMem = MEMAllocFromFrmHeapEx(bestFrmHeap, frmUsable, -0x100);
		if (subMem) {
			rwx_heap = MEMCreateExpHeapEx(subMem, frmUsable, MEM_HEAP_FLAG_ZERO_ALLOCATED | MEM_HEAP_FLAG_USE_LOCK);
			rwx_heap_shared = false;
			WiiU_LogJit("jit: Created RWX ExpHeap inside FrmHeap at %p size=%u", subMem, (unsigned)frmUsable);
			return;
		}
	}

	if (zeroUsable >= 1024 * 1024) {
		uintptr_t heapStart = bestZeroStart + guardBottom;
		rwx_heap = MEMCreateExpHeapEx((void *)heapStart, zeroUsable, MEM_HEAP_FLAG_ZERO_ALLOCATED | MEM_HEAP_FLAG_USE_LOCK);
		rwx_heap_shared = false;
		WiiU_LogJit("jit: Created RWX ExpHeap at 0x%08X size=%u handle=%p",
		            (unsigned)heapStart, (unsigned)zeroUsable, rwx_heap);
	} else {
		WiiU_LogJit("jit: ERROR - no suitable RWX region found!");
	}
}
#endif

#ifndef _WIN32
#include <unistd.h>
#endif
static int hint_location;
#ifdef __APPLE__
#define MEM_PAGE_SIZE (PAGE_SIZE)
#elif defined(_WIN32)
static SYSTEM_INFO sys_info;
#define MEM_PAGE_SIZE (uintptr_t)(sys_info.dwPageSize)
#elif defined(__wiiu__)
#define MEM_PAGE_SIZE OS_MMAP_PAGE_SIZE
#else
#define MEM_PAGE_SIZE (getpagesize())
#endif

#define MEM_PAGE_MASK ((MEM_PAGE_SIZE)-1)
#define ppsspp_round_page(x) ((((uintptr_t)(x)) + MEM_PAGE_MASK) & ~(MEM_PAGE_MASK))

#ifdef _WIN32
// Win32 memory protection flags are odd...
static uint32_t ConvertProtFlagsWin32(uint32_t flags) {
	uint32_t protect = 0;
	switch (flags) {
	case 0: protect = PAGE_NOACCESS; break;
	case MEM_PROT_READ: protect = PAGE_READONLY; break;
	case MEM_PROT_WRITE: protect = PAGE_READWRITE; break;   // Can't set write-only
	case MEM_PROT_EXEC: protect = PAGE_EXECUTE; break;
	case MEM_PROT_READ | MEM_PROT_EXEC: protect = PAGE_EXECUTE_READ; break;
	case MEM_PROT_WRITE | MEM_PROT_EXEC: protect = PAGE_EXECUTE_READWRITE; break;  // Can't set write-only
	case MEM_PROT_READ | MEM_PROT_WRITE: protect = PAGE_READWRITE; break;
	case MEM_PROT_READ | MEM_PROT_WRITE | MEM_PROT_EXEC: protect = PAGE_EXECUTE_READWRITE; break;
	}
	return protect;
}
#elif !defined(__wiiu__)

static uint32_t ConvertProtFlagsUnix(uint32_t flags) {
	uint32_t protect = 0;
	if (flags & MEM_PROT_READ)
		protect |= PROT_READ;
	if (flags & MEM_PROT_WRITE)
		protect |= PROT_WRITE;
	if (flags & MEM_PROT_EXEC)
		protect |= PROT_EXEC;
	return protect;
}

#endif

#if defined(_WIN32) && defined(_M_X64)
static uintptr_t last_executable_addr;
static void *SearchForFreeMem(size_t size) {
	if (!last_executable_addr)
		last_executable_addr = (uintptr_t) &hint_location - sys_info.dwPageSize;
	last_executable_addr -= size;

	MEMORY_BASIC_INFORMATION info;
	while (VirtualQuery((void *)last_executable_addr, &info, sizeof(info)) == sizeof(info)) {
		// went too far, unusable for executable memory
		if (last_executable_addr + 0x80000000 < (uintptr_t) &hint_location)
			return NULL;

		uintptr_t end = last_executable_addr + size;
		if (info.State != MEM_FREE)
		{
			last_executable_addr = (uintptr_t) info.AllocationBase - size;
			continue;
		}

		if ((uintptr_t)info.BaseAddress + (uintptr_t)info.RegionSize >= end &&
			(uintptr_t)info.BaseAddress <= last_executable_addr)
			return (void *)last_executable_addr;

		last_executable_addr -= size;
	}

	return NULL;
}
#endif

// This is purposely not a full wrapper for virtualalloc/mmap, but it
// provides exactly the primitive operations that PPSSPP needs.
#ifdef __wiiu__
size_t GetMaxExecutableAllocSize() {
	if (!rwx_heap)
		InitWiiUExecutableHeap();
	if (!rwx_heap)
		return 0;
	uint32_t maxAlloc = MEMGetAllocatableSizeForExpHeapEx(rwx_heap, 0x100);
	uint32_t reserve  = rwx_heap_shared ? (128 * 1024) : 0;
	return maxAlloc > reserve ? (size_t)(maxAlloc - reserve) : 0;
}
#endif
void *AllocateExecutableMemory(size_t size) {
#if defined(_WIN32)
	void *ptr = nullptr;
	DWORD prot = PAGE_EXECUTE_READWRITE;
	if (PlatformIsWXExclusive())
		prot = PAGE_READWRITE;
	if (sys_info.dwPageSize == 0)
		GetSystemInfo(&sys_info);
#if defined(_M_X64)
	if ((uintptr_t)&hint_location > 0xFFFFFFFFULL) {
		size_t aligned_size = ppsspp_round_page(size);
#if 1   // Turn off to hunt for RIP bugs on x86-64.
		ptr = SearchForFreeMem(aligned_size);
		if (!ptr) {
			// Let's try again, from the top.
			// When we deallocate, this doesn't change, so we eventually run out of space.
			last_executable_addr = 0;
			ptr = SearchForFreeMem(aligned_size);
		}
#endif
		if (ptr) {
			ptr = VirtualAlloc(ptr, aligned_size, MEM_RESERVE | MEM_COMMIT, prot);
		} else {
			WARN_LOG(COMMON, "Unable to find nearby executable memory for jit. Proceeding with far memory.");
			// Can still run, thanks to "RipAccessible".
			ptr = VirtualAlloc(nullptr, aligned_size, MEM_RESERVE | MEM_COMMIT, prot);
		}
	}
	else
#endif
	{
#if PPSSPP_PLATFORM(UWP)
		ptr = VirtualAllocFromApp(0, size, MEM_RESERVE | MEM_COMMIT, prot);
#else
		ptr = VirtualAlloc(0, size, MEM_RESERVE | MEM_COMMIT, prot);
#endif
	}
#elif defined(__wiiu__)
	if (!rwx_heap) {
		InitWiiUExecutableHeap();
	}
	void *ptr = nullptr;
	if (rwx_heap) {
		uint32_t maxAlloc = MEMGetAllocatableSizeForExpHeapEx(rwx_heap, 0x100);
		uint32_t reserve  = rwx_heap_shared ? (128 * 1024) : 0;
		if (maxAlloc > reserve && size <= (size_t)(maxAlloc - reserve)) {
			ptr = MEMAllocFromExpHeapEx(rwx_heap, (uint32_t)size, rwx_heap_shared ? -0x100 : 0x100);
			if (ptr && rwx_heap_shared) {
				memset(ptr, 0, size);
			}
		}
		WiiU_LogJit("jit: AllocateExecutableMemory(%u) -> %p (maxAlloc=%u reserve=%u)",
		            (unsigned)size, ptr, (unsigned)maxAlloc, (unsigned)reserve);
	}
	if (ptr) {
		INFO_LOG(MEMMAP, "Allocated %d bytes of executable memory at %p", (int)size, ptr);
	} else {
		ERROR_LOG(MEMMAP, "Failed to allocate %d bytes of executable memory", (int)size);
	}
#else
	static char *map_hint = 0;
#if defined(_M_X64)
	// Try to request one that is close to our memory location if we're in high memory.
	// We use a dummy global variable to give us a good location to start from.
	if (!map_hint) {
		if ((uintptr_t) &hint_location > 0xFFFFFFFFULL)
			map_hint = (char*)ppsspp_round_page(&hint_location) - 0x20000000; // 0.5gb lower than our approximate location
		else
			map_hint = (char*)0x20000000; // 0.5GB mark in memory
	} else if ((uintptr_t) map_hint > 0xFFFFFFFFULL) {
		map_hint -= ppsspp_round_page(size); /* round down to the next page if we're in high memory */
	}
#endif

	int prot = PROT_READ | PROT_WRITE | PROT_EXEC;
	if (PlatformIsWXExclusive())
		prot = PROT_READ | PROT_WRITE;  // POST_EXEC is added later in this case.

	void* ptr = mmap(map_hint, size, prot, MAP_ANON | MAP_PRIVATE, -1, 0);
#endif /* defined(_WIN32) */

#if !defined(_WIN32) && !defined(__wiiu__)
	static const void *failed_result = MAP_FAILED;
#else
	static const void *failed_result = nullptr;
#endif

	if (ptr == failed_result) {
		ptr = nullptr;
		ERROR_LOG(MEMMAP, "Failed to allocate executable memory (%d) errno=%d", (int)size, errno);
	}

#if defined(_M_X64) && !defined(_WIN32)
	else if ((uintptr_t)map_hint <= 0xFFFFFFFF) {
		// Round up if we're below 32-bit mark, probably allocating sequentially.
		map_hint += ppsspp_round_page(size);

		// If we moved ahead too far, skip backwards and recalculate.
		// When we free, we keep moving forward and eventually move too far.
		if ((uintptr_t)map_hint - (uintptr_t) &hint_location >= 0x70000000) {
			map_hint = 0;
		}
	}
#endif
	return ptr;
}

void *AllocateMemoryPages(size_t size, uint32_t memProtFlags) {
	size = ppsspp_round_page(size);
#ifdef _WIN32
	if (sys_info.dwPageSize == 0)
		GetSystemInfo(&sys_info);
	uint32_t protect = ConvertProtFlagsWin32(memProtFlags);
#if PPSSPP_PLATFORM(UWP)
	void* ptr = VirtualAllocFromApp(0, size, MEM_COMMIT, protect);
#else
	void* ptr = VirtualAlloc(0, size, MEM_COMMIT, protect);
#endif
	if (!ptr) {
		ERROR_LOG(MEMMAP, "Failed to allocate raw memory pages");
		return nullptr;
	}
#elif defined(__wiiu__)
	void* ptr = malloc(size);
#else
	uint32_t protect = ConvertProtFlagsUnix(memProtFlags);
	void *ptr = mmap(0, size, protect, MAP_ANON | MAP_PRIVATE, -1, 0);
	if (ptr == MAP_FAILED) {
		ERROR_LOG(MEMMAP, "Failed to allocate raw memory pages: errno=%d", errno);
		return nullptr;
	}
#endif

	// printf("Mapped memory at %p (size %ld)\n", ptr,
	//	(unsigned long)size);
	return ptr;
}

void *AllocateAlignedMemory(size_t size, size_t alignment) {
#ifdef _WIN32
	void* ptr = _aligned_malloc(size,alignment);
#else
	void* ptr = NULL;
#if defined(__ANDROID__) || defined(__wiiu__)
	ptr = memalign(alignment, size);
#else
	if (posix_memalign(&ptr, alignment, size) != 0) {
		ptr = nullptr;
	}
#endif
#endif

	_assert_msg_(ptr != nullptr, "Failed to allocate aligned memory");
	return ptr;
}

void FreeMemoryPages(void *ptr, size_t size) {
	if (!ptr)
		return;
	uintptr_t page_size = GetMemoryProtectPageSize();
	size = (size + page_size - 1) & (~(page_size - 1));
#ifdef _WIN32
	if (!VirtualFree(ptr, 0, MEM_RELEASE)) {
		ERROR_LOG(MEMMAP, "FreeMemoryPages failed!\n%s", GetLastErrorMsg().c_str());
	}
#elif defined(__wiiu__)
	if ((uintptr_t)ptr >= 0x00800000 && (uintptr_t)ptr < 0x01000000) {
		if (rwx_heap)
			MEMFreeToExpHeap(rwx_heap, ptr);
	} else {
		free(ptr);
	}
#else
	munmap(ptr, size);
#endif
}

void FreeAlignedMemory(void* ptr) {
	if (!ptr)
		return;
#ifdef _WIN32
	_aligned_free(ptr);
#else
	free(ptr);
#endif
}

bool PlatformIsWXExclusive() {
	// Needed on platforms that disable W^X pages for security. Even without block linking, still should be much faster than IR JIT.
	// This might also come in useful for UWP (Universal Windows Platform) if I'm understanding things correctly.
#if defined(IOS) || PPSSPP_PLATFORM(UWP) || defined(__OpenBSD__)
	return true;
#else
	// Returning true here lets you test the W^X path on Windows and other non-W^X platforms.
	return false;
#endif
}

bool ProtectMemoryPages(const void* ptr, size_t size, uint32_t memProtFlags) {
	VERBOSE_LOG(JIT, "ProtectMemoryPages: %p (%d) : r%d w%d x%d", ptr, (int)size,
			(memProtFlags & MEM_PROT_READ) != 0, (memProtFlags & MEM_PROT_WRITE) != 0, (memProtFlags & MEM_PROT_EXEC) != 0);

	if (PlatformIsWXExclusive()) {
		if ((memProtFlags & (MEM_PROT_WRITE | MEM_PROT_EXEC)) == (MEM_PROT_WRITE | MEM_PROT_EXEC)) {
			_assert_msg_(false, "Bad memory protect flags %d: W^X is in effect, can't both write and exec", memProtFlags);
		}
	}
	// Note - VirtualProtect will affect the full pages containing the requested range.
	// mprotect does not seem to, at least not on Android unless I made a mistake somewhere, so we manually round.
#ifdef _WIN32
	uint32_t protect = ConvertProtFlagsWin32(memProtFlags);

#if PPSSPP_PLATFORM(UWP)
	DWORD oldValue;
	if (!VirtualProtectFromApp((void *)ptr, size, protect, &oldValue)) {
		ERROR_LOG(MEMMAP, "WriteProtectMemory failed!\n%s", GetLastErrorMsg().c_str());
		return false;
	}
#else
	DWORD oldValue;
	if (!VirtualProtect((void *)ptr, size, protect, &oldValue)) {
		ERROR_LOG(MEMMAP, "WriteProtectMemory failed!\n%s", GetLastErrorMsg().c_str());
		return false;
	}
#endif
	return true;
#elif defined(__wiiu__)
	return true;
#else
	uint32_t protect = ConvertProtFlagsUnix(memProtFlags);
	uintptr_t page_size = GetMemoryProtectPageSize();

	uintptr_t start = (uintptr_t)ptr;
	uintptr_t end = (uintptr_t)ptr + size;
	start &= ~(page_size - 1);
	end = (end + page_size - 1) & ~(page_size - 1);
	int retval = mprotect((void *)start, end - start, protect);
	if (retval != 0) {
		ERROR_LOG(MEMMAP, "mprotect failed (%p)! errno=%d (%s)", (void *)start, errno, strerror(errno));
		return false;
	}
	return true;
#endif
}

int GetMemoryProtectPageSize() {
#ifdef _WIN32
	if (sys_info.dwPageSize == 0)
		GetSystemInfo(&sys_info);
	return sys_info.dwPageSize;
#endif
	return MEM_PAGE_SIZE;
}
