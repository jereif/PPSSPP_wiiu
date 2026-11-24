#pragma once
#ifdef __wiiu__

#include <atomic>
#include <functional>
#include <wiiu/os/thread.h>
#include <wiiu/os/time.h>
#include "Common/CommonTypes.h"

namespace WiiUAsyncGPU {

struct DeferredGeEvent {
	enum Type : u8 { INTR = 0, SYNC = 1, WAIT = 2 };
	Type type;
	int arg1;
	int arg2;
	u64 ticks;
};

// Launches cpuFn on Espresso Core 2 while the calling Core 1 (GX2 main core)
// concurrently services ProcessDLQueueInternal() and synchronous GX2 requests.
void RunCpuOnCore2(const std::function<void()> &cpuFn);

// True when called from the Core 2 CPU worker while RunCpuOnCore2 is active.
bool IsOnCpuCore();

// True while the Core 1 / Core 2 split is active.
bool IsAsyncActive();

// Asynchronous: asks Core 1 to run gpu->ProcessDLQueueInternal() without blocking Core 2.
void SignalProcessDL();

// True if Core 1 is currently running or about to run ProcessDLQueueInternal().
bool IsGfxBusy();

// Synchronous: waits until Core 1 finishes any pending ProcessDLQueueInternal().
void WaitGfxIdle();

// Synchronous: waits for pending DL work on Core 1, then runs fn on Core 1.
enum SyncReason : u8 { SYNC_OTHER = 0, SYNC_ENQ = 1, SYNC_WAIT = 2, SYNC_INV = 3, SYNC_DISP = 4 };
void RunOnGfxCore(const std::function<void()> &fn, SyncReason reason = SYNC_OTHER);

// Asynchronous cache invalidation queue (Core 2 pushes, Core 1 drains before ProcessDLQueue/syncFn).
bool PushAsyncInvalidate(u32 addr, int size, int type);

// Deferred GE -> CoreTiming/Kernel event queue (Core 1 pushes, Core 2 drains safely).
void PushGeEvent(DeferredGeEvent::Type type, int arg1, int arg2, u64 ticks);
void DrainGeEvents();

void SetLastCpuTicks(u64 ticks);
u64 GetLastCpuTicks();

} // namespace WiiUAsyncGPU
#endif
