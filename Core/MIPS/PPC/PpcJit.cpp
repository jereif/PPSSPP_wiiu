#include "ppsspp_config.h"
#if PPSSPP_ARCH(POWERPC)

#include "Common/Profiler/Profiler.h"
#include "Common/MemoryUtil.h"
#include "Common/Serialize/Serializer.h"
#include "Core/Core.h"
#include "Core/CoreTiming.h"
#include "Core/MemMap.h"
#include "Core/MIPS/MIPS.h"
#include "Core/MIPS/MIPSCodeUtils.h"
#include "Core/MIPS/MIPSInt.h"
#include "Core/MIPS/MIPSTables.h"
#include "Core/HLE/ReplaceTables.h"

#include "PpcRegCache.h"
#include "ppcEmitter.h"
#include "PpcJit.h"

#include <malloc.h>
#ifdef __wiiu__
#include <wiiu/os/time.h>
#include "GPU/GX2/ShaderManagerGX2.h"
#endif
#include <map>
#include <string>
#include <vector>
#include <algorithm>
//#include <ppcintrinsics.h>

using namespace PpcGen;

extern volatile CoreState coreState;

namespace MIPSComp
{

static u32 delaySlotFlagsValue;

// Runtime Comp_Generic fallback profiler (fixed table, zero heap allocs in hot path).
struct RtFallbackSlot {
	const char *name;
	u32 count;
};
static RtFallbackSlot g_rtFallbacks[128];
static u64 g_rtFallbackTotal = 0;

void PpcExecGeneric(u32 opEncoding, MIPSInterpretFunc func) {
	const char *name = MIPSGetName(MIPSOpcode(opEncoding));
	u32 h = ((uintptr_t)name >> 2) & 127;
	for (int i = 0; i < 16; ++i) {
		auto &e = g_rtFallbacks[(h + i) & 127];
		if (e.name == name || e.name == nullptr) {
			e.name = name;
			e.count++;
			break;
		}
	}
	g_rtFallbackTotal++;
	func(MIPSOpcode(opEncoding));
}

void PpcJitLogRuntimeFallbacks(int frames) {
	if (g_rtFallbackTotal == 0 || frames <= 0)
		return;
	std::vector<std::pair<u32, const char *>> sorted;
	for (int i = 0; i < 128; ++i) {
		if (g_rtFallbacks[i].count > 0 && g_rtFallbacks[i].name) {
			sorted.push_back(std::make_pair(g_rtFallbacks[i].count, g_rtFallbacks[i].name));
			g_rtFallbacks[i].name = nullptr;
			g_rtFallbacks[i].count = 0;
		}
	}
	u64 total = g_rtFallbackTotal;
	g_rtFallbackTotal = 0;
	std::sort(sorted.begin(), sorted.end(), [](const std::pair<u32, const char *> &a, const std::pair<u32, const char *> &b) {
		return a.first > b.first;
	});
	char line[384];
	int pos = snprintf(line, sizeof(line), "jit fallbacks: %llu/f | ", (unsigned long long)(total / frames));
	for (size_t i = 0; i < sorted.size() && i < 12 && pos < (int)sizeof(line) - 32; ++i) {
		pos += snprintf(line + pos, sizeof(line) - pos, "%s=%u ", sorted[i].second, sorted[i].first / frames);
	}
	NOTICE_LOG(JIT, "%s", line);
}

static int g_compiledBlocks = 0;
static int g_cacheClears = 0;
static int g_invalCalls = 0;
static u64 g_compileTicks = 0;
static u64 g_firstCompileTick = 0;

void Jit::CompileDelaySlot(int flags)
{
	MIPSOpcode op = Memory::Read_Instruction(js.compilerPC + 4);
	// Only CR0 needs preservation across non-NOP delay slots (FLAGREG is not used by conditional branches).
	bool needSaveCR = (flags & DELAYSLOT_SAFE) && (op.encoding != 0);
	if (needSaveCR) {
		MFCR(R19);
	}

	js.inDelaySlot = true;
	if (op.encoding != 0) {
		MIPSCompileOp(op, this);
	}
	js.inDelaySlot = false;

	if (flags & DELAYSLOT_FLUSH)
		FlushAll();

	if (needSaveCR) {
		MTCR(R19);
	}
}

void Jit::Compile(u32 em_address)
{
	PROFILE_THIS_SCOPE("jitc");
#ifdef __wiiu__
	// Adaptive tiered JIT for Wii U's 1.78 MB RWX space:
	// - <65% full: low threshold (16) so hot loops compile immediately (minimal interpreter overhead).
	// - 65-85% full: medium threshold (128) so only steady per-frame blocks take space.
	// - >85% full: high threshold (1024) so lukewarm blocks stay interpreted instead of thrashing ClearCache().
	const size_t spaceLeft = GetSpaceLeft();
	u16 hotThreshold;
	if (spaceLeft > 640 * 1024)
		hotThreshold = 16;
	else if (spaceLeft > 256 * 1024)
		hotThreshold = 128;
	else
		hotThreshold = 1024;

	u32 hidx = ((em_address >> 2) * 2654435761u) & (kHotTableSize - 1);
	if (hotAddr_[hidx] != em_address) {
		hotAddr_[hidx] = em_address;
		hotCounts_[hidx] = 0;
	}
	if (hotCounts_[hidx] < hotThreshold && !mips_->inDelaySlot) {
		u64 interpT0 = OSGetSystemTime();
		if (hotCounts_[hidx] < 65535)
			hotCounts_[hidx]++;
		int steps = 0;
		while (coreState == CORE_RUNNING && (steps < 64 || mips_->inDelaySlot)) {
			if (!mips_->inDelaySlot && steps > 0) {
				MIPSOpcode nextOp = Memory::Read_Opcode_JIT(mips_->pc);
				if (MIPS_IS_EMUHACK(nextOp))
					break;
			}
			bool wasDelay = mips_->inDelaySlot;
			u32 prevPC = mips_->pc;
			MIPS_SingleStep();
			mips_->downcount--;
			steps++;
			if (wasDelay || (!mips_->inDelaySlot && mips_->pc != prevPC + 4))
				break;
		}
		g_gx2Perf.jitInterpTicks += OSGetSystemTime() - interpT0;
		return;
	}
#endif
	u64 compileStart = OSGetSystemTime();
	if (!g_firstCompileTick)
		g_firstCompileTick = compileStart;
	if (GetSpaceLeft() < 0x4000 || blocks.IsFull())
	{
		g_cacheClears++;
		int numBlocksBefore = blocks.GetNumBlocks();
		int invalSinceClear = g_invalCalls;
		g_invalCalls = 0;
		u64 clearStart = OSGetSystemTime();
		ClearCache();
		int clearMs = (int)ticks_to_ms(OSGetSystemTime() - clearStart);
		NOTICE_LOG(JIT, "Cleared JIT cache #%d in %d ms (blocks=%d, cur=%d, inval=%d, compile=%d ms of %d ms wall)",
			g_cacheClears, clearMs, g_compiledBlocks, numBlocksBefore, invalSinceClear,
			(int)ticks_to_ms(g_compileTicks), g_firstCompileTick ? (int)ticks_to_ms(OSGetSystemTime() - g_firstCompileTick) : 0);
	}

	int block_num = blocks.AllocateBlock(em_address);
	JitBlock *b = blocks.GetBlock(block_num);
	DoJit(em_address, b);
	blocks.FinalizeBlock(block_num, jo.enableBlocklink);
	++g_compiledBlocks;

	// Drat.  The VFPU hit an uneaten prefix at the end of a block.
	if (js.startDefaultPrefix && js.MayHavePrefix())
	{
		js.startDefaultPrefix = false;
		// Our assumptions are all wrong so it's clean-slate time.
		ClearCache();

		// Let's try that one more time.  We won't get back here because we toggled the value.
		Compile(em_address);
	}
	u64 compileElapsed = OSGetSystemTime() - compileStart;
	g_compileTicks += compileElapsed;
#ifdef __wiiu__
	g_gx2Perf.jitCompileTicks += compileElapsed;
#endif
}

bool Jit::DescribeCodePtr(const u8 *ptr, std::string &name)
{
	// TODO: Not used by anything yet.
	return false;
}

void Jit::LinkBlock(u8 *exitPoint, const u8 *checkedEntry) {
	if (PlatformIsWXExclusive()) {
		ProtectMemoryPages(exitPoint, 32, MEM_PROT_READ | MEM_PROT_WRITE);
	}
	PPCXEmitter emit(exitPoint);
	emit.B(checkedEntry);
	emit.FlushIcache();
	if (PlatformIsWXExclusive()) {
		ProtectMemoryPages(exitPoint, 32, MEM_PROT_READ | MEM_PROT_EXEC);
	}
}

void Jit::UnlinkBlock(u8 *checkedEntry, u32 originalAddress) {
	if (PlatformIsWXExclusive()) {
		ProtectMemoryPages(checkedEntry, 16, MEM_PROT_READ | MEM_PROT_WRITE);
	}
	PPCXEmitter emit(checkedEntry);
	emit.MOVI2R(SREG, originalAddress);
	emit.STW(SREG, CTXREG, offsetof(MIPSState, pc));
	emit.B(dispatcher);
	emit.FlushIcache();
	if (PlatformIsWXExclusive()) {
		ProtectMemoryPages(checkedEntry, 16, MEM_PROT_READ | MEM_PROT_EXEC);
	}
}

void Jit::MovFromPC(PPCReg r) {
	LWZ(r, CTXREG, offsetof(MIPSState, pc));
}

void Jit::MovToPC(PPCReg r) {
	STW(r, CTXREG, offsetof(MIPSState, pc));
}

void Jit::SaveDowncount(PPCReg r) {
	STW(r, CTXREG, offsetof(MIPSState, downcount));
}

void Jit::RestoreDowncount(PPCReg r) {
	LWZ(r, CTXREG, offsetof(MIPSState, downcount));
}

static void ShowDownCount() {
	if (currentMIPS->downcount<0) {
		//ERROR_LOG(DYNA_REC, "MIPSState, downcount %08x", currentMIPS->downcount);
		Crash();
	}
}

void Jit::WriteDownCount(int offset)
{
	int theDowncount = js.downcountAmount + offset;
	if (jo.downcountInRegister) {
		if (theDowncount >= -32767 && theDowncount <= 32768) {
			// Single instruction: DCNTREG -= theDowncount, updates CR0 for BLT.
			ADDIC_(DCNTREG, DCNTREG, (short)(-theDowncount));
		} else {
			MOVI2R(SREG, theDowncount);
			SUBF(DCNTREG, SREG, DCNTREG, 1);
		}
	} else {
		// DCNTREG = MIPSState->downcount - theDowncount;
		MOVI2R(SREG, theDowncount);
		LWZ(DCNTREG, CTXREG, offsetof(MIPSState, downcount));
		SUBF(DCNTREG, SREG, DCNTREG, 1);
		STW(DCNTREG, CTXREG, offsetof(MIPSState, downcount));
	}
}

void Jit::Comp_Generic(MIPSOpcode op) {
	FlushAll();

	// basic jit !!
	MIPSInterpretFunc func = MIPSGetInterpretFunc(op);
	if (func)
	{
		MOVI2R(R3, op.encoding);
		MOVI2R(R4, js.compilerPC);
		MOVI2R(R5, (u32)func);
		BL(compGenericTrampoline);
	}
	const MIPSInfo info = MIPSGetInfo(op);
	if ((info & IS_VFPU) != 0 && (info & VFPU_NO_PREFIX) == 0)
	{
		// If it does eat them, it'll happen in MIPSCompileOp().
		if ((info & OUT_EAT_PREFIX) == 0)
			js.PrefixUnknown();
	}
}
	
void Jit::EatInstruction(MIPSOpcode op) {
	MIPSInfo info = MIPSGetInfo(op);
	_dbg_assert_msg_(!(info & DELAYSLOT), "Never eat a branch op.");
	_dbg_assert_msg_(!js.inDelaySlot, "Never eat an instruction inside a delayslot.");

	js.compilerPC += 4;
	js.downcountAmount += MIPSGetInstructionCycleEstimate(op);
}

void Jit::Comp_RunBlock(MIPSOpcode op) {
	// This shouldn't be necessary, the dispatcher should catch us before we get here.
	ERROR_LOG(JIT, "Comp_RunBlock should never be reached!");
}

void Jit::Comp_ReplacementFunc(MIPSOpcode op)
{
	int index = op.encoding & MIPS_EMUHACK_VALUE_MASK;
	const ReplacementTableEntry *entry = GetReplacementFunc(index);
	if (!entry) {
		ERROR_LOG(HLE, "Invalid replacement op %08x", op.encoding);
		return;
	}

	bool disabled = (entry->flags & REPFLAG_DISABLED) != 0;
	if (disabled) {
		MIPSCompileOp(Memory::Read_Instruction(js.compilerPC, true), this);
	} else if (entry->replaceFunc) {
		FlushAll();
		MOVI2R(SREG, js.compilerPC);
		MovToPC(SREG);
		QuickCallFunction((void *)entry->replaceFunc);
		if (entry->flags & (REPFLAG_HOOKENTER | REPFLAG_HOOKEXIT)) {
			MIPSCompileOp(Memory::Read_Instruction(js.compilerPC, true), this);
		} else {
			// R3 holds the cycle count returned by entry->replaceFunc.
			SUBF(DCNTREG, R3, DCNTREG);
			LWZ(FLAGREG, CTXREG, MIPS_REG_RA * 4);
			WriteExitDestInR(FLAGREG);
			js.compiling = false;
		}
	} else {
		MIPSCompileOp(Memory::Read_Instruction(js.compilerPC, true), this);
	}
}

void Jit::Comp_DoNothing(MIPSOpcode op) {

}

void Jit::FlushAll()
{
	gpr.FlushAll();
	fpr.FlushAll();
	FlushPrefixV();
}

void Jit::FlushPrefixV()
{
	if ((js.prefixSFlag & PpcJitState::PREFIX_DIRTY) != 0) {
		MOVI2R(SREG, js.prefixS);
		STW(SREG, CTXREG, offsetof(MIPSState, vfpuCtrl[VFPU_CTRL_SPREFIX]));
		js.prefixSFlag = (PpcJitState::PrefixState) (js.prefixSFlag & ~PpcJitState::PREFIX_DIRTY);
	}

	if ((js.prefixTFlag & PpcJitState::PREFIX_DIRTY) != 0) {
		MOVI2R(SREG, js.prefixT);
		STW(SREG, CTXREG, offsetof(MIPSState, vfpuCtrl[VFPU_CTRL_TPREFIX]));
		js.prefixTFlag = (PpcJitState::PrefixState) (js.prefixTFlag & ~PpcJitState::PREFIX_DIRTY);
	}

	if ((js.prefixDFlag & PpcJitState::PREFIX_DIRTY) != 0) {
		MOVI2R(SREG, js.prefixD);
		STW(SREG, CTXREG, offsetof(MIPSState, vfpuCtrl[VFPU_CTRL_DPREFIX]));
		js.prefixDFlag = (PpcJitState::PrefixState) (js.prefixDFlag & ~PpcJitState::PREFIX_DIRTY);
	}
}

void Jit::ClearCache() {
	blocks.Clear();
	ClearCodeSpace(0);
	GenerateFixedCode();
#ifdef __wiiu__
	// Decay hotness instead of zeroing so truly hot blocks recompile immediately after a clear
	// while lukewarm blocks stay interpreted.
	for (size_t i = 0; i < kHotTableSize; ++i)
		hotCounts_[i] >>= 2;
#endif
}

void Jit::InvalidateCache() {
	blocks.Clear();
}

void Jit::InvalidateCacheAt(u32 em_address, int length) {
	g_invalCalls++;
	blocks.InvalidateICache(em_address, length);
}

Jit::Jit(MIPSState *mips) : blocks(mips, this), gpr(mips, &jo),fpr(mips),mips_(mips)
{ 
#ifdef __wiiu__
	memset(hotAddr_, 0, sizeof(hotAddr_));
	memset(hotCounts_, 0, sizeof(hotCounts_));
#endif
	blocks.Init();
	gpr.SetEmitter(this);
	fpr.SetEmitter(this);
#ifdef __wiiu__
	// RWX memory in 0x00800000-0x01000000 is shared with Aroma modules; use as much as is available (max 8 MB).
	{
		size_t maxSz = GetMaxExecutableAllocSize() & ~(size_t)0xFFF;
		if (maxSz > 8 * 1024 * 1024)
			maxSz = 8 * 1024 * 1024;
		for (int sz = (int)maxSz; sz >= 512 * 1024; sz -= 16 * 1024) {
			AllocCodeSpace(sz);
			if (GetBasePtr() != nullptr) {
				NOTICE_LOG(JIT, "JIT code space: %d bytes at %p", sz, GetBasePtr());
				break;
			}
		}
	}
#else
	AllocCodeSpace(1024 * 1024 * 16);
#endif
	GenerateFixedCode();

	js.startDefaultPrefix = true;
}

void Jit::RunLoopUntil(u64 globalticks) {	
	PROFILE_THIS_SCOPE("jit");
#ifdef __wiiu__
	u64 t0 = OSGetSystemTime();
#endif
	// Run the compiled code
	((void (*)())enterCode)();
#ifdef __wiiu__
	g_gx2Perf.jitTicks += OSGetSystemTime() - t0;
#endif
}


// IDEA - could have a WriteDualExit that takes two destinations and two condition flags,
// and just have conditional that set PC "twice". This only works when we fall back to dispatcher
// though, as we need to have the SUBS flag set in the end. So with block linking in the mix,
// I don't think this gives us that much benefit.
void Jit::WriteExit(u32 destination, int exit_num)
{
	WriteDownCount(); 
	//If nobody has taken care of this yet (this can be removed when all branches are done)
	JitBlock *b = js.curBlock;
	b->exitAddress[exit_num] = destination;
	b->exitPtrs[exit_num] = GetWritableCodePtr();

	// Link opportunity!
	int block = blocks.GetBlockNumberFromStartAddress(destination);
	if (block >= 0 && jo.enableBlocklink) {
		// It exists! Joy of joy!
		B(blocks.GetBlock(block)->checkedEntry);
		b->linkStatus[exit_num] = true;
	} else {
		MOVI2R(SREG, destination);
		B((const void *)dispatcherPCInR0);	
	}
}

void Jit::WriteExitDestInR(PPCReg Reg) 
{
	//Break();
	MovToPC(Reg);
	WriteDownCount();
	// TODO: shouldn't need an indirect branch here...
	B((const void *)dispatcher);
}

void Jit::WriteSyscallExit()
{
	WriteDownCount();
	B((const void *)dispatcherCheckCoreState);
}

MIPSOpcode Jit::GetOriginalOp(MIPSOpcode op) {
	JitBlockCache *bc = GetBlockCache();
	int block_num = bc->GetBlockNumberFromEmuHackOp(op, true);
	if (block_num >= 0) {
		return bc->GetOriginalFirstOp(block_num);
	} else {
		return op;
	}
}

}
#endif
