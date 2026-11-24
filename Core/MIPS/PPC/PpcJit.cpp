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

#include "PpcRegCache.h"
#include "ppcEmitter.h"
#include "PpcJit.h"

#include <malloc.h>
#ifdef __wiiu__
#include <wiiu/os/time.h>
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

// Diagnostics: count how many ops fall back to the interpreter (per opcode name, at compile time).
static std::map<std::string, int> g_fallbackCounts;
static int g_compiledBlocks = 0;
static int g_cacheClears = 0;
static u64 g_compileTicks = 0;
static u64 g_firstCompileTick = 0;

static void LogJitStats(const char *reason, size_t spaceLeft, size_t regionSize) {
	NOTICE_LOG(JIT, "JIT stats [%s]: blocks=%d clears=%d used=%d/%d bytes compile=%d ms of %d ms wall",
		reason, g_compiledBlocks, g_cacheClears, (int)(regionSize - spaceLeft), (int)regionSize,
		(int)ticks_to_ms(g_compileTicks), g_firstCompileTick ? (int)ticks_to_ms(OSGetSystemTime() - g_firstCompileTick) : 0);
	std::vector<std::pair<int, std::string>> sorted;
	for (auto &it : g_fallbackCounts)
		sorted.push_back(std::make_pair(it.second, it.first));
	std::sort(sorted.begin(), sorted.end(), [](const std::pair<int, std::string> &a, const std::pair<int, std::string> &b) {
		return a.first > b.first;
	});
	std::string line;
	for (size_t i = 0; i < sorted.size() && i < 25; i++) {
		char buf[64];
		snprintf(buf, sizeof(buf), "%s=%d ", sorted[i].second.c_str(), sorted[i].first);
		line += buf;
	}
	NOTICE_LOG(JIT, "JIT fallbacks: %s", line.c_str());
}

void Jit::CompileDelaySlot(int flags)
{
	// preserve flag around the delay slot! Maybe this is not always necessary on ARM where 
	// we can (mostly) control whether we set the flag or not. Of course, if someone puts an slt in to the
	// delay slot, we're screwed.
	if (flags & DELAYSLOT_SAFE) {
		// Save flags register
		MOVI2R(SREG, (u32)&delaySlotFlagsValue);
		STW(FLAGREG, SREG);
		MFCR(R19);
	}

	js.inDelaySlot = true;
	MIPSOpcode op = Memory::Read_Instruction(js.compilerPC + 4);
	MIPSCompileOp(op, this);
	js.inDelaySlot = false;

	if (flags & DELAYSLOT_FLUSH)
		FlushAll();

	if (flags & DELAYSLOT_SAFE) {
		// Restore flags register
		MOVI2R(SREG, (u32)&delaySlotFlagsValue);
		LWZ(FLAGREG, SREG);
		MTCR(R19);
	}
}

void Jit::Compile(u32 em_address)
{
	PROFILE_THIS_SCOPE("jitc");
	u64 compileStart = OSGetSystemTime();
	if (!g_firstCompileTick)
		g_firstCompileTick = compileStart;
	if (GetSpaceLeft() < 0x4000 || blocks.IsFull())
	{
		g_cacheClears++;
		NOTICE_LOG(JIT, "Clearing JIT cache (spaceLeft=%d, blocksFull=%d)", (int)GetSpaceLeft(), (int)blocks.IsFull());
		LogJitStats("clear", GetSpaceLeft(), region_size);
		ClearCache();
	}

	int block_num = blocks.AllocateBlock(em_address);
	JitBlock *b = blocks.GetBlock(block_num);
	DoJit(em_address, b);
	blocks.FinalizeBlock(block_num, jo.enableBlocklink);
	if ((++g_compiledBlocks % 4000) == 0) {
		LogJitStats("periodic", GetSpaceLeft(), region_size);
	}

	// Drat.  The VFPU hit an uneaten prefix at the end of a block.
	if (js.startDefaultPrefix && js.MayHavePrefix())
	{
		js.startDefaultPrefix = false;
		// Our assumptions are all wrong so it's clean-slate time.
		ClearCache();

		// Let's try that one more time.  We won't get back here because we toggled the value.
		Compile(em_address);
	}
	g_compileTicks += OSGetSystemTime() - compileStart;
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
	emit.MOVI2R(R3, originalAddress);
	emit.STW(R0, CTXREG, offsetof(MIPSState, pc));
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
	// don't know if the result is correct
	int theDowncount = js.downcountAmount + offset;
	if (jo.downcountInRegister) {		
		// DCNTREG = DCNTREG - theDowncount;
		MOVI2R(SREG, theDowncount);	
		SUBF(DCNTREG, SREG, DCNTREG, 1);
		STW(DCNTREG, CTXREG, offsetof(MIPSState, downcount));
	} else {
		// DCNTREG = MIPSState->downcount - theDowncount;
		MOVI2R(SREG, theDowncount);	
		LWZ(DCNTREG, CTXREG, offsetof(MIPSState, downcount));
		SUBF(DCNTREG, SREG, DCNTREG, 1);
		STW(DCNTREG, CTXREG, offsetof(MIPSState, downcount));
	}

	//QuickCallFunction(ShowDownCount);

	CMPI(DCNTREG, 0);
}

void Jit::Comp_Generic(MIPSOpcode op) {
	{
		const char *name = MIPSGetName(op);
		g_fallbackCounts[name ? name : "?"]++;
	}
	FlushAll();

	// basic jit !!
	MIPSInterpretFunc func = MIPSGetInterpretFunc(op);
	if (func)
	{
		// Save mips PC and cycles
		SaveDowncount(DCNTREG);
		MOVI2R(SREG, js.compilerPC);
		MovToPC(SREG);

		// call interpreted function
		MOVI2R(R3, op.encoding);
#ifdef __wiiu__
		// R3 is expected to contain the address of the 4-byte MIPSOpcode struct ...
		STW(R3, R1, 20);
		ADDI(R3, R1, 20);
#endif
		QuickCallFunction((void *)func);

		// restore pc and cycles
		RestoreDowncount(DCNTREG);
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
	// None of the code of this function is relevant so we'll just
	// call the replacement and move RA to PC.
	int replacementFunc = op & 0xFFFFFFF;


	// We could even do this in the jal that is branching to the function
	// but having the op is necessary for the interpreter anyway.
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
}

void Jit::InvalidateCache() {
	blocks.Clear();
}

void Jit::InvalidateCacheAt(u32 em_address, int length) {
	blocks.InvalidateICache(em_address, length);
}

Jit::Jit(MIPSState *mips) : blocks(mips, this), gpr(mips, &jo),fpr(mips),mips_(mips)
{ 
	blocks.Init();
	gpr.SetEmitter(this);
	fpr.SetEmitter(this);
#ifdef __wiiu__
	// RWX memory in 0x00800000-0x01000000 is shared with Aroma modules; use as much as is available (max 8 MB).
	{
		size_t maxSz = GetMaxExecutableAllocSize() & ~(size_t)0xFFFF;
		if (maxSz > 8 * 1024 * 1024)
			maxSz = 8 * 1024 * 1024;
		for (int sz = (int)maxSz; sz >= 512 * 1024; sz -= 64 * 1024) {
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
#ifdef _XBOX
	// force stack alinement
	//_alloca(16*1024);
#endif
	
	// Run the compiled code
	((void (*)())enterCode)();
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
