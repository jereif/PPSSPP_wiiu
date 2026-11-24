
#include "ppsspp_config.h"
#if PPSSPP_ARCH(POWERPC)

#include "Common/Serialize/Serializer.h"
#include "Core/Core.h"
#include "Core/CoreTiming.h"
#include "Core/MemMap.h"
#include "Core/MIPS/MIPS.h"
#include "Core/MIPS/MIPSCodeUtils.h"
#include "Core/MIPS/MIPSInt.h"
#include "Core/MIPS/MIPSTables.h"
#include "Core/MIPS/JitCommon/JitCommon.h"

#include "PpcRegCache.h"
#include "ppcEmitter.h"
#include "PpcJit.h"

//#include <ppcintrinsics.h>

using namespace PpcGen;

extern volatile CoreState coreState;

namespace MIPSComp
{
static int dontLogBlocks = 20;
static int logBlocks = 40;

const u8 *Jit::DoJit(u32 em_address, JitBlock *b)
{
	js.cancel = false;
	js.blockStart = js.compilerPC = mips_->pc;
	js.downcountAmount = 0;
	js.curBlock = b;
	js.compiling = true;
	js.inDelaySlot = false;
	js.PrefixStart();

	// We add a check before the block, used when entering from a linked block.
	// Callers (WriteExit / LinkBlock) always run WriteDownCount() right before jumping here,
	// so CR0 already reflects DCNTREG vs 0. Keep >= 16 bytes for UnlinkBlock().
	b->checkedEntry = (u8*)GetCodePtr();
	FixupBranch skipOuter = B_Cond(_BGE);
	MOVI2R(SREG, js.blockStart);
	B((const void *)outerLoopPCInR0);
	while (GetCodePtr() - b->checkedEntry < 16)
		ORI(R0, R0, 0);
	SetJumpTarget(skipOuter);

	b->normalEntry = GetCodePtr();
	// TODO: this needs work
	MIPSAnalyst::AnalysisResults analysis; // = MIPSAnalyst::Analyze(em_address);

	gpr.Start(analysis);
	fpr.Start(analysis);

	int numInstructions = 0;
	int cycles = 0;
	int partialFlushOffset = 0;
	if (logBlocks > 0) logBlocks--;
	if (dontLogBlocks > 0) dontLogBlocks--;

// #define LOGASM
#ifdef LOGASM
	char temp[256];
#endif
	while (js.compiling)
	{
		gpr.SetCompilerPC(js.compilerPC);  // Let it know for log messages
		fpr.SetCompilerPC(js.compilerPC);
		MIPSOpcode inst = Memory::Read_Instruction(js.compilerPC);
		js.downcountAmount += MIPSGetInstructionCycleEstimate(inst);

		MIPSCompileOp(inst, this);
	
		js.compilerPC += 4;
		numInstructions++;
	}
#ifdef LOGASM
	if (logBlocks > 0 && dontLogBlocks == 0) {
		for (u32 cpc = em_address; cpc != js.compilerPC + 4; cpc += 4) {
			MIPSDisAsm(Memory::Read_Instruction(cpc), cpc, temp, true);
			INFO_LOG(JIT, "M: %08x   %s", cpc, temp);
		}
	}
#endif

	b->codeSize = GetCodePtr() - b->normalEntry;

#if defined(LOGASM) && defined(__wiiu__)
	if (logBlocks > 0 && dontLogBlocks == 0) {
		INFO_LOG(JIT, "=============== ARM ===============");
		DisassemblePPC(b->normalEntry, GetCodePtr() - b->normalEntry);
	}
#endif	
	//DumpJit();

	// Don't forget to zap the instruction cache!
	FlushIcache();

	b->originalSize = numInstructions;
	return b->normalEntry;
}

void Jit::DumpJit() {
#if defined(_XBOX) || defined(__wiiu__)
#ifdef _XBOX
	const char* filename = "game:\\jit.bin";
#else
	const char* filename = "sd:/jit.bin";
#endif
	u32 len = (u32)GetCodePtr() - (u32)GetBasePtr();
	FILE * fd;
	fd = fopen(filename, "wb");
	fwrite(GetBasePtr(), len, 1, fd);
	fclose(fd);
#endif
}

void Jit::GenerateFixedCode() {
	enterCode = AlignCode16();
	
	INFO_LOG(JIT, "Base: %08x", (u32)Memory::base);
	INFO_LOG(JIT, "enterCode: 0x%08x", (u32)enterCode);
	INFO_LOG(JIT, "GetBasePtr: 0x%08x", (u32)GetBasePtr());

	Prologue();
	
	// Map fixed register
	MOVI2R(BASEREG,	(u32)Memory::base);
	MOVI2R(CTXREG,	(u32)mips_);
	MOVI2R(CODEREG, (u32)GetBasePtr());

	// Update downcount reg value from memory
	RestoreDowncount(DCNTREG);

	// SREG = mips->pc
	MovFromPC(SREG);

	// Keep current location, TODO rename it, outerLoopPCInR0 to outerLoopPCInR3 ?? 
	outerLoopPCInR0 = GetCodePtr();
	
	// mips->pc = SREG
	MovToPC(SREG);

	// Keep current location
	outerLoop = GetCodePtr();

	// Jit loop
	// {
		// Save downcount reg value to memory
		SaveDowncount(DCNTREG);
		// Call CoreTiming::Advance() => update donwcount
		QuickCallFunction((void *)&CoreTiming::Advance);
		// Update downcount reg value from memory
		RestoreDowncount(DCNTREG);

		// branch to skipToRealDispatch
		FixupBranch skipToRealDispatch = B(); //skip the sync and compare first time

		// Keep current location dispatcherCheckCoreState:
		dispatcherCheckCoreState = GetCodePtr();

		// The result of slice decrementation should be in flags if somebody jumped here
		// IMPORTANT - We jump on negative, not carry!!!
		// branch to bailCoreState: (jump if(what ??) negative )
		FixupBranch bailCoreState = BLT(); // BLT ???

		// SREG = coreState
		MOVI2R(SREG, (u32)&coreState);
		// Compare coreState and CORE_RUNNING
		LWZ(SREG, SREG); // SREG = *SREG
		CMPI(SREG, 0); // compare 0(CORE_RUNNING) and CR0

		// branch to badCoreState: (jump if coreState != CORE_RUNNING)
		FixupBranch badCoreState = BNE(); 

		// branch to skipToRealDispatch2:
		FixupBranch skipToRealDispatch2 = B(); //skip the sync and compare first time

		// Keep current location, TODO rename it, outerLoopPCInR0 to outerLoopPCInSREG ?? 
		dispatcherPCInR0 = GetCodePtr();

		// mips->pc = SREG
		MovToPC(SREG);

		// At this point : flags = EQ. Fine for the next check, no need to jump over it.
		// label dispatcher:
		dispatcher = GetCodePtr(); 

		// {
			// The result of slice decrementation should be in flags if somebody jumped here
			// IMPORTANT - We jump on negative, not carry!!!
			// label bail:
			// arm B_CC(CC_MI);
			FixupBranch bail = BLT();

			// label skipToRealDispatch:
			SetJumpTarget(skipToRealDispatch);

			// label skipToRealDispatch2:
			SetJumpTarget(skipToRealDispatch2);

			// Keep current location
			dispatcherNoCheck = GetCodePtr();
			
			// read op
			// R3 = mips->pc & Memory::MEMVIEW32_MASK
			LWZ(R3, CTXREG, offsetof(MIPSState, pc));
			// & Memory::MEMVIEW32_MASK
			RLWINM(R3, R3, 0, 2, 31);

			// R3 = memory::base[r3] (byte-swapped);
			LWBRX(R3, BASEREG, R3);

			// R4 = R3 & MIPS_EMUHACK_VALUE_MASK
			RLWINM(R4, R3, 0, 8, 31);

			// compare top 8 bits == MIPS_EMUHACK_OPCODE >> 24 (RUNBLOCK only, not CALL_REPLACEMENT)
			SRWI(R3, R3, 24);
			CMPLI(R3, MIPS_EMUHACK_OPCODE >> 24);

			// Branch if func block not found 
			FixupBranch notfound = BNE();

			// {
				// R3 = R4 +  GetBasePtr()
				ADD(R3, R4, CODEREG);
				
				MTCTR(R3);
				BCTR();
			// }

			// label notfound:
			SetJumpTarget(notfound);

			//Ok, no block, let's jit
			// Save downcount reg value to memory
			SaveDowncount(DCNTREG);

			// Exec JitAt => Compile block !
			QuickCallFunction((void *)&JitAt);
						
			// Update downcount reg value from memory
			RestoreDowncount(DCNTREG);
			CMPI(DCNTREG, 0);

			// Check downcount/coreState and dispatch next block (needed when Compile interprets a cold block).
			B(dispatcherCheckCoreState);
		// }

		// label bail:
		SetJumpTarget(bail);

		// label bailCoreState:
		SetJumpTarget(bailCoreState);

		// Compare coreState and CORE_RUNNING
		MOVI2R(SREG, (u32)&coreState);
		LWZ(SREG, SREG); // SREG = *SREG => SREG = coreState
		CMPLI(SREG, 0); // compare 0(CORE_RUNNING) and corestate

		BEQ(outerLoop);
	// }

	// badCoreState label:
	SetJumpTarget(badCoreState);

	// Keep current location
	breakpointBailout = GetCodePtr();

	// mips->downcount = DCNTREG
	SaveDowncount(DCNTREG);

	Epilogue();

	// Go back to caller
	BLR();

	// Fast leaf helper: FPR4 = 1.0 / sqrt(FPR4) using PPC750 frsqrte + 3 double Newton-Raphson steps.
	// Uses scratch FPR4, FPR5, FPR9, FPR10, FPR11, CR0, and R6 (via MOVI2F); preserves FPR14..31 and GPRs.
	ppcInvSqrt = AlignCode16();
	FMR(FPR5, FPR4);
	FRSQRTE(FPR4, FPR4); // handles <= 0 and NaN natively; gives +0.0 for +Inf
	MOVI2F(FPR9, 0.0f);
	FCMPU(0, FPR5, FPR9);
	FixupBranch invPos = B_Cond(_BGT);
	BLR();
	SetJumpTarget(invPos);
	FCMPU(0, FPR4, FPR9); // skip NR if +Inf (frsqrte(+Inf) == +0.0)
	FixupBranch invPosEst = B_Cond(_BGT);
	BLR();
	SetJumpTarget(invPosEst);
	MOVI2F(FPR9, 0.5f);
	MOVI2F(FPR10, 1.5f);
	FMUL(FPR11, FPR5, FPR9);
	for (int i = 0; i < 3; i++) {
		FMUL(FPR9, FPR4, FPR4);
		FMUL(FPR9, FPR9, FPR11);
		FSUB(FPR9, FPR10, FPR9);
		FMUL(FPR4, FPR4, FPR9);
	}
	BLR();

	// Fast leaf helper: FPR4 = sqrt(FPR4) = x * invSqrt(x) for finite positive x.
	ppcSqrt = AlignCode16();
	FMR(FPR5, FPR4);
	FRSQRTE(FPR4, FPR4); // QNaN if x < 0
	MOVI2F(FPR9, 0.0f);
	FCMPU(0, FPR5, FPR9);
	FixupBranch sqrtPos = B_Cond(_BGT);
	FixupBranch sqrtNeg = B_Cond(_BLT);
	FMR(FPR4, FPR5); // +-0.0 or NaN -> preserve x
	SetJumpTarget(sqrtNeg);
	BLR();
	SetJumpTarget(sqrtPos);
	FCMPU(0, FPR4, FPR9); // skip NR if +Inf -> return +Inf
	FixupBranch sqrtPosEst = B_Cond(_BGT);
	FMR(FPR4, FPR5);
	BLR();
	SetJumpTarget(sqrtPosEst);
	MOVI2F(FPR9, 0.5f);
	MOVI2F(FPR10, 1.5f);
	FMUL(FPR11, FPR5, FPR9);
	for (int i = 0; i < 3; i++) {
		FMUL(FPR9, FPR4, FPR4);
		FMUL(FPR9, FPR9, FPR11);
		FSUB(FPR9, FPR10, FPR9);
		FMUL(FPR4, FPR4, FPR9);
	}
	FMUL(FPR4, FPR5, FPR4);
	BLR();

	// Fast leaf helper: FPR1 = vfpu_sin(FPR1) = sin(pi/2 * x), preserves R14..31 and FPR14..31.
	static float (*const pPpcVfpuSin)(float) = [](float x) -> float {
		if (!(x > -1e6f && x < 1e6f))
			return sinf(x * 1.5707963268f);
		int q = (int)(x >= 0.0f ? (x * 0.25f + 0.5f) : (x * 0.25f - 0.5f));
		x -= (float)q * 4.0f;
		if (x > 1.0f) x = 2.0f - x;
		else if (x < -1.0f) x = -2.0f - x;
		float x2 = x * x;
		return x * (1.5707963268f + x2 * (-0.6459640975f + x2 * (0.0796926262f + x2 * (-0.0046817541f + x2 * 0.0001604412f))));
	};
	static float (*const pPpcVfpuCos)(float) = [](float x) -> float {
		return pPpcVfpuSin(x + 1.0f);
	};

	ppcVSin = AlignCode16();
	MFLR(R0);
	STW(R0, R1, 24);
	QuickCallFunction((void *)pPpcVfpuSin);
	LWZ(R0, R1, 24);
	MTLR(R0);
	BLR();

	ppcVCos = AlignCode16();
	MFLR(R0);
	STW(R0, R1, 24);
	QuickCallFunction((void *)pPpcVfpuCos);
	LWZ(R0, R1, 24);
	MTLR(R0);
	BLR();

	static float (*const pPpcVfpuAsin)(float) = [](float x) -> float {
		return asinf(x) * 0.63661977236758134308f; // 1 / (pi/2)
	};
	ppcVAsin = AlignCode16();
	MFLR(R0);
	STW(R0, R1, 24);
	QuickCallFunction((void *)pPpcVfpuAsin);
	LWZ(R0, R1, 24);
	MTLR(R0);
	BLR();

	// Shared trampoline for Comp_Generic: R3 = op.encoding, R4 = compilerPC, R5 = interpret func.
	extern void PpcExecGeneric(u32 opEncoding, MIPSInterpretFunc func);
	compGenericTrampoline = AlignCode16();
	MFLR(R0);
	STW(R0, R1, 24);
	SaveDowncount(DCNTREG);
	MovToPC(R4);
	MR(R4, R5);
	QuickCallFunction((void *)&PpcExecGeneric);
	RestoreDowncount(DCNTREG);
	LWZ(R0, R1, 24);
	MTLR(R0);
	BLR();

	// Don't forget to zap the instruction cache!
	FlushIcache();
}

}

#endif
