#include "ppsspp_config.h"
#if PPSSPP_ARCH(POWERPC)

#include "Common/Math/math_util.h"
#include "Common/Serialize/Serializer.h"
#include "Core/MemMap.h"
#include "Core/Config.h"
#include "Core/Core.h"
#include "Core/CoreTiming.h"
#include "Core/MIPS/MIPS.h"
#include "Core/MIPS/MIPSCodeUtils.h"
#include "Core/MIPS/MIPSInt.h"
#include "Core/MIPS/MIPSTables.h"
#include "Core/MIPS/MIPSDebugInterface.h"

#include "PpcRegCache.h"
#include "ppcEmitter.h"
#include "PpcJit.h"

const bool disablePrefixes = false;

// All functions should have CONDITIONAL_DISABLE, so we can narrow things down to a file quickly.
// Currently known non working ones should have DISABLE.

// #define CONDITIONAL_DISABLE { fpr.ReleaseSpillLocksAndDiscardTemps(); Comp_Generic(op); return; }
#define CONDITIONAL_DISABLE ;
#define DISABLE { fpr.ReleaseSpillLocksAndDiscardTemps(); Comp_Generic(op); return; }

#define _RS MIPS_GET_RS(op)
#define _RT MIPS_GET_RT(op)
#define _RD MIPS_GET_RD(op)
#define _FS MIPS_GET_FS(op)
#define _FT MIPS_GET_FT(op)
#define _FD MIPS_GET_FD(op)
#define _SA MIPS_GET_SA(op)
#define _POS  ((op>> 6) & 0x1F)
#define _SIZE ((op>>11) & 0x1F)
#define _IMM16 (signed short)(op & 0xFFFF)
#define _IMM26 (op & 0x03FFFFFF)

using namespace PpcGen;
using namespace PpcJitConstants;

// #define USE_VMX128

namespace MIPSComp
{
	// Vector regs can overlap in all sorts of swizzled ways.
	// This does allow a single overlap in sregs[i].
	static bool IsOverlapSafeAllowS(int dreg, int di, int sn, u8 sregs[], int tn = 0, u8 tregs[] = NULL)
	{
		for (int i = 0; i < sn; ++i)
		{
			if (sregs[i] == dreg && i != di)
				return false;
		}
		for (int i = 0; i < tn; ++i)
		{
			if (tregs[i] == dreg)
				return false;
		}

		// Hurray, no overlap, we can write directly.
		return true;
	}

	static bool IsOverlapSafe(int dreg, int di, int sn, u8 sregs[], int tn = 0, u8 tregs[] = NULL)
	{
		return IsOverlapSafeAllowS(dreg, di, sn, sregs, tn, tregs) && sregs[di] != dreg;
	}

	void Jit::ApplyPrefixST(u8 *vregs, u32 prefix, VectorSize sz) {
		if (prefix == 0xE4) return;

		int n = GetNumVectorElements(sz);
		u8 origV[4];
		static const float constantArray[8] = {0.f, 1.f, 2.f, 0.5f, 3.f, 1.f/3.f, 0.25f, 1.f/6.f};

		for (int i = 0; i < n; i++)
			origV[i] = vregs[i];

		for (int i = 0; i < n; i++)
		{
			int regnum = (prefix >> (i*2)) & 3;
			int abs    = (prefix >> (8+i)) & 1;
			int negate = (prefix >> (16+i)) & 1;
			int constants = (prefix >> (12+i)) & 1;

			// Unchanged, hurray.
			if (!constants && regnum == i && !abs && !negate)
				continue;

			// This puts the value into a temp reg, so we won't write the modified value back.
			vregs[i] = fpr.GetTempV();
			if (!constants) {
				fpr.MapDirtyInV(vregs[i], origV[regnum]);
				fpr.SpillLockV(vregs[i]);

				// Prefix may say "z, z, z, z" but if this is a pair, we force to x.
				// TODO: But some ops seem to use const 0 instead?
				if (regnum >= n) {
					WARN_LOG(CPU, "JIT: Invalid VFPU swizzle: %08x : %d / %d at PC = %08x (%s)", prefix, regnum, n, js.compilerPC, currentDebugMIPS->disasm(js.compilerPC, 0));
					regnum = 0;
				}
				
				if (abs) {
					FABS(fpr.V(vregs[i]), fpr.V(origV[regnum]));
					if (negate)
						FNEG(fpr.V(vregs[i]), fpr.V(vregs[i]));
				} else {
					if (negate)
						FNEG(fpr.V(vregs[i]), fpr.V(origV[regnum]));
					else
						FMR(fpr.V(vregs[i]), fpr.V(origV[regnum]));
				}
			} else {
				fpr.MapRegV(vregs[i], MAP_DIRTY | MAP_NOINIT);
				fpr.SpillLockV(vregs[i]);
				MOVI2F(fpr.V(vregs[i]), constantArray[regnum + (abs<<2)], negate);
			}
		}
	}

	void Jit::GetVectorRegsPrefixD(u8 *regs, VectorSize sz, int vectorReg) {
		_assert_(js.prefixDFlag & PpcJitState::PREFIX_KNOWN);

		GetVectorRegs(regs, sz, vectorReg);
		if (js.prefixD == 0)
			return;

		int n = GetNumVectorElements(sz);
		for (int i = 0; i < n; i++) {
			// Hopefully this is rare, we'll just write it into a reg we drop.
			if (js.VfpuWriteMask(i))
				regs[i] = fpr.GetTempV();
		}
	}

	void Jit::ApplyPrefixD(const u8 *vregs, VectorSize sz) {
		_assert_(js.prefixDFlag & PpcJitState::PREFIX_KNOWN);
		if (!js.prefixD) return;

		int n = GetNumVectorElements(sz);
		for (int i = 0; i < n; i++) 	{
			if (js.VfpuWriteMask(i))
				continue;

			// TODO: These clampers are wrong - put this into google
			// and look at the plot:   abs(x) - abs(x-0.5) + 0.5
			// It's too steep.

			// Also, they mishandle NaN and Inf.
			int sat = (js.prefixD >> (i * 2)) & 3;
			if (sat == 1) {
				fpr.MapRegV(vregs[i], MAP_DIRTY);
				
				MOVI2F(FPR6, 0.0f);
				MOVI2F(FPR7, 1.0f);

				FMAX(fpr.V(vregs[i]), fpr.V(vregs[i]), FPR6);
				FMIN(fpr.V(vregs[i]), fpr.V(vregs[i]), FPR7);
			} else if (sat == 3) {
				fpr.MapRegV(vregs[i], MAP_DIRTY);

				MOVI2F(FPR6, -1.0f);
				MOVI2F(FPR7, 1.0f);

				FMAX(fpr.V(vregs[i]), fpr.V(vregs[i]), FPR6);
				FMIN(fpr.V(vregs[i]), fpr.V(vregs[i]), FPR7);
			}
		}
	}

	void Jit::Comp_SV(MIPSOpcode op) {
		CONDITIONAL_DISABLE;

		s32 imm = (signed short)(op&0xFFFC);
		int vt = ((op >> 16) & 0x1f) | ((op & 3) << 5);
		int rs = _RS;

		if (!g_Config.bFastMemory) {
			DISABLE;
		}

		bool doCheck = false;
		switch (op >> 26)
		{
		case 50: //lv.s  // VI(vt) = Memory::Read_U32(addr);
			{
				// CC might be set by slow path below, so load regs first.
				fpr.MapRegV(vt, MAP_DIRTY | MAP_NOINIT);
				if (gpr.IsImm(rs)) {
					u32 addr = (imm + gpr.GetImm(rs)) & 0x3FFFFFFF;
					MOVI2R(SREG, addr);
				} else {
					gpr.MapReg(rs);					
					SetRegToEffectiveAddress(SREG, rs, imm);
				}

				LoadFloatSwap(fpr.V(vt), BASEREG, SREG);
			}
			break;

		case 58: //sv.s   // Memory::Write_U32(VI(vt), addr);
			{
				// CC might be set by slow path below, so load regs first.
				fpr.MapRegV(vt);
				if (gpr.IsImm(rs)) {
					u32 addr = (imm + gpr.GetImm(rs)) & 0x3FFFFFFF;
					MOVI2R(SREG, addr);
				} else {
					gpr.MapReg(rs);
					SetRegToEffectiveAddress(SREG, rs, imm);
				}
				SaveFloatSwap(fpr.V(vt), BASEREG, SREG);
			}
			break;


		default:
			DISABLE;
		}
	}

	void Jit::Comp_SVQ(MIPSOpcode op) {
		// Comp_Generic(op);
		CONDITIONAL_DISABLE;

		int imm = (signed short)(op&0xFFFC);
		int vt = (((op >> 16) & 0x1f)) | ((op&1) << 5);
		int rs = _RS;

		if (!g_Config.bFastMemory) {
			DISABLE;
		}

		bool doCheck = false;
		switch (op >> 26)
		{
		case 54: //lv.q
			{
				u8 vregs[4];
				GetVectorRegs(vregs, V_Quad, vt);
				fpr.MapRegsAndSpillLockV(vregs, V_Quad, MAP_DIRTY | MAP_NOINIT);

				if (gpr.IsImm(rs)) {
					u32 addr = (imm + gpr.GetImm(rs)) & 0x3FFFFFFF;
					MOVI2R(SREG, addr + (u32)Memory::base);
				} else {
					gpr.MapReg(rs);
					SetRegToEffectiveAddress(SREG, rs, imm);
					ADD(SREG, SREG, BASEREG);
				}

				// Pipeline all 4 byte-reversed loads and stack stores before LFS to avoid 4x store-forwarding stalls.
				LI(R7, 4);
				LI(R8, 8);
				LI(R9, 12);
				LWBRX(R6, R0, SREG);
				LWBRX(R7, SREG, R7);
				LWBRX(R8, SREG, R8);
				LWBRX(R9, SREG, R9);
				STW(R6, R1, 16);
				STW(R7, R1, 20);
				STW(R8, R1, 24);
				STW(R9, R1, 28);
				LFS(fpr.V(vregs[0]), R1, 16);
				LFS(fpr.V(vregs[1]), R1, 20);
				LFS(fpr.V(vregs[2]), R1, 24);
				LFS(fpr.V(vregs[3]), R1, 28);
			}
			break;

		case 62: //sv.q
			{
				// CC might be set by slow path below, so load regs first.
				u8 vregs[4];
				GetVectorRegs(vregs, V_Quad, vt);
				fpr.MapRegsAndSpillLockV(vregs, V_Quad, 0);

				if (gpr.IsImm(rs)) {
					u32 addr = (imm + gpr.GetImm(rs)) & 0x3FFFFFFF;
					MOVI2R(SREG, addr + (u32)Memory::base);
				} else {
					gpr.MapReg(rs);
					SetRegToEffectiveAddress(SREG, rs, imm);
					ADD(SREG, SREG, BASEREG);
				}

				// Pipeline all 4 float stores to stack before integer loads and STWBRX to avoid 4x forwarding stalls.
				SFS(fpr.V(vregs[0]), R1, 16);
				SFS(fpr.V(vregs[1]), R1, 20);
				SFS(fpr.V(vregs[2]), R1, 24);
				SFS(fpr.V(vregs[3]), R1, 28);
				LWZ(R6, R1, 16);
				LWZ(R7, R1, 20);
				LWZ(R8, R1, 24);
				LWZ(R9, R1, 28);
				LI(R10, 4);
				LI(R11, 8);
				LI(R12, 12);
				STWBRX(R6, R0, SREG);
				STWBRX(R7, SREG, R10);
				STWBRX(R8, SREG, R11);
				STWBRX(R9, SREG, R12);
			}
			break;

		default:
			DISABLE;
			break;
		}
		fpr.ReleaseSpillLocksAndDiscardTemps();
	}

	void Jit::Comp_VPFX(MIPSOpcode op) {
		CONDITIONAL_DISABLE;
		int data = op & 0xFFFFF;
		int regnum = (op >> 24) & 3;
		switch (regnum) {
		case 0:  // S
			js.prefixS = data;
			js.prefixSFlag = PpcJitState::PREFIX_KNOWN_DIRTY;
			break;
		case 1:  // T
			js.prefixT = data;
			js.prefixTFlag = PpcJitState::PREFIX_KNOWN_DIRTY;
			break;
		case 2:  // D
			js.prefixD = data;
			js.prefixDFlag = PpcJitState::PREFIX_KNOWN_DIRTY;
			break;
		default:
			ERROR_LOG(CPU, "VPFX - bad regnum %i : data=%08x", regnum, data);
			break;
		}
	}

	void Jit::Comp_VVectorInit(MIPSOpcode op) {
		CONDITIONAL_DISABLE;

		// WARNING: No prefix support!
		if (js.HasUnknownPrefix() || disablePrefixes) {
			DISABLE;
		}

		switch ((op >> 16) & 0xF)
		{
		case 6: // v=zeros; break;  //vzero
			MOVI2F(FPR5, 0.0f);
			break;
		case 7: // v=ones; break;   //vone
			MOVI2F(FPR5, 1.0f);
			break;
		default:
			DISABLE;
			break;
		}

		VectorSize sz = GetVecSize(op);
		int n = GetNumVectorElements(sz);

		u8 dregs[4];
		GetVectorRegsPrefixD(dregs, sz, _VD);
		fpr.MapRegsAndSpillLockV(dregs, sz, MAP_NOINIT | MAP_DIRTY);

		for (int i = 0; i < n; ++i)
			FMR(fpr.V(dregs[i]), FPR5);

		ApplyPrefixD(dregs, sz);

		fpr.ReleaseSpillLocksAndDiscardTemps();
	}

	void Jit::Comp_VMatrixInit(MIPSOpcode op) {
		CONDITIONAL_DISABLE;

		if (js.HasUnknownPrefix() || disablePrefixes) {
			DISABLE;
		}

		MatrixSize sz = GetMtxSize(op);
		int n = GetMatrixSide(sz);

		u8 dregs[16];
		GetMatrixRegs(dregs, sz, _VD);

		switch ((op >> 16) & 0xF) {
		case 3: // vmidt
			MOVI2F(FPR6, 0.0f);
			MOVI2F(FPR7, 1.0f);
			for (int a = 0; a < n; a++) {
				for (int b = 0; b < n; b++) {
					fpr.MapRegV(dregs[a * 4 + b], MAP_DIRTY | MAP_NOINIT);
					FMR(fpr.V(dregs[a * 4 + b]), a == b ? FPR7 : FPR6);
				}
			}
			break;
		case 6: // vmzero
			MOVI2F(FPR6, 0.0f);
			for (int a = 0; a < n; a++) {
				for (int b = 0; b < n; b++) {
					fpr.MapRegV(dregs[a * 4 + b], MAP_DIRTY | MAP_NOINIT);
					FMR(fpr.V(dregs[a * 4 + b]), FPR6);
				}
			}
			break;
		case 7: // vmone
			MOVI2F(FPR7, 1.0f);
			for (int a = 0; a < n; a++) {
				for (int b = 0; b < n; b++) {
					fpr.MapRegV(dregs[a * 4 + b], MAP_DIRTY | MAP_NOINIT);
					FMR(fpr.V(dregs[a * 4 + b]), FPR7);
				}
			}
			break;
		}

		fpr.ReleaseSpillLocksAndDiscardTemps();
	}

	void Jit::Comp_VDot(MIPSOpcode op) {
		CONDITIONAL_DISABLE;
		if (js.HasUnknownPrefix() || disablePrefixes) {
			DISABLE;
		}

		int vd = _VD;
		int vs = _VS;
		int vt = _VT;
		VectorSize sz = GetVecSize(op);

		// TODO: Force read one of them into regs? probably not.
		u8 sregs[4], tregs[4], dregs[1];
		GetVectorRegsPrefixS(sregs, sz, vs);
		GetVectorRegsPrefixT(tregs, sz, vt);
		GetVectorRegsPrefixD(dregs, V_Single, vd);

		// TODO: applyprefixST here somehow (shuffle, etc...)
		fpr.MapRegsAndSpillLockV(sregs, sz, 0);
		fpr.MapRegsAndSpillLockV(tregs, sz, 0);
		FMULS(FPR6, fpr.V(sregs[0]), fpr.V(tregs[0]));

		int n = GetNumVectorElements(sz);
		for (int i = 1; i < n; i++) {
			// sum += s[i]*t[i];
			FMADDS(FPR6, fpr.V(sregs[i]), fpr.V(tregs[i]), FPR6);
		}
		fpr.ReleaseSpillLocksAndDiscardTemps();

		fpr.MapRegV(dregs[0], MAP_NOINIT | MAP_DIRTY);

		// TODO: applyprefixD here somehow (write mask etc..)
		FMR(fpr.V(dregs[0]), FPR6);
		ApplyPrefixD(dregs, V_Single);
		fpr.ReleaseSpillLocksAndDiscardTemps();
	}

	void Jit::Comp_VecDo3(MIPSOpcode op) {
		CONDITIONAL_DISABLE;
		
		if (js.HasUnknownPrefix() || disablePrefixes) {
			DISABLE;
		}

		int vd = _VD;
		int vs = _VS;
		int vt = _VT;

		VectorSize sz = GetVecSize(op);
		int n = GetNumVectorElements(sz);

		u8 sregs[4], tregs[4], dregs[4];
		GetVectorRegsPrefixS(sregs, sz, _VS);
		GetVectorRegsPrefixT(tregs, sz, _VT);
		GetVectorRegsPrefixD(dregs, sz, _VD);

		MIPSReg tempregs[4];
		for (int i = 0; i < n; i++) {
			if (!IsOverlapSafe(dregs[i], i, n, sregs, n, tregs)) {
				tempregs[i] = fpr.GetTempV();
			} else {
				tempregs[i] = dregs[i];
			}
		}

		for (int i = 0; i < n; i++) {
			fpr.MapDirtyInInV(tempregs[i], sregs[i], tregs[i]);
			switch (op >> 26) {
			case 24: //VFPU0
				switch ((op >> 23)&7) {
				case 0: // d[i] = s[i] + t[i]; break; //vadd
					FADDS(fpr.V(tempregs[i]), fpr.V(sregs[i]), fpr.V(tregs[i]));
					break;
				case 1: // d[i] = s[i] - t[i]; break; //vsub
					FSUBS(fpr.V(tempregs[i]), fpr.V(sregs[i]), fpr.V(tregs[i]));
					break;
				case 7: // d[i] = s[i] / t[i]; break; //vdiv
					FDIVS(fpr.V(tempregs[i]), fpr.V(sregs[i]), fpr.V(tregs[i]));
					break;
				default:
					DISABLE;
				}
				break;
			case 25: //VFPU1
				switch ((op >> 23) & 7) {
				case 0: // d[i] = s[i] * t[i]; break; //vmul
					FMULS(fpr.V(tempregs[i]), fpr.V(sregs[i]), fpr.V(tregs[i]));
					break;
				default:
					DISABLE;
				}
				break;
			case 27: //VFPU3				
				// DISABLE
		
				switch ((op >> 23) & 7)	{
				case 2:  // vmin
					FMIN(fpr.V(tempregs[i]), fpr.V(sregs[i]), fpr.V(tregs[i]));
					break;
				case 3:  // vmax
					FMAX(fpr.V(tempregs[i]), fpr.V(sregs[i]), fpr.V(tregs[i]));
					break;
				case 6:  // vsge
					// DISABLE;  // pending testing
					MOVI2F(FPR6, 1.0f);
					MOVI2F(FPR7, 0.0f);
					FSUBS(FPR8, fpr.V(sregs[i]), fpr.V(tregs[i]));
					FSEL(fpr.V(tempregs[i]), FPR8, FPR6, FPR7);
					break;
				case 7:  // vslt
					// DISABLE;  // pending testing
					MOVI2F(FPR6, 1.0f);
					MOVI2F(FPR7, 0.0f);
					FSUBS(FPR8, fpr.V(sregs[i]), fpr.V(tregs[i]));
					FSEL(fpr.V(tempregs[i]), FPR8, FPR7, FPR6);
					break;
				}
				break;

			default:
				DISABLE;
			}
		}

		for (int i = 0; i < n; i++) {
			if (dregs[i] != tempregs[i]) {
				fpr.MapDirtyInV(dregs[i], tempregs[i]);
				FMR(fpr.V(dregs[i]), fpr.V(tempregs[i]));
			}
		}
		ApplyPrefixD(dregs, sz);
		
		fpr.ReleaseSpillLocksAndDiscardTemps();
	}

	void Jit::Comp_VV2Op(MIPSOpcode op) {
		CONDITIONAL_DISABLE;

		if (js.HasUnknownPrefix() || disablePrefixes) {
			DISABLE;
		}

		// Pre-processing: Eliminate silly no-op VMOVs, common in Wipeout Pure
		if (((op >> 16) & 0x1f) == 0 && _VS == _VD && js.HasNoPrefix()) {
			return;
		}

		VectorSize sz = GetVecSize(op);
		int n = GetNumVectorElements(sz);

		u8 sregs[4], dregs[4];
		GetVectorRegsPrefixS(sregs, sz, _VS);
		GetVectorRegsPrefixD(dregs, sz, _VD);

		MIPSReg tempregs[4];
		for (int i = 0; i < n; ++i) {
			if (!IsOverlapSafe(dregs[i], i, n, sregs)) {
				tempregs[i] = fpr.GetTempV();
			} else {
				tempregs[i] = dregs[i];
			}
		}

		// Warning: sregs[i] and tempxregs[i] may be the same reg.
		// Helps for vmov, hurts for vrcp, etc.
		for (int i = 0; i < n; ++i) {
			switch ((op >> 16) & 0x1f) {
			case 0: // d[i] = s[i]; break; //vmov
				// Probably for swizzle.
				fpr.MapDirtyInV(tempregs[i], sregs[i]);
				FMR(fpr.V(tempregs[i]), fpr.V(sregs[i]));
				break;
			case 1: // d[i] = fabsf(s[i]); break; //vabs
				fpr.MapDirtyInV(tempregs[i], sregs[i]);
				FABS(fpr.V(tempregs[i]), fpr.V(sregs[i]));
				break;
			case 2: // d[i] = -s[i]; break; //vneg
				fpr.MapDirtyInV(tempregs[i], sregs[i]);
				FNEG(fpr.V(tempregs[i]), fpr.V(sregs[i]));
				break;

			/*  These are probably just as broken as the prefix.
			case 4: // if (s[i] < 0) d[i] = 0; else {if(s[i] > 1.0f) d[i] = 1.0f; else d[i] = s[i];} break;    // vsat0
				fpr.MapDirtyInV(tempregs[i], sregs[i]);
				MOVI2F(S0, 0.5f, R0);
				VABS(S1, fpr.V(sregs[i]));                          // S1 = fabs(x)
				VSUB(fpr.V(tempregs[i]), fpr.V(sregs[i]), S0);     // S2 = fabs(x-0.5f) {VABD}
				VABS(fpr.V(tempregs[i]), fpr.V(tempregs[i]));
				VSUB(fpr.V(tempregs[i]), S1, fpr.V(tempregs[i])); // v[i] = S1 - S2 + 0.5f
				VADD(fpr.V(tempregs[i]), fpr.V(tempregs[i]), S0);
				break;
			case 5: // if (s[i] < -1.0f) d[i] = -1.0f; else {if(s[i] > 1.0f) d[i] = 1.0f; else d[i] = s[i];} break;  // vsat1
				fpr.MapDirtyInV(tempregs[i], sregs[i]);
				MOVI2F(S0, 1.0f, R0);
				VABS(S1, fpr.V(sregs[i]));                          // S1 = fabs(x)
				VSUB(fpr.V(tempregs[i]), fpr.V(sregs[i]), S0);     // S2 = fabs(x-1.0f) {VABD}
				VABS(fpr.V(tempregs[i]), fpr.V(tempregs[i]));
				VSUB(fpr.V(tempregs[i]), S1, fpr.V(tempregs[i])); // v[i] = S1 - S2
				break;
				*/

			case 16: // d[i] = 1.0f / s[i]; break; //vrcp
				fpr.MapDirtyInV(tempregs[i], sregs[i]);
				MOVI2F(FPR6, 1.0f);
				FDIVS(fpr.V(tempregs[i]), FPR6, fpr.V(sregs[i]));
				break;
			case 17: // d[i] = 1.0f / sqrtf(s[i]); break; //vrsq
				fpr.MapDirtyInV(tempregs[i], sregs[i]);
#if !PPSSPP_ARCH(PPC750)
				MOVI2F(FPR6, 1.0f);
				FSQRTS(FPR7, fpr.V(sregs[i]));
				FDIVS(fpr.V(tempregs[i]), FPR6, FPR7);
#else
				FMR(FPR4, fpr.V(sregs[i]));
				BL(ppcInvSqrt);
				FRSP(fpr.V(tempregs[i]), FPR4);
#endif
				break;
			case 18: // d[i] = sinf((float)M_PI_2 * s[i]); break; //vsin
				if (js.MayHavePrefix()) {
					DISABLE;
				}
				fpr.MapDirtyInV(tempregs[i], sregs[i]);
				FMR(FPR1, fpr.V(sregs[i]));
				BL(ppcVSin);
				FMR(fpr.V(tempregs[i]), FPR1);
				break;
			case 19: // d[i] = cosf((float)M_PI_2 * s[i]); break; //vcos
				if (js.MayHavePrefix()) {
					DISABLE;
				}
				fpr.MapDirtyInV(tempregs[i], sregs[i]);
				FMR(FPR1, fpr.V(sregs[i]));
				BL(ppcVCos);
				FMR(fpr.V(tempregs[i]), FPR1);
				break;
				break;
			case 20: // d[i] = powf(2.0f, s[i]); break; //vexp2
				DISABLE;
				break;
			case 21: // d[i] = logf(s[i])/log(2.0f); break; //vlog2
				DISABLE;
				break;
			case 22: // d[i] = sqrtf(s[i]); break; //vsqrt
				fpr.MapDirtyInV(tempregs[i], sregs[i]);
#if !PPSSPP_ARCH(PPC750)
				FSQRTS(fpr.V(tempregs[i]), fpr.V(sregs[i]));
#else
				FMR(FPR4, fpr.V(sregs[i]));
				BL(ppcSqrt);
				FRSP(fpr.V(tempregs[i]), FPR4);
#endif
				FABS(fpr.V(tempregs[i]), fpr.V(tempregs[i]));
				break;
			case 23: // d[i] = asinf(s[i] * (float)M_2_PI); break; //vasin
				if (js.MayHavePrefix()) {
					DISABLE;
				}
				fpr.MapDirtyInV(tempregs[i], sregs[i]);
				FMR(FPR1, fpr.V(sregs[i]));
				BL(ppcVAsin);
				FMR(fpr.V(tempregs[i]), FPR1);
				break;
			case 24: // d[i] = -1.0f / s[i]; break; // vnrcp
				fpr.MapDirtyInV(tempregs[i], sregs[i]);
				MOVI2F(FPR6, -1.0f);
				FDIVS(fpr.V(tempregs[i]), FPR6, fpr.V(sregs[i]));
				break;
			case 26: // d[i] = -sinf((float)M_PI_2 * s[i]); break; // vnsin
				DISABLE;
				break;
			case 28: // d[i] = 1.0f / expf(s[i] * (float)M_LOG2E); break; // vrexp2
				DISABLE;
				break;
			default:
				DISABLE;
				break;
			}
		}

		for (int i = 0; i < n; ++i) {
			if (dregs[i] != tempregs[i]) {
				fpr.MapDirtyInV(dregs[i], tempregs[i]);
				FMR(fpr.V(dregs[i]), fpr.V(tempregs[i]));
			}
		}

		ApplyPrefixD(dregs, sz);

		fpr.ReleaseSpillLocksAndDiscardTemps();
	}

	void Jit::Comp_Mftv(MIPSOpcode op) {
		CONDITIONAL_DISABLE;

		int imm = op & 0xFF;
		MIPSGPReg rt = _RT;
		switch ((op >> 21) & 0x1f)
		{
		case 3: //mfv / mfvc
			// rt = 0, imm = 255 appears to be used as a CPU interlock by some games.
			if (rt != 0) {
				if (imm < 128) {  //R(rt) = VI(imm);
					fpr.FlushV(imm);
					gpr.MapReg(rt, MAP_NOINIT | MAP_DIRTY);
					LWZ(gpr.R(rt), CTXREG, fpr.GetMipsRegOffsetV(imm));
				} else if (imm < 128 + VFPU_CTRL_MAX) { //mfvc
					FlushPrefixV();
					gpr.MapReg(rt, MAP_NOINIT | MAP_DIRTY);
					LWZ(gpr.R(rt), CTXREG, offsetof(MIPSState, vfpuCtrl) + 4 * (imm - 128));
				} else {
					//ERROR - maybe need to make this value too an "interlock" value?
					ERROR_LOG(CPU, "mfv - invalid register %i", imm);
				}
			}
			break;

		case 7: // mtv
			if (imm < 128) {
				gpr.FlushR(rt);
				fpr.MapRegV(imm, MAP_DIRTY | MAP_NOINIT);
				LFS(fpr.V(imm), CTXREG, gpr.GetMipsRegOffset(rt));
			} else if (imm < 128 + VFPU_CTRL_MAX) { //mtvc //currentMIPS->vfpuCtrl[imm - 128] = R(rt);
				gpr.MapReg(rt);
				STW(gpr.R(rt), CTXREG, offsetof(MIPSState, vfpuCtrl) + 4 * (imm - 128));
				//gpr.BindToRegister(rt, true, false);
				//MOV(32, M(&mips_->vfpuCtrl[imm - 128]), gpr.R(rt));

				// TODO: Optimization if rt is Imm?
				// Set these BEFORE disable!
				if (imm - 128 == VFPU_CTRL_SPREFIX) {
					js.prefixSFlag = PpcJitState::PREFIX_UNKNOWN;
				} else if (imm - 128 == VFPU_CTRL_TPREFIX) {
					js.prefixTFlag = PpcJitState::PREFIX_UNKNOWN;
				} else if (imm - 128 == VFPU_CTRL_DPREFIX) {
					js.prefixDFlag = PpcJitState::PREFIX_UNKNOWN;
				}
			} else {
				//ERROR
				_dbg_assert_msg_(0,"mtv - invalid register");
			}
			break;

		default:
			DISABLE;
		}

		fpr.ReleaseSpillLocksAndDiscardTemps();
	}

	void Jit::Comp_Vmfvc(MIPSOpcode op) {
		CONDITIONAL_DISABLE;

		int vs = _VS;
		int imm = op & 0xFF;
		if (imm >= 128 && imm < 128 + VFPU_CTRL_MAX) {
			fpr.MapRegV(vs);
			ADDI(SREG, CTXREG, offsetof(MIPSState, vfpuCtrl[0]) + (imm - 128) * 4);
			LFS(fpr.V(vs), SREG, 0);
			fpr.ReleaseSpillLocksAndDiscardTemps();
		}
	}

	void Jit::Comp_Vmtvc(MIPSOpcode op) {
		CONDITIONAL_DISABLE;

		int vs = _VS;
		int imm = op & 0xFF;
		if (imm >= 128 && imm < 128 + VFPU_CTRL_MAX) {
			fpr.MapRegV(vs);
			ADDI(SREG, CTXREG, offsetof(MIPSState, vfpuCtrl[0]) + (imm - 128) * 4);
			SFS(fpr.V(vs), SREG, 0);
			fpr.ReleaseSpillLocksAndDiscardTemps();

			if (imm - 128 == VFPU_CTRL_SPREFIX) {
				js.prefixSFlag = PpcJitState::PREFIX_UNKNOWN;
			} else if (imm - 128 == VFPU_CTRL_TPREFIX) {
				js.prefixTFlag = PpcJitState::PREFIX_UNKNOWN;
			} else if (imm - 128 == VFPU_CTRL_DPREFIX) {
				js.prefixDFlag = PpcJitState::PREFIX_UNKNOWN;
			}
		}
	}

	static void PpcVfpuVmmovHelper(MIPSState *mips, u32 opEncoding) {
		MIPSOpcode op(opEncoding);
		MatrixSize sz = GetMtxSize(op);
		int n = GetMatrixSide(sz);
		u8 sregs[16], dregs[16];
		GetMatrixRegs(sregs, sz, _VS);
		GetMatrixRegs(dregs, sz, _VD);
		float * __restrict v = mips->v;
		float tmp[16];
		for (int a = 0; a < n; a++) {
			for (int b = 0; b < n; b++) {
				tmp[a * 4 + b] = v[voffset[sregs[a * 4 + b]]];
			}
		}
		for (int a = 0; a < n; a++) {
			for (int b = 0; b < n; b++) {
				v[voffset[dregs[a * 4 + b]]] = tmp[a * 4 + b];
			}
		}
	}

	void Jit::Comp_Vmmov(MIPSOpcode op) {
		CONDITIONAL_DISABLE;

		// TODO: This probably ignores prefixes?
		//if (js.MayHavePrefix()) {
		//	DISABLE;
		//}

		if (_VS == _VD) {
			// A lot of these in Wipeout... Just drop the instruction entirely.
			return;	
		}

		MatrixSize sz = GetMtxSize(op);
		int n = GetMatrixSide(sz);

		u8 sregs[16], dregs[16];
		GetMatrixRegs(sregs, sz, _VS);
		GetMatrixRegs(dregs, sz, _VD);

		// For 3x3 and 4x4, call compact helper (handles overlap too, saves ~140B RWX per vmmov.q).
		if (n >= 3) {
			for (int a = 0; a < n; a++) {
				for (int b = 0; b < n; b++) {
					fpr.FlushV(sregs[a * 4 + b]);
				}
			}
			for (int a = 0; a < n; a++) {
				for (int b = 0; b < n; b++) {
					fpr.DiscardV(dregs[a * 4 + b]);
				}
			}
			MR(R3, CTXREG);
			MOVI2R(R4, op.encoding);
			QuickCallFunction((void *)&PpcVfpuVmmovHelper);
			fpr.ReleaseSpillLocksAndDiscardTemps();
			return;
		}

		// Rough overlap check.
		bool overlap = false;
		if (GetMtx(_VS) == GetMtx(_VD)) {
			// Potential overlap (guaranteed for 3x3 or more).
			overlap = true;
		}

		if (overlap) {
			// Not so common, fallback.
			DISABLE;
		} else {
			for (int a = 0; a < n; a++) {
				for (int b = 0; b < n; b++) {
					fpr.MapDirtyInV(dregs[a * 4 + b], sregs[a * 4 + b]);
					FMR(fpr.V(dregs[a * 4 + b]), fpr.V(sregs[a * 4 + b]));
				}
			}
			fpr.ReleaseSpillLocksAndDiscardTemps();
		}
	}

	void Jit::Comp_VScl(MIPSOpcode op) {
		CONDITIONAL_DISABLE;

		if (js.HasUnknownPrefix() || disablePrefixes) {
			DISABLE;
		}

		VectorSize sz = GetVecSize(op);
		int n = GetNumVectorElements(sz);

		u8 sregs[4], dregs[4], treg;
		GetVectorRegsPrefixS(sregs, sz, _VS);
		GetVectorRegs(&treg, V_Single, _VT);
		GetVectorRegsPrefixD(dregs, sz, _VD);

		// Move to S0 early, so we don't have to worry about overlap with scale.
		fpr.LoadToRegV(FPR6, treg);

		// For prefixes to work, we just have to ensure that none of the output registers spill
		// and that there's no overlap.
		MIPSReg tempregs[4];
		for (int i = 0; i < n; ++i) {
			if (!IsOverlapSafe(dregs[i], i, n, sregs)) {
				// Need to use temp regs
				tempregs[i] = fpr.GetTempV();
			} else {
				tempregs[i] = dregs[i];
			}
		}

		// The meat of the function!
		for (int i = 0; i < n; i++) {
			fpr.MapDirtyInV(tempregs[i], sregs[i]);
			FMULS(fpr.V(tempregs[i]), fpr.V(sregs[i]), FPR6);
		}

		for (int i = 0; i < n; i++) {
			// All must be mapped for prefixes to work.
			if (dregs[i] != tempregs[i]) {
				fpr.MapDirtyInV(dregs[i], tempregs[i]);
				FMR(fpr.V(dregs[i]), fpr.V(tempregs[i]));
			}
		}

		ApplyPrefixD(dregs, sz);

		fpr.ReleaseSpillLocksAndDiscardTemps();
	}

	static void PpcVfpuVmmulHelper(MIPSState *mips, u32 opEncoding) {
		MIPSOpcode op(opEncoding);
		MatrixSize sz = GetMtxSize(op);
		int n = GetMatrixSide(sz);
		u8 sregs[16], tregs[16], dregs[16];
		GetMatrixRegs(sregs, sz, _VS);
		GetMatrixRegs(tregs, sz, _VT);
		GetMatrixRegs(dregs, sz, _VD);
		float * __restrict v = mips->v;
		if (n == 4) {
#ifdef __wiiu__
			// PowerPC 750CL Paired Singles 4x4 matrix multiply:
			// scol[k*4 + b] = s[b*4 + k], t[a*4 + k] -> res[a*4 + b] = sum_k scol[k*4 + b] * t[a*4 + k].
			alignas(8) float scol[16];
			alignas(8) float t[16];
			alignas(8) float res[16];
			for (int a = 0; a < 4; a++) {
				for (int b = 0; b < 4; b++) {
					scol[b * 4 + a] = v[voffset[sregs[a * 4 + b]]];
					t[a * 4 + b] = v[voffset[tregs[a * 4 + b]]];
				}
			}
			asm volatile(
				".machine push\n\t"
				".machine gekko\n\t"
				"psq_l   0,   0(%0), 0, 0\n\t"
				"psq_l   1,   8(%0), 0, 0\n\t"
				"psq_l   2,  16(%0), 0, 0\n\t"
				"psq_l   3,  24(%0), 0, 0\n\t"
				"psq_l   4,  32(%0), 0, 0\n\t"
				"psq_l   5,  40(%0), 0, 0\n\t"
				"psq_l   6,  48(%0), 0, 0\n\t"
				"psq_l   7,  56(%0), 0, 0\n\t"
				// Column 0
				"psq_l   8,   0(%1), 0, 0\n\t"
				"psq_l   9,   8(%1), 0, 0\n\t"
				"ps_muls0  10, 0, 8\n\t"
				"ps_muls0  11, 1, 8\n\t"
				"ps_madds1 10, 2, 8, 10\n\t"
				"ps_madds1 11, 3, 8, 11\n\t"
				"ps_madds0 10, 4, 9, 10\n\t"
				"ps_madds0 11, 5, 9, 11\n\t"
				"ps_madds1 10, 6, 9, 10\n\t"
				"ps_madds1 11, 7, 9, 11\n\t"
				"psq_st  10,   0(%2), 0, 0\n\t"
				"psq_st  11,   8(%2), 0, 0\n\t"
				// Column 1
				"psq_l   8,  16(%1), 0, 0\n\t"
				"psq_l   9,  24(%1), 0, 0\n\t"
				"ps_muls0  10, 0, 8\n\t"
				"ps_muls0  11, 1, 8\n\t"
				"ps_madds1 10, 2, 8, 10\n\t"
				"ps_madds1 11, 3, 8, 11\n\t"
				"ps_madds0 10, 4, 9, 10\n\t"
				"ps_madds0 11, 5, 9, 11\n\t"
				"ps_madds1 10, 6, 9, 10\n\t"
				"ps_madds1 11, 7, 9, 11\n\t"
				"psq_st  10,  16(%2), 0, 0\n\t"
				"psq_st  11,  24(%2), 0, 0\n\t"
				// Column 2
				"psq_l   8,  32(%1), 0, 0\n\t"
				"psq_l   9,  40(%1), 0, 0\n\t"
				"ps_muls0  10, 0, 8\n\t"
				"ps_muls0  11, 1, 8\n\t"
				"ps_madds1 10, 2, 8, 10\n\t"
				"ps_madds1 11, 3, 8, 11\n\t"
				"ps_madds0 10, 4, 9, 10\n\t"
				"ps_madds0 11, 5, 9, 11\n\t"
				"ps_madds1 10, 6, 9, 10\n\t"
				"ps_madds1 11, 7, 9, 11\n\t"
				"psq_st  10,  32(%2), 0, 0\n\t"
				"psq_st  11,  40(%2), 0, 0\n\t"
				// Column 3
				"psq_l   8,  48(%1), 0, 0\n\t"
				"psq_l   9,  56(%1), 0, 0\n\t"
				"ps_muls0  10, 0, 8\n\t"
				"ps_muls0  11, 1, 8\n\t"
				"ps_madds1 10, 2, 8, 10\n\t"
				"ps_madds1 11, 3, 8, 11\n\t"
				"ps_madds0 10, 4, 9, 10\n\t"
				"ps_madds0 11, 5, 9, 11\n\t"
				"ps_madds1 10, 6, 9, 10\n\t"
				"ps_madds1 11, 7, 9, 11\n\t"
				"psq_st  10,  48(%2), 0, 0\n\t"
				"psq_st  11,  56(%2), 0, 0\n\t"
				".machine pop\n\t"
				:
				: "b"(scol), "b"(t), "b"(res)
				: "fr0", "fr1", "fr2", "fr3", "fr4", "fr5", "fr6", "fr7",
				  "fr8", "fr9", "fr10", "fr11", "memory");
			for (int a = 0; a < 4; a++) {
				for (int b = 0; b < 4; b++) {
					v[voffset[dregs[a * 4 + b]]] = res[a * 4 + b];
				}
			}
#else
			float s[16], t[16];
			for (int a = 0; a < 4; a++) {
				for (int b = 0; b < 4; b++) {
					s[a * 4 + b] = v[voffset[sregs[a * 4 + b]]];
					t[a * 4 + b] = v[voffset[tregs[a * 4 + b]]];
				}
			}
			for (int a = 0; a < 4; a++) {
				const float t0 = t[a * 4 + 0];
				const float t1 = t[a * 4 + 1];
				const float t2 = t[a * 4 + 2];
				const float t3 = t[a * 4 + 3];
				for (int b = 0; b < 4; b++) {
					v[voffset[dregs[a * 4 + b]]] =
						s[b * 4 + 0] * t0 + s[b * 4 + 1] * t1 +
						s[b * 4 + 2] * t2 + s[b * 4 + 3] * t3;
				}
			}
#endif
		} else {
			float s[9], t[9];
			for (int a = 0; a < 3; a++) {
				for (int b = 0; b < 3; b++) {
					s[a * 3 + b] = v[voffset[sregs[a * 4 + b]]];
					t[a * 3 + b] = v[voffset[tregs[a * 4 + b]]];
				}
			}
			for (int a = 0; a < 3; a++) {
				const float t0 = t[a * 3 + 0];
				const float t1 = t[a * 3 + 1];
				const float t2 = t[a * 3 + 2];
				for (int b = 0; b < 3; b++) {
					v[voffset[dregs[a * 4 + b]]] =
						s[b * 3 + 0] * t0 + s[b * 3 + 1] * t1 + s[b * 3 + 2] * t2;
				}
			}
		}
	}

	void Jit::Comp_Vmmul(MIPSOpcode op) {
		CONDITIONAL_DISABLE;

		// TODO: This probably ignores prefixes?
		if (js.MayHavePrefix() || disablePrefixes) {
			DISABLE;
		}

		MatrixSize sz = GetMtxSize(op);
		int n = GetMatrixSide(sz);

		u8 sregs[16], tregs[16], dregs[16];
		GetMatrixRegs(sregs, sz, _VS);
		GetMatrixRegs(tregs, sz, _VT);
		GetMatrixRegs(dregs, sz, _VD);

		// For 3x3 and 4x4, call compact compiled helper (handles overlap too, saves ~500B RWX per vmmul.q).
		if (n >= 3) {
			for (int a = 0; a < n; a++) {
				for (int b = 0; b < n; b++) {
					fpr.FlushV(sregs[a * 4 + b]);
					fpr.FlushV(tregs[a * 4 + b]);
				}
			}
			for (int a = 0; a < n; a++) {
				for (int b = 0; b < n; b++) {
					fpr.DiscardV(dregs[a * 4 + b]);
				}
			}
			MR(R3, CTXREG);
			MOVI2R(R4, op.encoding);
			QuickCallFunction((void *)&PpcVfpuVmmulHelper);
			fpr.ReleaseSpillLocksAndDiscardTemps();
			return;
		}

		// Rough overlap check.
		bool overlap = false;
		if (GetMtx(_VS) == GetMtx(_VD) || GetMtx(_VT) == GetMtx(_VD)) {
			// Potential overlap (guaranteed for 3x3 or more).
			overlap = true;
		}

		if (overlap) {
			DISABLE;
		} else {
			for (int a = 0; a < n; a++) {
				for (int b = 0; b < n; b++) {
					fpr.MapInInV(sregs[b * 4], tregs[a * 4]);
					FMULS(FPR6, fpr.V(sregs[b * 4]), fpr.V(tregs[a * 4]));
					for (int c = 1; c < n; c++) {
						fpr.MapInInV(sregs[b * 4 + c], tregs[a * 4 + c]);
						FMADDS(FPR6, fpr.V(sregs[b * 4 + c]), fpr.V(tregs[a * 4 + c]), FPR6);
					}
					fpr.MapRegV(dregs[a * 4 + b], MAP_DIRTY | MAP_NOINIT);
					FMR(fpr.V(dregs[a * 4 + b]), FPR6);
				}
			}
			fpr.ReleaseSpillLocksAndDiscardTemps();
		}
	}

	void Jit::Comp_Vmscl(MIPSOpcode op) {
		DISABLE;
	}

	static void PpcVfpuVtfmHelper(MIPSState *mips, u32 opEncoding) {
		MIPSOpcode op(opEncoding);
		VectorSize sz = GetVecSize(op);
		MatrixSize msz = GetMtxSize(op);
		int n = GetNumVectorElements(sz);
		int ins = (op >> 23) & 7;
		bool homogenous = false;
		if (n == ins) {
			n++;
			sz = (VectorSize)((int)(sz) + 1);
			msz = (MatrixSize)((int)(msz) + 1);
			homogenous = true;
		}
		u8 sregs[16], dregs[4], tregs[4];
		GetMatrixRegs(sregs, msz, _VS);
		GetVectorRegs(tregs, sz, _VT);
		GetVectorRegs(dregs, sz, _VD);
		float * __restrict v = mips->v;
		if (n == 4) {
#ifdef __wiiu__
			alignas(8) float scol[16];
			alignas(8) float t[4];
			alignas(8) float res[4];
			for (int i = 0; i < 4; i++) {
				for (int k = 0; k < 4; k++) {
					scol[k * 4 + i] = v[voffset[sregs[i * 4 + k]]];
				}
			}
			t[0] = v[voffset[tregs[0]]];
			t[1] = v[voffset[tregs[1]]];
			t[2] = v[voffset[tregs[2]]];
			t[3] = homogenous ? 1.0f : v[voffset[tregs[3]]];
			asm volatile(
				".machine push\n\t"
				".machine gekko\n\t"
				"psq_l   0,   0(%0), 0, 0\n\t"
				"psq_l   1,   8(%0), 0, 0\n\t"
				"psq_l   2,  16(%0), 0, 0\n\t"
				"psq_l   3,  24(%0), 0, 0\n\t"
				"psq_l   4,  32(%0), 0, 0\n\t"
				"psq_l   5,  40(%0), 0, 0\n\t"
				"psq_l   6,  48(%0), 0, 0\n\t"
				"psq_l   7,  56(%0), 0, 0\n\t"
				"psq_l   8,   0(%1), 0, 0\n\t"
				"psq_l   9,   8(%1), 0, 0\n\t"
				"ps_muls0  10, 0, 8\n\t"
				"ps_muls0  11, 1, 8\n\t"
				"ps_madds1 10, 2, 8, 10\n\t"
				"ps_madds1 11, 3, 8, 11\n\t"
				"ps_madds0 10, 4, 9, 10\n\t"
				"ps_madds0 11, 5, 9, 11\n\t"
				"ps_madds1 10, 6, 9, 10\n\t"
				"ps_madds1 11, 7, 9, 11\n\t"
				"psq_st  10,   0(%2), 0, 0\n\t"
				"psq_st  11,   8(%2), 0, 0\n\t"
				".machine pop\n\t"
				:
				: "b"(scol), "b"(t), "b"(res)
				: "fr0", "fr1", "fr2", "fr3", "fr4", "fr5", "fr6", "fr7",
				  "fr8", "fr9", "fr10", "fr11", "memory");
			for (int i = 0; i < 4; i++) {
				v[voffset[dregs[i]]] = res[i];
			}
#else
			float t0 = v[voffset[tregs[0]]];
			float t1 = v[voffset[tregs[1]]];
			float t2 = v[voffset[tregs[2]]];
			float t3 = homogenous ? 1.0f : v[voffset[tregs[3]]];
			float res[4];
			for (int i = 0; i < 4; i++) {
				res[i] = v[voffset[sregs[i * 4 + 0]]] * t0
				       + v[voffset[sregs[i * 4 + 1]]] * t1
				       + v[voffset[sregs[i * 4 + 2]]] * t2
				       + v[voffset[sregs[i * 4 + 3]]] * t3;
			}
			for (int i = 0; i < 4; i++) {
				v[voffset[dregs[i]]] = res[i];
			}
#endif
		} else {
			float t0 = v[voffset[tregs[0]]];
			float t1 = v[voffset[tregs[1]]];
			float t2 = homogenous ? 1.0f : v[voffset[tregs[2]]];
			float res[3];
			for (int i = 0; i < 3; i++) {
				res[i] = v[voffset[sregs[i * 4 + 0]]] * t0
				       + v[voffset[sregs[i * 4 + 1]]] * t1
				       + v[voffset[sregs[i * 4 + 2]]] * t2;
			}
			for (int i = 0; i < 3; i++) {
				v[voffset[dregs[i]]] = res[i];
			}
		}
	}

	void Jit::Comp_Vtfm(MIPSOpcode op) {
		CONDITIONAL_DISABLE;

		// TODO: This probably ignores prefixes?  Or maybe uses D?
		if (js.MayHavePrefix() || disablePrefixes) {
			DISABLE;
		}

		VectorSize sz = GetVecSize(op);
		MatrixSize msz = GetMtxSize(op);
		int n = GetNumVectorElements(sz);
		int ins = (op >> 23) & 7;

		bool homogenous = false;
		if (n == ins) {
			n++;
			sz = (VectorSize)((int)(sz) + 1);
			msz = (MatrixSize)((int)(msz) + 1);
			homogenous = true;
		}
		// Otherwise, n should already be ins + 1.
		else if (n != ins + 1) {
			DISABLE;
		}

		u8 sregs[16], dregs[4], tregs[4];
		GetMatrixRegs(sregs, msz, _VS);
		GetVectorRegs(tregs, sz, _VT);
		GetVectorRegs(dregs, sz, _VD);

		// For 3x3 and 4x4, call compact Paired-Singles helper (saves ~180B RWX and avoids 10+ FPR spills).
		if (n >= 3) {
			for (int i = 0; i < n; i++) {
				for (int k = 0; k < n; k++) {
					fpr.FlushV(sregs[i * 4 + k]);
				}
				if (!homogenous || i != n - 1) {
					fpr.FlushV(tregs[i]);
				}
			}
			for (int i = 0; i < n; i++) {
				fpr.DiscardV(dregs[i]);
			}
			MR(R3, CTXREG);
			MOVI2R(R4, op.encoding);
			QuickCallFunction((void *)&PpcVfpuVtfmHelper);
			fpr.ReleaseSpillLocksAndDiscardTemps();
			return;
		}

		// TODO: test overlap, optimize.
		int tempregs[4];
		for (int i = 0; i < n; i++) {
			fpr.MapInInV(sregs[i * 4], tregs[0]);
			FMULS(FPR6, fpr.V(sregs[i * 4]), fpr.V(tregs[0]));
			for (int k = 1; k < n; k++) {
				if (!homogenous || k != n - 1) {
					fpr.MapInInV(sregs[i * 4 + k], tregs[k]);
					FMADDS(FPR6, fpr.V(sregs[i * 4 + k]), fpr.V(tregs[k]), FPR6);
				} else {
					fpr.MapRegV(sregs[i * 4 + k]);
					FADDS(FPR6, FPR6, fpr.V(sregs[i * 4 + k]));
				}
			}

			int temp = fpr.GetTempV();
			fpr.MapRegV(temp, MAP_NOINIT | MAP_DIRTY);
			fpr.SpillLockV(temp);
			FMR(fpr.V(temp), FPR6);
			tempregs[i] = temp;
		}
		for (int i = 0; i < n; i++) {
			u8 temp = tempregs[i];
			fpr.MapRegV(dregs[i], MAP_NOINIT | MAP_DIRTY);
			FMR(fpr.V(dregs[i]), fpr.V(temp));
		}

		fpr.ReleaseSpillLocksAndDiscardTemps();
	}

	void Jit::Comp_VHdp(MIPSOpcode op) {
		CONDITIONAL_DISABLE;
		if (js.HasUnknownPrefix() || disablePrefixes || js.MayHavePrefix())
			DISABLE;

		VectorSize sz = GetVecSize(op);
		if (sz == V_Single)
			DISABLE;
		int n = GetNumVectorElements(sz);

		u8 sregs[4], tregs[4], dreg;
		GetVectorRegs(sregs, sz, _VS);
		GetVectorRegs(tregs, sz, _VT);
		GetVectorRegs(&dreg, V_Single, _VD);

		fpr.MapRegsAndSpillLockV(tregs, sz, 0);
		fpr.SpillLockV(sregs, sz);
		for (int i = 0; i < n - 1; ++i)
			fpr.MapRegV(sregs[i], 0);

		FMULS(FPR6, fpr.V(sregs[0]), fpr.V(tregs[0]));
		for (int i = 1; i < n - 1; ++i)
			FMADDS(FPR6, fpr.V(sregs[i]), fpr.V(tregs[i]), FPR6);
		FADDS(FPR6, FPR6, fpr.V(tregs[n - 1]));

		fpr.MapRegV(dreg, MAP_DIRTY | MAP_NOINIT);
		FMR(fpr.V(dreg), FPR6);

		fpr.ReleaseSpillLocksAndDiscardTemps();
	}

	void Jit::Comp_VCrs(MIPSOpcode op) {
		DISABLE;
	}

	void Jit::Comp_VDet(MIPSOpcode op) {
		DISABLE;
	}

	void Jit::Comp_Vi2x(MIPSOpcode op) {
		DISABLE;
	}

	void Jit::Comp_Vx2i(MIPSOpcode op) {
		DISABLE;
	}

	void Jit::Comp_Vf2i(MIPSOpcode op) {
		CONDITIONAL_DISABLE;
		if (js.HasUnknownPrefix() || disablePrefixes || js.MayHavePrefix())
			DISABLE;

		int op2 = (op >> 21) & 0x1f;
		if (op2 != 16 && op2 != 17)
			DISABLE;

		VectorSize sz = GetVecSize(op);
		int n = GetNumVectorElements(sz);
		int imm = (op >> 16) & 0x1f;
		const float mult = (float)(1U << imm);

		u8 sregs[4], dregs[4];
		GetVectorRegs(sregs, sz, _VS);
		GetVectorRegs(dregs, sz, _VD);

		if (imm != 0)
			MOVI2F(FPR8, mult, false);

		fpr.MapRegsAndSpillLockV(sregs, sz, 0);
		for (int i = 0; i < n; ++i) {
			PPCReg src = fpr.V(sregs[i]);
			if (imm != 0) {
				FMULS(FPR7, src, FPR8);
				src = FPR7;
			}
			if (op2 == 16)
				FCTIW(FPR6, src);
			else
				FCTIWZ(FPR6, src);
			SFD(FPR6, R1, 16);
			LWZ(SREG, R1, 20);
			STW(SREG, R1, 32 + i * 4);
		}
		for (int i = 0; i < n; ++i) {
			fpr.DiscardV(dregs[i]);
			LWZ(SREG, R1, 32 + i * 4);
			STW(SREG, CTXREG, fpr.GetMipsRegOffsetV(dregs[i]));
		}

		fpr.ReleaseSpillLocksAndDiscardTemps();
	}

	void Jit::Comp_Vi2f(MIPSOpcode op) {
		CONDITIONAL_DISABLE;

		if (js.HasUnknownPrefix() || disablePrefixes || js.prefixS != 0xE4)
			DISABLE;

		VectorSize sz = GetVecSize(op);
		int n = GetNumVectorElements(sz);

		int imm = (op >> 16) & 0x1f;
		const float mult = 1.0f / (float)(1UL << imm);

		u8 sregs[4], dregs[4];
		GetVectorRegs(sregs, sz, _VS);
		GetVectorRegsPrefixD(dregs, sz, _VD);

		MIPSReg tempregs[4];
		for (int i = 0; i < n; ++i) {
			if (!IsOverlapSafe(dregs[i], i, n, sregs)) {
				tempregs[i] = fpr.GetTempV();
			} else {
				tempregs[i] = dregs[i];
			}
		}

		if (mult != 1.0f)
			MOVI2F(FPR8, mult, false);

		for (int i = 0; i < n; ++i)
			fpr.FlushV(sregs[i]);

		LIS(R6, 0x4330);
		STW(R6, R1, 16);
		STW(R6, R1, 24);
		LIS(R7, 0x8000);
		STW(R7, R1, 28);
		LFD(FPR7, R1, 24);

		for (int i = 0; i < n; ++i) {
			LWZ(SREG, CTXREG, fpr.GetMipsRegOffsetV(sregs[i]));
			XORIS(SREG, SREG, 0x8000);
			STW(SREG, R1, 20);
			LFD(FPR6, R1, 16);
			fpr.MapRegV(tempregs[i], MAP_DIRTY | MAP_NOINIT);
			FSUB(fpr.V(tempregs[i]), FPR6, FPR7);
			FRSP(fpr.V(tempregs[i]), fpr.V(tempregs[i]));

			if (mult != 1.0f)
				FMULS(fpr.V(tempregs[i]), fpr.V(tempregs[i]), FPR8);
		}

		for (int i = 0; i < n; ++i) {
			if (dregs[i] != tempregs[i]) {
				fpr.MapDirtyInV(dregs[i], tempregs[i]);
				FMR(fpr.V(dregs[i]), fpr.V(tempregs[i]));
			}
		}

		ApplyPrefixD(dregs, sz);
		fpr.ReleaseSpillLocksAndDiscardTemps();
	}

	void Jit::Comp_Vh2f(MIPSOpcode op) {
		DISABLE;
	}

	void Jit::Comp_Vcst(MIPSOpcode op) {
		CONDITIONAL_DISABLE;

		if (js.HasUnknownPrefix() || disablePrefixes) {
			DISABLE;
		}

		int conNum = (op >> 16) & 0x1f;
		int vd = _VD;

		VectorSize sz = GetVecSize(op);
		int n = GetNumVectorElements(sz);

		u8 dregs[4];
		GetVectorRegsPrefixD(dregs, sz, _VD);
		fpr.MapRegsAndSpillLockV(dregs, sz, MAP_NOINIT | MAP_DIRTY);

		MOVI2R(SREG, (u32)(void *)&cst_constants[conNum]);
		LFS(FPR6, SREG, 0);
		for (int i = 0; i < n; ++i)
			FMR(fpr.V(dregs[i]), FPR6);

		ApplyPrefixD(dregs, sz);
		fpr.ReleaseSpillLocksAndDiscardTemps();
	}

	void Jit::Comp_Vhoriz(MIPSOpcode op) {
		DISABLE;
	}

	void Jit::Comp_VRot(MIPSOpcode op) {
		DISABLE;
	}

	void Jit::Comp_VIdt(MIPSOpcode op) {
		CONDITIONAL_DISABLE

		if (js.HasUnknownPrefix() || disablePrefixes) {
			DISABLE;
		}

		int vd = _VD;
		VectorSize sz = GetVecSize(op);
		int n = GetNumVectorElements(sz);
		MOVI2F(FPR6, 0.0f);
		MOVI2F(FPR7, 1.0f);
		u8 dregs[4];
		GetVectorRegsPrefixD(dregs, sz, _VD);
		fpr.MapRegsAndSpillLockV(dregs, sz, MAP_NOINIT | MAP_DIRTY);
		switch (sz)
		{
		case V_Pair:
			FMR(fpr.V(dregs[0]), (vd&1)==0 ? FPR7 : FPR6);
			FMR(fpr.V(dregs[1]), (vd&1)==1 ? FPR7 : FPR6);
			break;								   
		case V_Quad:							   
			FMR(fpr.V(dregs[0]), (vd&3)==0 ? FPR7 : FPR6);
			FMR(fpr.V(dregs[1]), (vd&3)==1 ? FPR7 : FPR6);
			FMR(fpr.V(dregs[2]), (vd&3)==2 ? FPR7 : FPR6);
			FMR(fpr.V(dregs[3]), (vd&3)==3 ? FPR7 : FPR6);
			break;
		default:
			_dbg_assert_msg_(0,"Trying to interpret instruction that can't be interpreted");
			break;
		}
		
		ApplyPrefixD(dregs, sz);

		fpr.ReleaseSpillLocksAndDiscardTemps();
	}

	void Jit::Comp_Vcmp(MIPSOpcode op) {
		CONDITIONAL_DISABLE;
		if (js.HasUnknownPrefix() || disablePrefixes)
			DISABLE;

		int cond = op & 0xf;
		if (cond == VC_EI || cond == VC_ES || cond == VC_NI || cond == VC_NS)
			DISABLE;

		VectorSize sz = GetVecSize(op);
		int n = GetNumVectorElements(sz);
		int affected_bits = (1 << 4) | (1 << 5) | ((1 << n) - 1);

		if (cond == VC_FL || cond == VC_TR) {
			LWZ(R3, CTXREG, offsetof(MIPSState, vfpuCtrl[VFPU_CTRL_CC]));
			if (cond == VC_FL) {
				MOVI2R(R9, ~affected_bits);
				AND(R3, R3, R9);
			} else {
				ORI(R3, R3, affected_bits);
			}
			STW(R3, CTXREG, offsetof(MIPSState, vfpuCtrl[VFPU_CTRL_CC]));
			return;
		}

		bool needT = (cond < VC_EZ);
		u8 sregs[4], tregs[4];
		GetVectorRegsPrefixS(sregs, sz, _VS);
		if (needT)
			GetVectorRegsPrefixT(tregs, sz, _VT);

		fpr.MapRegsAndSpillLockV(sregs, sz, 0);
		if (needT)
			fpr.MapRegsAndSpillLockV(tregs, sz, 0);

		if (cond == VC_EZ || cond == VC_NZ)
			MOVI2F(FPR6, 0.0f, false);

		LI(R4, 0);
		LI(R7, 0);
		LI(R8, 1);

		for (int i = 0; i < n; ++i) {
			PPCReg a = fpr.V(sregs[i]);
			PPCReg b = needT ? fpr.V(tregs[i]) : ((cond == VC_EZ || cond == VC_NZ) ? FPR6 : a);
			FCMPU(0, a, b);

			switch (cond) {
			case VC_EQ:
			case VC_EZ:
				MFCR(SREG);
				RLWINM(SREG, SREG, 3, 31, 31);
				break;
			case VC_NE:
			case VC_NZ:
				MFCR(SREG);
				RLWINM(SREG, SREG, 3, 31, 31);
				XORI(SREG, SREG, 1);
				break;
			case VC_LT:
				MFCR(SREG);
				RLWINM(SREG, SREG, 1, 31, 31);
				break;
			case VC_LE:
				CROR(0, 0, 2);
				MFCR(SREG);
				RLWINM(SREG, SREG, 1, 31, 31);
				break;
			case VC_GT:
				MFCR(SREG);
				RLWINM(SREG, SREG, 2, 31, 31);
				break;
			case VC_GE:
				CROR(1, 1, 2);
				MFCR(SREG);
				RLWINM(SREG, SREG, 2, 31, 31);
				break;
			case VC_EN:
				MFCR(SREG);
				RLWINM(SREG, SREG, 4, 31, 31);
				break;
			case VC_NN:
				MFCR(SREG);
				RLWINM(SREG, SREG, 4, 31, 31);
				XORI(SREG, SREG, 1);
				break;
			default:
				break;
			}

			OR(R7, R7, SREG);
			AND(R8, R8, SREG);
			if (i > 0)
				SLWI(SREG, SREG, i);
			OR(R4, R4, SREG);
		}

		SLWI(R7, R7, 4);
		SLWI(R8, R8, 5);
		OR(R4, R4, R7);
		OR(R4, R4, R8);

		LWZ(R3, CTXREG, offsetof(MIPSState, vfpuCtrl[VFPU_CTRL_CC]));
		MOVI2R(R9, ~affected_bits);
		AND(R3, R3, R9);
		OR(R3, R3, R4);
		STW(R3, CTXREG, offsetof(MIPSState, vfpuCtrl[VFPU_CTRL_CC]));

		fpr.ReleaseSpillLocksAndDiscardTemps();
	}

	void Jit::Comp_Vcmov(MIPSOpcode op) {
		CONDITIONAL_DISABLE;
		if (js.HasUnknownPrefix() || disablePrefixes || js.MayHavePrefix())
			DISABLE;

		int imm3 = (op >> 16) & 7;
		int tf = (op >> 19) & 1;
		if (imm3 >= 7)
			DISABLE;

		VectorSize sz = GetVecSize(op);
		int n = GetNumVectorElements(sz);

		u8 sregs[4], dregs[4];
		GetVectorRegs(sregs, sz, _VS);
		GetVectorRegs(dregs, sz, _VD);

		MIPSReg tempregs[4];
		bool needTemp = false;
		for (int i = 0; i < n; ++i) {
			if (!IsOverlapSafeAllowS(dregs[i], i, n, sregs)) {
				tempregs[i] = fpr.GetTempV();
				needTemp = true;
			} else {
				tempregs[i] = sregs[i];
			}
		}

		fpr.MapRegsAndSpillLockV(sregs, sz, 0);
		if (needTemp) {
			for (int i = 0; i < n; ++i) {
				if (tempregs[i] != sregs[i]) {
					fpr.MapRegV(tempregs[i], MAP_DIRTY | MAP_NOINIT);
					fpr.SpillLockV(tempregs[i], V_Single);
					FMR(fpr.V(tempregs[i]), fpr.V(sregs[i]));
				}
			}
		}
		fpr.MapRegsAndSpillLockV(dregs, sz, MAP_DIRTY);

		LWZ(R3, CTXREG, offsetof(MIPSState, vfpuCtrl[VFPU_CTRL_CC]));

		if (imm3 < 6) {
			ANDI(SREG, R3, 1 << imm3);
			FixupBranch skip = B_Cond(tf ? _BNE : _BEQ);
			for (int i = 0; i < n; ++i) {
				if (dregs[i] != tempregs[i])
					FMR(fpr.V(dregs[i]), fpr.V(tempregs[i]));
			}
			SetJumpTarget(skip);
		} else {
			for (int i = 0; i < n; ++i) {
				ANDI(SREG, R3, 1 << i);
				FixupBranch skip = B_Cond(tf ? _BNE : _BEQ);
				if (dregs[i] != tempregs[i])
					FMR(fpr.V(dregs[i]), fpr.V(tempregs[i]));
				SetJumpTarget(skip);
			}
		}

		fpr.ReleaseSpillLocksAndDiscardTemps();
	}

	void Jit::Comp_Viim(MIPSOpcode op) {
		CONDITIONAL_DISABLE;

		if (js.HasUnknownPrefix() || disablePrefixes) {
			DISABLE;
		}

		u8 dreg;
		GetVectorRegs(&dreg, V_Single, _VT);

		s32 imm = (s32)(s16)(u16)(op & 0xFFFF);
		fpr.MapRegV(dreg, MAP_DIRTY | MAP_NOINIT);
		MOVI2F(fpr.V(dreg), (float)imm);

		ApplyPrefixD(&dreg, V_Single);
		fpr.ReleaseSpillLocksAndDiscardTemps();
	}

	void Jit::Comp_Vfim(MIPSOpcode op) {
		CONDITIONAL_DISABLE;

		if (js.HasUnknownPrefix() || disablePrefixes) {
			DISABLE;
		}

		u8 dreg;
		GetVectorRegs(&dreg, V_Single, _VT);

		FP16 half;
		half.u = op & 0xFFFF;
		FP32 fval = half_to_float_fast5(half);
		fpr.MapRegV(dreg, MAP_DIRTY | MAP_NOINIT);
		MOVI2F(fpr.V(dreg), fval.f);

		ApplyPrefixD(&dreg, V_Single);
		fpr.ReleaseSpillLocksAndDiscardTemps();
	}

	void Jit::Comp_VCrossQuat(MIPSOpcode op) {
		CONDITIONAL_DISABLE;

		if (js.HasUnknownPrefix() || disablePrefixes || js.MayHavePrefix()) {
			DISABLE;
		}

		VectorSize sz = GetVecSize(op);
		if (sz != V_Triple) {
			// Quaternion product vqmul.q untested
			DISABLE;
		}

		u8 sregs[4], tregs[4], dregs[4];
		GetVectorRegs(sregs, sz, _VS);
		GetVectorRegs(tregs, sz, _VT);
		GetVectorRegs(dregs, sz, _VD);

		// Map everything into registers.
		fpr.MapRegsAndSpillLockV(sregs, sz, 0);
		fpr.MapRegsAndSpillLockV(tregs, sz, 0);

		int temp3 = fpr.GetTempV();
		fpr.MapRegV(temp3, MAP_DIRTY | MAP_NOINIT);
		fpr.SpillLockV(temp3, V_Single);
		// Cross product vcrsp.t: d = s x t

		// Compute X = s1*t2 - s2*t1
		FMULS(FPR6, fpr.V(sregs[2]), fpr.V(tregs[1]));
		FMSUBS(FPR6, fpr.V(sregs[1]), fpr.V(tregs[2]), FPR6);

		// Compute Y = s2*t0 - s0*t2
		FMULS(FPR7, fpr.V(sregs[0]), fpr.V(tregs[2]));
		FMSUBS(FPR7, fpr.V(sregs[2]), fpr.V(tregs[0]), FPR7);

		// Compute Z = s0*t1 - s1*t0
		FMULS(fpr.V(temp3), fpr.V(sregs[1]), fpr.V(tregs[0]));
		FMSUBS(fpr.V(temp3), fpr.V(sregs[0]), fpr.V(tregs[1]), fpr.V(temp3));

		fpr.MapRegsAndSpillLockV(dregs, V_Triple, MAP_DIRTY | MAP_NOINIT);
		FMR(fpr.V(dregs[0]), FPR6);
		FMR(fpr.V(dregs[1]), FPR7);
		FMR(fpr.V(dregs[2]), fpr.V(temp3));

		fpr.ReleaseSpillLocksAndDiscardTemps();
	}
	void Jit::Comp_Vsgn(MIPSOpcode op) {
		CONDITIONAL_DISABLE;
		if (js.HasUnknownPrefix() || disablePrefixes)
			DISABLE;

		VectorSize sz = GetVecSize(op);
		int n = GetNumVectorElements(sz);

		u8 sregs[4], dregs[4];
		GetVectorRegsPrefixS(sregs, sz, _VS);
		GetVectorRegsPrefixD(dregs, sz, _VD);

		MIPSReg tempregs[4];
		for (int i = 0; i < n; ++i) {
			if (!IsOverlapSafe(dregs[i], i, n, sregs))
				tempregs[i] = fpr.GetTempV();
			else
				tempregs[i] = dregs[i];
		}

		MOVI2F(FPR6, 0.0f, false);
		MOVI2F(FPR7, 1.0f, false);
		MOVI2F(FPR8, -1.0f, false);
		for (int i = 0; i < n; ++i) {
			fpr.MapDirtyInV(tempregs[i], sregs[i]);
			PPCReg s = fpr.V(sregs[i]);
			PPCReg d = fpr.V(tempregs[i]);
			FCMPU(0, s, FPR6);
			FixupBranch isGt = B_Cond(_BGT);
			FixupBranch isLt = B_Cond(_BLT);
			FMR(d, FPR6);
			FixupBranch done1 = B();
			SetJumpTarget(isGt);
			FMR(d, FPR7);
			FixupBranch done2 = B();
			SetJumpTarget(isLt);
			FMR(d, FPR8);
			SetJumpTarget(done1);
			SetJumpTarget(done2);
		}
		for (int i = 0; i < n; ++i) {
			if (dregs[i] != tempregs[i]) {
				fpr.MapDirtyInV(dregs[i], tempregs[i]);
				FMR(fpr.V(dregs[i]), fpr.V(tempregs[i]));
			}
		}
		ApplyPrefixD(dregs, sz);
		fpr.ReleaseSpillLocksAndDiscardTemps();
	}
	void Jit::Comp_Vocp(MIPSOpcode op) {
		CONDITIONAL_DISABLE;
		if (js.HasUnknownPrefix() || disablePrefixes)
			DISABLE;

		VectorSize sz = GetVecSize(op);
		int n = GetNumVectorElements(sz);

		u8 sregs[4], dregs[4];
		GetVectorRegsPrefixS(sregs, sz, _VS);
		GetVectorRegsPrefixD(dregs, sz, _VD);

		MIPSReg tempregs[4];
		for (int i = 0; i < n; ++i) {
			if (!IsOverlapSafe(dregs[i], i, n, sregs))
				tempregs[i] = fpr.GetTempV();
			else
				tempregs[i] = dregs[i];
		}

		MOVI2F(FPR6, 1.0f, false);
		for (int i = 0; i < n; ++i) {
			fpr.MapDirtyInV(tempregs[i], sregs[i]);
			FSUBS(fpr.V(tempregs[i]), FPR6, fpr.V(sregs[i]));
		}

		for (int i = 0; i < n; ++i) {
			if (dregs[i] != tempregs[i]) {
				fpr.MapDirtyInV(dregs[i], tempregs[i]);
				FMR(fpr.V(dregs[i]), fpr.V(tempregs[i]));
			}
		}

		ApplyPrefixD(dregs, sz);
		fpr.ReleaseSpillLocksAndDiscardTemps();
	}
	void Jit::Comp_ColorConv(MIPSOpcode op) {
		DISABLE;
	}
	void Jit::Comp_Vbfy(MIPSOpcode op) {
		DISABLE;
	}
}

#endif
