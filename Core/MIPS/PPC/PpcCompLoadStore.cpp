#include "ppsspp_config.h"
#if PPSSPP_ARCH(POWERPC)

#include "Common/Serialize/Serializer.h"
#include "Core/Config.h"
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


#define _RS   ((op>>21) & 0x1F)
#define _RT   ((op>>16) & 0x1F)
#define _RD   ((op>>11) & 0x1F)
#define _FS   ((op>>11) & 0x1F)
#define _FT   ((op>>16) & 0x1F)
#define _FD   ((op>>6 ) & 0x1F)
#define _POS  ((op>>6 ) & 0x1F)
#define _SIZE ((op>>11) & 0x1F)

// All functions should have CONDITIONAL_DISABLE, so we can narrow things down to a file quickly.
// Currently known non working ones should have DISABLE.

//#define CONDITIONAL_DISABLE { Comp_Generic(op); return; }
#define CONDITIONAL_DISABLE ;
#define DISABLE { Comp_Generic(op); return; }

using namespace PpcGen;
using namespace PpcJitConstants;

namespace MIPSComp
{

static u32 PpcMemLWL(u32 maskedAddr, u32 rtVal) {
	u32 shift = (maskedAddr & 3) * 8;
	u32 mem = Memory::ReadUnchecked_U32(maskedAddr & 0xFFFFFFFCu);
	return (rtVal & (0x00FFFFFFu >> shift)) | (mem << (24 - shift));
}

static u32 PpcMemLWR(u32 maskedAddr, u32 rtVal) {
	u32 shift = (maskedAddr & 3) * 8;
	u32 mem = Memory::ReadUnchecked_U32(maskedAddr & 0xFFFFFFFCu);
	return (rtVal & (0xFFFFFF00u << (24 - shift))) | (mem >> shift);
}

static void PpcMemSWL(u32 maskedAddr, u32 rtVal) {
	u32 shift = (maskedAddr & 3) * 8;
	u32 aligned = maskedAddr & 0xFFFFFFFCu;
	u32 mem = Memory::ReadUnchecked_U32(aligned);
	u32 result = (rtVal >> (24 - shift)) | (mem & (0xFFFFFF00u << shift));
	Memory::WriteUnchecked_U32(result, aligned);
}

static void PpcMemSWR(u32 maskedAddr, u32 rtVal) {
	u32 shift = (maskedAddr & 3) << 3;
	u32 aligned = maskedAddr & 0xFFFFFFFCu;
	u32 mem = Memory::ReadUnchecked_U32(aligned);
	u32 result = (rtVal << shift) | (mem & (0x00FFFFFFu >> (24 - shift)));
	Memory::WriteUnchecked_U32(result, aligned);
}

void Jit::SetRegToEffectiveAddress(PpcGen::PPCReg r, int rs, s16 offset) {
	if (offset) {
		ADDI(SREG, gpr.R(rs), offset);
		RLWINM(SREG, SREG, 0, 2, 31); // &= 0x3FFFFFFF
	} else {
		RLWINM(SREG, gpr.R(rs), 0, 2, 31); // &= 0x3FFFFFFF
	}
		
}
void Jit::Comp_ITypeMem(MIPSOpcode op) {
	CONDITIONAL_DISABLE;

		int offset = (signed short)(op&0xFFFF);
		bool load = false;
		int rt = _RT;
		int rs = _RS;
		int o = op>>26;
		if (((op >> 29) & 1) == 0 && rt == 0) {
			// Don't load anything into $zr
			return;
		}

		if (!g_Config.bFastMemory) {
			DISABLE;
		}

		u32 iaddr = gpr.IsImm(rs) ? offset + gpr.GetImm(rs) : 0xFFFFFFFF;
		bool doCheck = false;

		switch (o)
		{
		case 32: //lb
		case 33: //lh
		case 35: //lw
		case 36: //lbu
		case 37: //lhu
			load = true;
		case 40: //sb
		case 41: //sh
		case 43: //sw

			if (gpr.IsImm(rs) && Memory::IsValidAddress(iaddr)) {
				// We can compute the full address at compile time. Kickass.
				u32 addr = iaddr & 0x3FFFFFFF;
				// Must be OK even if rs == rt since we have the value from imm already.
				gpr.MapReg(rt, load ? MAP_NOINIT | MAP_DIRTY : 0);
				MOVI2R(SREG, addr);
			} else {
				_dbg_assert_msg_(!gpr.IsImm(rs), "Invalid immediate address?  CPU bug?");
				load ? gpr.MapDirtyIn(rt, rs) : gpr.MapInIn(rt, rs);
				
				SetRegToEffectiveAddress(SREG, rs, offset);
			}
			switch (o)
			{
			// Load
			case 32:  //lb
				LBZX(gpr.R(rt), BASEREG, SREG); 
				EXTSB(gpr.R(rt), gpr.R(rt));
				break;
			case 33:  //lh
				LHBRX(gpr.R(rt), BASEREG, SREG); 
				EXTSH(gpr.R(rt), gpr.R(rt));
				break;
			case 35: //lw
				LWBRX(gpr.R(rt), BASEREG, SREG); 
				break;
			case 36: //lbu
				LBZX (gpr.R(rt), BASEREG, SREG); 
				break;
			case 37: //lhu
				LHBRX (gpr.R(rt), BASEREG, SREG); 
				break;
			// Store
			case 40:  //sb
				STBX (gpr.R(rt), BASEREG, SREG); 
				break;
			case 41: //sh
				STHBRX(gpr.R(rt), BASEREG, SREG); 		
				break;
			case 43: //sw
				STWBRX(gpr.R(rt), BASEREG, SREG); 
				break;
			}
			break;
		case 34: //lwl
		case 38: //lwr
			load = true;
		case 42: //swl
		case 46: //swr
			if (!js.inDelaySlot) {
				// Optimisation: Combine to single unaligned load/store
				bool isLeft = (o == 34 || o == 42);
				MIPSOpcode nextOp = Memory::Read_Instruction(js.compilerPC + 4);
				// Find a matching shift in opposite direction with opposite offset.
				if (nextOp == (isLeft ? (op.encoding + (4<<26) - 3)
				                      : (op.encoding - (4<<26) + 3)))
				{
					EatInstruction(nextOp);
					nextOp = MIPSOpcode(((load ? 35 : 43) << 26) | ((isLeft ? nextOp : op) & 0x03FFFFFF)); //lw, sw
					Comp_ITypeMem(nextOp);
					return;
				}
			}
			{
				// Fast leaf helper (preserves all cached GPRs/FPRs in R14..R31/FPR14..FPR31).
				if (gpr.IsImm(rs) && Memory::IsValidAddress(iaddr)) {
					gpr.MapReg(rt, load ? MAP_DIRTY : 0);
					MOVI2R(R3, iaddr & 0x3FFFFFFF);
				} else {
					gpr.MapInIn(rt, rs);
					if (load)
						gpr.MapReg(rt, MAP_DIRTY);
					SetRegToEffectiveAddress(SREG, rs, offset);
					MR(R3, SREG);
				}
				MR(R4, gpr.R(rt));
				if (o == 34) {
					QuickCallFunction((void *)&PpcMemLWL);
					MR(gpr.R(rt), R3);
				} else if (o == 38) {
					QuickCallFunction((void *)&PpcMemLWR);
					MR(gpr.R(rt), R3);
				} else if (o == 42) {
					QuickCallFunction((void *)&PpcMemSWL);
				} else {
					QuickCallFunction((void *)&PpcMemSWR);
				}
			}
			break;
		default:
			Comp_Generic(op);
			return ;
		}
	}

	void Jit::Comp_Cache(MIPSOpcode op) {
		return;
	}
}
#endif
