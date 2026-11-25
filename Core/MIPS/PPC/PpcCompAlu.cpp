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

#include "PpcRegCache.h"
#include "ppcEmitter.h"
#include "PpcJit.h"

/***************************************************************************************************
*	Current issues:
*		Comp_RType3(min/max): Can't select start in disgaea
*		Comp_ShiftType(srl/srlv?): Crash ridge racer 2
***************************************************************************************************/


using namespace MIPSAnalyst;
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

// All functions should have CONDITIONAL_DISABLE, so we can narrow things down to a file quickly.
// Currently known non working ones should have DISABLE.

//#define CONDITIONAL_DISABLE { Comp_Generic(op); return; }
#define CONDITIONAL_DISABLE ;
#define DISABLE { Comp_Generic(op); return; }

namespace MIPSComp
{
using namespace PpcJitConstants;

	static u32 EvalOr(u32 a, u32 b) { return a | b; }
	static u32 EvalXor(u32 a, u32 b) { return a ^ b; }
	static u32 EvalAnd(u32 a, u32 b) { return a & b; }
	static u32 EvalAdd(u32 a, u32 b) { return a + b; }
	static u32 EvalSub(u32 a, u32 b) { return a - b; }
	static u32 EvalNor(u32 a, u32 b) { return ~(a | b); }

	// Utilities to reduce duplicated code
	void Jit::CompType3(int rd, int rs, int rt, void (PPCXEmitter::*arith)(PPCReg Rd, PPCReg Ra, PPCReg Rb), u32 (*eval)(u32 a, u32 b), bool isSub) {
		if (gpr.IsImm(rs) && gpr.IsImm(rt)) {
			gpr.SetImm(rd, (*eval)(gpr.GetImm(rs), gpr.GetImm(rt)));
		} else if (gpr.IsImm(rt)) {
			u32 rtImm = gpr.GetImm(rt);
			gpr.MapDirtyIn(rd, rs);
			MOVI2R(SREG, rtImm);
			(this->*arith)(gpr.R(rd), gpr.R(rs), SREG);
		} else if (gpr.IsImm(rs)) {
			u32 rsImm = gpr.GetImm(rs);
			gpr.MapDirtyIn(rd, rt);
			// TODO: Special case when rsImm can be represented as an Operand2
			MOVI2R(SREG, rsImm);
			(this->*arith)(gpr.R(rd), SREG, gpr.R(rt));
		} else {
			// Generic solution
			gpr.MapDirtyInIn(rd, rs, rt);
			(this->*arith)(gpr.R(rd), gpr.R(rs), gpr.R(rt));
		}
	}

	void Jit::CompImmLogic(int rs, int rt, u32 uimm, void (PPCXEmitter::*arith)(PPCReg Rd, PPCReg Ra, unsigned short imm), u32 (*eval)(u32 a, u32 b))
	{
		if (gpr.IsImm(rs)) {
			gpr.SetImm(rt, (*eval)(gpr.GetImm(rs), uimm));
		} else {
			gpr.MapDirtyIn(rt, rs);
			(this->*arith)(gpr.R(rt), gpr.R(rs), uimm);
		}
	}

	void Jit::Comp_IType(MIPSOpcode op)
	{
		CONDITIONAL_DISABLE;
		s32 simm = (s32)(s16)(op & 0xFFFF);  // sign extension
		u32 uimm = op & 0xFFFF;
		u32 suimm = (u32)(s32)simm;

		int rt = _RT;
		int rs = _RS;

		int o = op>>26;

		// noop, won't write to ZERO.
		if (rt == 0)
			return;

		switch (op >> 26) 
		{

		case 8:	// same as addiu?
		case 9:	// R(rt) = R(rs) + simm; break;	//addiu
			{
				if (gpr.IsImm(rs)) {
					gpr.SetImm(rt, gpr.GetImm(rs) + simm);
				} else {
					gpr.MapDirtyIn(rt, rs);
					ADDI(gpr.R(rt), gpr.R(rs), simm);
				}
				break;
			}

			// Use with caution can change CR0 !
		case 12: CompImmLogic(rs, rt, uimm, &PPCXEmitter::ANDI, &EvalAnd); break;
			// Safe
		case 13: CompImmLogic(rs, rt, uimm, &PPCXEmitter::ORI, &EvalOr); break;
		case 14: CompImmLogic(rs, rt, uimm, &PPCXEmitter::XORI, &EvalXor); break;
		case 15: // R(rt) = uimm << 16;	 //lui
			gpr.SetImm(rt, uimm << 16);
			break;

		case 10: // slti - R(rt) = (s32)R(rs) < simm
			if (gpr.IsImm(rs))
			{
				gpr.SetImm(rt, (s32)gpr.GetImm(rs) < simm);
				break;
			} else {
				//DISABLE;
				gpr.MapDirtyIn(rt, rs);

				PPCReg ppc_rt = gpr.R(rt);
				PPCReg ppc_rs = gpr.R(rs);

				ADDI(SREG, R0, uimm);

				SUBFC(R0, SREG, ppc_rs);
				EQV(ppc_rt, SREG, ppc_rs);
				SRWI(ppc_rt, ppc_rt, 31);
				ADDZE(ppc_rt, ppc_rt);
				RLWINM(ppc_rt, ppc_rt, 0, 31, 31);
				//Break();
				break;
			}

		case 11: //sltiu
			if (gpr.IsImm(rs))
			{
				gpr.SetImm(rt, gpr.GetImm(rs) < suimm);
				break;
			} else {
				//DISABLE;
				gpr.MapDirtyIn(rt, rs);

				PPCReg ppc_rt = gpr.R(rt);

				ADDI(SREG, R0, suimm);
				SUBFC(ppc_rt, SREG, gpr.R(rs));
				SUBFE(ppc_rt, ppc_rt, ppc_rt);
				NEG(ppc_rt, ppc_rt);

				break;
			}

		default:
			Comp_Generic(op);
			break;
		}
	}

	void Jit::Comp_RType2(MIPSOpcode op) {
		CONDITIONAL_DISABLE;
		MIPSGPReg rs = _RS;
		MIPSGPReg rd = _RD;
		if (rd == 0)
			return;

		switch (op & 63) {
		case 22: // clz
			if (gpr.IsImm(rs)) {
				u32 val = gpr.GetImm(rs);
				gpr.SetImm(rd, val ? __builtin_clz(val) : 32);
			} else {
				gpr.MapDirtyIn(rd, rs);
				CNTLZW(gpr.R(rd), gpr.R(rs));
			}
			break;
		case 23: // clo
			if (gpr.IsImm(rs)) {
				u32 val = ~gpr.GetImm(rs);
				gpr.SetImm(rd, val ? __builtin_clz(val) : 32);
			} else {
				gpr.MapDirtyIn(rd, rs);
				NOR(SREG, gpr.R(rs), gpr.R(rs));
				CNTLZW(gpr.R(rd), SREG);
			}
			break;
		default:
			Comp_Generic(op);
			break;
		}
	}


	void Jit::Comp_RType3(MIPSOpcode op) {
		CONDITIONAL_DISABLE;
		int rt = _RT;
		int rs = _RS;
		int rd = _RD;

		// noop, won't write to ZERO.
		if (rd == 0)
			return;

		u8 o = op & 63;

		switch (op & 63) 
		{
		case 10: // if (R(rt) == 0) R(rd) = R(rs); break; //movz
			if (rd == rs)
				break;
			if (!gpr.IsImm(rt))
			{
				gpr.MapDirtyInIn(rd, rt, rs, false);
				CMPI(gpr.R(rt), 0);
				PpcGen::FixupBranch ptr;

				ptr = B_Cond(_BNE);

				MR(gpr.R(rd), gpr.R(rs));

				SetJumpTarget(ptr);

			}
			else if (gpr.GetImm(rt) == 0)
			{
				// Yes, this actually happens.
				if (gpr.IsImm(rs))
					gpr.SetImm(rd, gpr.GetImm(rs));
				else
				{
					gpr.MapDirtyIn(rd, rs);
					MR(gpr.R(rd), gpr.R(rs));
				}
			}
			break;

		case 11:// if (R(rt) != 0) R(rd) = R(rs); break; //movn
			if (rd == rs)
				break;
			if (!gpr.IsImm(rt))
			{
				gpr.MapDirtyInIn(rd, rt, rs, false);
				CMPI(gpr.R(rt), 0);

				PpcGen::FixupBranch ptr;

				ptr = B_Cond(_BEQ);

				MR(gpr.R(rd), gpr.R(rs));

				SetJumpTarget(ptr);
			}
			else if (gpr.GetImm(rt) != 0)
			{
				// Yes, this actually happens.
				if (gpr.IsImm(rs))
					gpr.SetImm(rd, gpr.GetImm(rs));
				else
				{
					gpr.MapDirtyIn(rd, rs);
					MR(gpr.R(rd), gpr.R(rs));
				}
			}
			break;

		case 32: //R(rd) = R(rs) + R(rt);           break; //add
		case 33: //R(rd) = R(rs) + R(rt);           break; //addu
			// Some optimized special cases
			if (gpr.IsImm(rs) && gpr.IsImm(rt)) {
				gpr.SetImm(rd, gpr.GetImm(rs) + gpr.GetImm(rt));
			} else if (gpr.IsImm(rs) && (s32)gpr.GetImm(rs) >= -32768 && (s32)gpr.GetImm(rs) <= 32767) {
				s32 imm = (s32)gpr.GetImm(rs);
				if (imm == 0) {
					if (rd != rt) {
						gpr.MapDirtyIn(rd, rt);
						MR(gpr.R(rd), gpr.R(rt));
					}
				} else {
					gpr.MapDirtyIn(rd, rt);
					ADDI(gpr.R(rd), gpr.R(rt), imm);
				}
			} else if (gpr.IsImm(rt) && (s32)gpr.GetImm(rt) >= -32768 && (s32)gpr.GetImm(rt) <= 32767) {
				s32 imm = (s32)gpr.GetImm(rt);
				if (imm == 0) {
					if (rd != rs) {
						gpr.MapDirtyIn(rd, rs);
						MR(gpr.R(rd), gpr.R(rs));
					}
				} else {
					gpr.MapDirtyIn(rd, rs);
					ADDI(gpr.R(rd), gpr.R(rs), imm);
				}
			} else {
				CompType3(rd, rs, rt, &PPCXEmitter::ADD, &EvalAdd);
			}
			break;
		case 34: //R(rd) = R(rs) - R(rt);           break; //sub
		case 35: //R(rd) = R(rs) - R(rt);           break; //subu
			if (gpr.IsImm(rs) && gpr.IsImm(rt)) {
				gpr.SetImm(rd, gpr.GetImm(rs) - gpr.GetImm(rt));
			} else if (gpr.IsImm(rt) && gpr.GetImm(rt) == 0) {
				if (rd != rs) {
					gpr.MapDirtyIn(rd, rs);
					MR(gpr.R(rd), gpr.R(rs));
				}
			} else if (gpr.IsImm(rs) && gpr.GetImm(rs) == 0) {
				gpr.MapDirtyIn(rd, rt);
				NEG(gpr.R(rd), gpr.R(rt));
			} else {
				CompType3(rd, rs, rt, &PPCXEmitter::SUB, &EvalSub, true);
			}
			break;
		case 36: //R(rd) = R(rs) & R(rt);           break; //and
			if (gpr.IsImm(rs) && gpr.IsImm(rt)) {
				gpr.SetImm(rd, gpr.GetImm(rs) & gpr.GetImm(rt));
			} else if ((gpr.IsImm(rs) && gpr.GetImm(rs) == 0) || (gpr.IsImm(rt) && gpr.GetImm(rt) == 0)) {
				gpr.SetImm(rd, 0);
			} else {
				CompType3(rd, rs, rt, &PPCXEmitter::AND, &EvalAnd);
			}
			break;
		case 37: //R(rd) = R(rs) | R(rt);           break; //or
			if (gpr.IsImm(rs) && gpr.IsImm(rt)) {
				gpr.SetImm(rd, gpr.GetImm(rs) | gpr.GetImm(rt));
			} else if (gpr.IsImm(rt) && gpr.GetImm(rt) <= 0xFFFF) {
				u32 imm = gpr.GetImm(rt);
				if (imm == 0) {
					if (rd != rs) {
						gpr.MapDirtyIn(rd, rs);
						MR(gpr.R(rd), gpr.R(rs));
					}
				} else {
					gpr.MapDirtyIn(rd, rs);
					ORI(gpr.R(rd), gpr.R(rs), (u16)imm);
				}
			} else if (gpr.IsImm(rs) && gpr.GetImm(rs) <= 0xFFFF) {
				u32 imm = gpr.GetImm(rs);
				if (imm == 0) {
					if (rd != rt) {
						gpr.MapDirtyIn(rd, rt);
						MR(gpr.R(rd), gpr.R(rt));
					}
				} else {
					gpr.MapDirtyIn(rd, rt);
					ORI(gpr.R(rd), gpr.R(rt), (u16)imm);
				}
			} else {
				CompType3(rd, rs, rt, &PPCXEmitter::OR, &EvalOr);
			}
			break;
		case 38: //R(rd) = R(rs) ^ R(rt);           break; //xor/eor	
			if (gpr.IsImm(rs) && gpr.IsImm(rt)) {
				gpr.SetImm(rd, gpr.GetImm(rs) ^ gpr.GetImm(rt));
			} else if (gpr.IsImm(rt) && gpr.GetImm(rt) <= 0xFFFF) {
				u32 imm = gpr.GetImm(rt);
				if (imm == 0) {
					if (rd != rs) {
						gpr.MapDirtyIn(rd, rs);
						MR(gpr.R(rd), gpr.R(rs));
					}
				} else {
					gpr.MapDirtyIn(rd, rs);
					XORI(gpr.R(rd), gpr.R(rs), (u16)imm);
				}
			} else if (gpr.IsImm(rs) && gpr.GetImm(rs) <= 0xFFFF) {
				u32 imm = gpr.GetImm(rs);
				if (imm == 0) {
					if (rd != rt) {
						gpr.MapDirtyIn(rd, rt);
						MR(gpr.R(rd), gpr.R(rt));
					}
				} else {
					gpr.MapDirtyIn(rd, rt);
					XORI(gpr.R(rd), gpr.R(rt), (u16)imm);
				}
			} else {
				CompType3(rd, rs, rt, &PPCXEmitter::XOR, &EvalXor);
			}
			break;
			// Not tested !
		case 39: // R(rd) = ~(R(rs) | R(rt));       break; //nor
			CompType3(rd, rs, rt, &PPCXEmitter::NOR, &EvalNor);
			break;

		case 42: //R(rd) = (int)R(rs) < (int)R(rt); break; //slt
			if (gpr.IsImm(rs) && gpr.IsImm(rt)) {
				gpr.SetImm(rd, (s32)gpr.GetImm(rs) < (s32)gpr.GetImm(rt));
			} else {
				gpr.MapDirtyInIn(rd, rs, rt);

				PPCReg ppc_rd = gpr.R(rd);			
				PPCReg ppc_rs = gpr.R(rs);			
				PPCReg ppc_rt = gpr.R(rt);

				SUBFC(R0, ppc_rt, ppc_rs);
				EQV(ppc_rd, ppc_rt, ppc_rs);
				SRWI(ppc_rd, ppc_rd, 31);
				ADDZE(ppc_rd, ppc_rd);
				RLWINM(ppc_rd, ppc_rd, 0, 31, 31);
			}

			break; 

		case 43: //R(rd) = R(rs) < R(rt);           break; //sltu
			if (gpr.IsImm(rs) && gpr.IsImm(rt)) {
				gpr.SetImm(rd, gpr.GetImm(rs) < gpr.GetImm(rt));
			} else {
				gpr.MapDirtyInIn(rd, rs, rt);

				PPCReg ppc_rd = gpr.R(rd);

				SUBFC(ppc_rd, gpr.R(rt), gpr.R(rs));
				SUBFE(ppc_rd, ppc_rd, ppc_rd);
				NEG(ppc_rd, ppc_rd);
			}
			break;


		case 44:// R(rd) = ((s32)R(rs) > (s32)R(rt)) ? R(rs) : R(rt); break; //max
			if (gpr.IsImm(rs) && gpr.IsImm(rt))
				gpr.SetImm(rd, std::max((s32)gpr.GetImm(rs), (s32)gpr.GetImm(rt)));
			else
			{
				gpr.MapDirtyInIn(rd, rs, rt);
				PpcGen::FixupBranch useRs, end;

				CMP(gpr.R(rs), gpr.R(rt));
				useRs = B_Cond(_BGT);
				MR(gpr.R(rd), gpr.R(rt));
				end = B();
				SetJumpTarget(useRs);
				MR(gpr.R(rd), gpr.R(rs));
				SetJumpTarget(end);
			}
			break;

		case 45: //min
			if (gpr.IsImm(rs) && gpr.IsImm(rt))
				gpr.SetImm(rd, std::min((s32)gpr.GetImm(rs), (s32)gpr.GetImm(rt)));
			else
			{
				gpr.MapDirtyInIn(rd, rs, rt);
				PpcGen::FixupBranch useRs, end;

				CMP(gpr.R(rs), gpr.R(rt));
				useRs = B_Cond(_BLT);
				MR(gpr.R(rd), gpr.R(rt));
				end = B();
				SetJumpTarget(useRs);
				MR(gpr.R(rd), gpr.R(rs));
				SetJumpTarget(end);
			}
			break;


		default:
			Comp_Generic(op);
			break;
		}
	}

	void Jit::Comp_ShiftType(MIPSOpcode op) {
		CONDITIONAL_DISABLE;
		MIPSGPReg rs = _RS;
		MIPSGPReg rd = _RD;
		int	fd = _FD;
		MIPSGPReg rt = _RT;
		int sa = _SA;

		// noop, won't write to ZERO.
		if (rd == 0)
			return;

		// WARNING : ROTR
		switch (op & 0x3f)
		{
		case 0:  //sll
			if (gpr.IsImm(rt)) {
				gpr.SetImm(rd, gpr.GetImm(rt) << sa);
			} else {
				// Peephole: fuse MIPS sll+sra/srl sign/zero-extend sequences into 1 PPC instruction.
				if (!js.inDelaySlot && (sa == 16 || sa == 24)) {
					MIPSOpcode nextOp = Memory::Read_Opcode_JIT(js.compilerPC + 4);
					if ((nextOp >> 26) == 0 && (int)MIPS_GET_RT(nextOp) == rd && (int)MIPS_GET_RD(nextOp) == rd && (int)MIPS_GET_SA(nextOp) == sa) {
						u32 nfn = nextOp & 0x3f;
						if (nfn == 3) { // sra -> extsb / extsh
							EatInstruction(nextOp);
							gpr.MapDirtyIn(rd, rt);
							if (sa == 24)
								EXTSB(gpr.R(rd), gpr.R(rt));
							else
								EXTSH(gpr.R(rd), gpr.R(rt));
							break;
						} else if (nfn == 2 && MIPS_GET_RS(nextOp) == 0) { // srl -> zero-extend byte/half
							EatInstruction(nextOp);
							gpr.MapDirtyIn(rd, rt);
							RLWINM(gpr.R(rd), gpr.R(rt), 0, sa, 31);
							break;
						}
					}
				}
				gpr.MapDirtyIn(rd, rt);
				SLWI(gpr.R(rd), gpr.R(rt), sa);
			}
			break;

		case 2:
			if (rs == 0) // srl
			{
				if (gpr.IsImm(rt)) {
					gpr.SetImm(rd, gpr.GetImm(rt) >> sa);
				} else {
					gpr.MapDirtyIn(rd, rt);
					SRWI(gpr.R(rd), gpr.R(rt), sa);
				}
				break;
			}
			else if (rs == 1) // rotr
			{
				if (gpr.IsImm(rt)) {
					u32 v = gpr.GetImm(rt);
					gpr.SetImm(rd, (v >> sa) | (v << ((32 - sa) & 0x1f)));
				} else {
					gpr.MapDirtyIn(rd, rt);
					ROTRWI(gpr.R(rd), gpr.R(rt), sa);
				}
				break;
			}
			else
			{
				Comp_Generic(op);
				break;
			}

		case 3: //sra
			if (gpr.IsImm(rt)) {
				gpr.SetImm(rd, (u32)(((s32)gpr.GetImm(rt)) >> sa));
			} else {
				gpr.MapDirtyIn(rd, rt);
				SRAWI(gpr.R(rd), gpr.R(rt), sa);
			}
			break;

		case 4: //sllv
			if (gpr.IsImm(rs) && gpr.IsImm(rt)) {
				gpr.SetImm(rd, gpr.GetImm(rt) << (gpr.GetImm(rs) & 0x1F));
				break;
			}
			if (gpr.IsImm(rs))
			{
				int sa = gpr.GetImm(rs) & 0x1F;
				gpr.MapDirtyIn(rd, rt);
				SLWI(gpr.R(rd), gpr.R(rt), sa);
				break;
			}
			gpr.MapDirtyInIn(rd, rs, rt);
			ANDI(SREG, gpr.R(rs), 0x1F);
			SLW(gpr.R(rd), gpr.R(rt), SREG);
			break;

		case 6:
			if (fd == 0) { //srlv
				if (gpr.IsImm(rs) && gpr.IsImm(rt)) {
					gpr.SetImm(rd, gpr.GetImm(rt) >> (gpr.GetImm(rs) & 0x1F));
					break;
				}
				if (gpr.IsImm(rs))
				{
					int sa = gpr.GetImm(rs) & 0x1F;
					gpr.MapDirtyIn(rd, rt);
					SRWI(gpr.R(rd), gpr.R(rt), sa);
					break;
				} else {
					gpr.MapDirtyInIn(rd, rs, rt);
					ANDI(SREG, gpr.R(rs), 0x1F);
					SRW(gpr.R(rd), gpr.R(rt), SREG);
					break;
				}
			} else if (fd == 1) { // rotrv
				if (gpr.IsImm(rs))
				{
					int sa = gpr.GetImm(rs) & 0x1F;
					gpr.MapDirtyIn(rd, rt);
					ROTRWI(gpr.R(rd), gpr.R(rt), sa);
					break;
				}
				gpr.MapDirtyInIn(rd, rs, rt);
				ANDI(SREG, gpr.R(rs), 0x1F);
				ROTRW(gpr.R(rd), gpr.R(rt), SREG);
			} else {
				Comp_Generic(op);
			}
			break;

		case 7: //srav
			if (gpr.IsImm(rs) && gpr.IsImm(rt)) {
				gpr.SetImm(rd, (u32)(((s32)gpr.GetImm(rt)) >> (gpr.GetImm(rs) & 0x1F)));
				break;
			}
			if (gpr.IsImm(rs))
			{
				int sa = gpr.GetImm(rs) & 0x1F;
				gpr.MapDirtyIn(rd, rt);
				SRAWI(gpr.R(rd), gpr.R(rt), sa);
				break;
			}
			gpr.MapDirtyInIn(rd, rs, rt);
			ANDI(SREG, gpr.R(rs), 0x1F);
			SRAW(gpr.R(rd), gpr.R(rt), SREG);
			break;

		default:
			Comp_Generic(op);
			break;
		}
	}

	static u32 PpcBitrev32(u32 v) {
		v = ((v >> 1) & 0x55555555u) | ((v & 0x55555555u) << 1);
		v = ((v >> 2) & 0x33333333u) | ((v & 0x33333333u) << 2);
		v = ((v >> 4) & 0x0F0F0F0Fu) | ((v & 0x0F0F0F0Fu) << 4);
		v = ((v >> 8) & 0x00FF00FFu) | ((v & 0x00FF00FFu) << 8);
		return (v >> 16) | (v << 16);
	}

	void Jit::Comp_Allegrex(MIPSOpcode op) {
		CONDITIONAL_DISABLE;
		MIPSGPReg rt = _RT;
		MIPSGPReg rd = _RD;
		if (rd == 0)
			return;

		switch ((op >> 6) & 31) {
		case 16: // seb
			if (gpr.IsImm(rt)) {
				gpr.SetImm(rd, (u32)(s32)(s8)(u8)gpr.GetImm(rt));
			} else {
				gpr.MapDirtyIn(rd, rt);
				EXTSB(gpr.R(rd), gpr.R(rt));
			}
			break;
		case 20: // bitrev
			if (gpr.IsImm(rt)) {
				gpr.SetImm(rd, PpcBitrev32(gpr.GetImm(rt)));
			} else {
				gpr.MapDirtyIn(rd, rt);
				MR(R3, gpr.R(rt));
				QuickCallFunction((void *)&PpcBitrev32);
				MR(gpr.R(rd), R3);
			}
			break;
		case 24: // seh
			if (gpr.IsImm(rt)) {
				gpr.SetImm(rd, (u32)(s32)(s16)(u16)gpr.GetImm(rt));
			} else {
				gpr.MapDirtyIn(rd, rt);
				EXTSH(gpr.R(rd), gpr.R(rt));
			}
			break;
		default:
			Comp_Generic(op);
			break;
		}
	}

	static u32 PpcWsbh(u32 v) {
		return ((v & 0xFF00FF00u) >> 8) | ((v & 0x00FF00FFu) << 8);
	}

	static u32 PpcWsbw(u32 v) {
		return __builtin_bswap32(v);
	}

	void Jit::Comp_Allegrex2(MIPSOpcode op) {
		CONDITIONAL_DISABLE;
		MIPSGPReg rt = _RT;
		MIPSGPReg rd = _RD;
		if (rd == 0)
			return;

		switch (op & 0x3FF) {
		case 0xA0: // wsbh
			if (gpr.IsImm(rt)) {
				gpr.SetImm(rd, PpcWsbh(gpr.GetImm(rt)));
			} else {
				gpr.MapDirtyIn(rd, rt);
				MR(R3, gpr.R(rt));
				QuickCallFunction((void *)&PpcWsbh);
				MR(gpr.R(rd), R3);
			}
			break;
		case 0xE0: // wsbw
			if (gpr.IsImm(rt)) {
				gpr.SetImm(rd, PpcWsbw(gpr.GetImm(rt)));
			} else {
				gpr.MapDirtyIn(rd, rt);
				MR(R3, gpr.R(rt));
				QuickCallFunction((void *)&PpcWsbw);
				MR(gpr.R(rd), R3);
			}
			break;
		default:
			Comp_Generic(op);
			break;
		}
	}

	void Jit::Comp_MulDivType(MIPSOpcode op) {
		CONDITIONAL_DISABLE;
		MIPSGPReg rt = _RT;
		MIPSGPReg rs = _RS;
		MIPSGPReg rd = _RD;

		switch (op & 63) 
		{
		case 16: // R(rd) = HI; //mfhi
			if (rd != 0) {
				gpr.MapDirtyIn(rd, MIPSREG_HI);
				MR(gpr.R(rd), gpr.R(MIPSREG_HI));
			}
			break; 

		case 17: // HI = R(rs); //mthi
			gpr.MapDirtyIn(MIPSREG_HI, rs);
			MR(gpr.R(MIPSREG_HI), gpr.R(rs));
			break; 

		case 18: // R(rd) = LO; break; //mflo
			if (rd != 0) {
				gpr.MapDirtyIn(rd, MIPSREG_LO);
				MR(gpr.R(rd), gpr.R(MIPSREG_LO));
			}
			break;

		case 19: // LO = R(rs); break; //mtlo
			gpr.MapDirtyIn(MIPSREG_LO, rs);
			MR(gpr.R(MIPSREG_LO), gpr.R(rs));
			break; 

		case 24: //mult (the most popular one). lo,hi  = signed mul (rs * rt)
			gpr.MapDirtyDirtyInIn(MIPSREG_LO, MIPSREG_HI, rs, rt);
			MULLW(gpr.R(MIPSREG_LO), gpr.R(rs), gpr.R(rt));
			MULHW(gpr.R(MIPSREG_HI), gpr.R(rs), gpr.R(rt));
			break;

		case 25: //multu (2nd) lo,hi  = unsigned mul (rs * rt)
			gpr.MapDirtyDirtyInIn(MIPSREG_LO, MIPSREG_HI, rs, rt);
			MULLW(gpr.R(MIPSREG_LO), gpr.R(rs), gpr.R(rt));
			MULHWU(gpr.R(MIPSREG_HI), gpr.R(rs), gpr.R(rt));
			break;

		case 26: //div
			gpr.MapDirtyDirtyInIn(MIPSREG_LO, MIPSREG_HI, rs, rt);
			DIVW(gpr.R(MIPSREG_LO), gpr.R(rs), gpr.R(rt));
			MULLW(SREG, gpr.R(rt), gpr.R(MIPSREG_LO));
			SUB(gpr.R(MIPSREG_HI), gpr.R(rs), SREG);			
			break;

		case 27: //divu
			gpr.MapDirtyDirtyInIn(MIPSREG_LO, MIPSREG_HI, rs, rt);
			DIVWU(gpr.R(MIPSREG_LO), gpr.R(rs), gpr.R(rt));
			MULLW(SREG, gpr.R(rt), gpr.R(MIPSREG_LO));
			SUB(gpr.R(MIPSREG_HI), gpr.R(rs), SREG);
			break;

		case 28: //madd
			gpr.MapDirtyDirtyInIn(MIPSREG_LO, MIPSREG_HI, rs, rt, false);
			MULLW(SREG, gpr.R(rs), gpr.R(rt));
			MULHW(R0, gpr.R(rs), gpr.R(rt));
			ADDC(gpr.R(MIPSREG_LO), gpr.R(MIPSREG_LO), SREG);
			ADDZE(gpr.R(MIPSREG_HI), gpr.R(MIPSREG_HI));
			ADD(gpr.R(MIPSREG_HI), gpr.R(MIPSREG_HI), R0);
			break;

		case 29: //maddu
			gpr.MapDirtyDirtyInIn(MIPSREG_LO, MIPSREG_HI, rs, rt, false);
			MULLW(SREG, gpr.R(rs), gpr.R(rt));
			MULHWU(R0, gpr.R(rs), gpr.R(rt));
			ADDC(gpr.R(MIPSREG_LO), gpr.R(MIPSREG_LO), SREG);
			ADDZE(gpr.R(MIPSREG_HI), gpr.R(MIPSREG_HI));
			ADD(gpr.R(MIPSREG_HI), gpr.R(MIPSREG_HI), R0);
			break;

		case 46: // msub
			gpr.MapDirtyDirtyInIn(MIPSREG_LO, MIPSREG_HI, rs, rt, false);
			MULLW(SREG, gpr.R(rs), gpr.R(rt));
			MULHW(R0, gpr.R(rs), gpr.R(rt));
			SUBFC(gpr.R(MIPSREG_LO), SREG, gpr.R(MIPSREG_LO));
			SUBFE(gpr.R(MIPSREG_HI), R0, gpr.R(MIPSREG_HI));
			break;

		case 47: // msubu
			gpr.MapDirtyDirtyInIn(MIPSREG_LO, MIPSREG_HI, rs, rt, false);
			MULLW(SREG, gpr.R(rs), gpr.R(rt));
			MULHWU(R0, gpr.R(rs), gpr.R(rt));
			SUBFC(gpr.R(MIPSREG_LO), SREG, gpr.R(MIPSREG_LO));
			SUBFE(gpr.R(MIPSREG_HI), R0, gpr.R(MIPSREG_HI));
			break;

		default:
			Comp_Generic(op);
			break;
		}
	}

	void Jit::Comp_Special3(MIPSOpcode op) {
		CONDITIONAL_DISABLE;
		MIPSGPReg rs = _RS;
		MIPSGPReg rt = _RT;
		int pos = _POS;
		int size = _SIZE + 1;
		u32 mask = 0xFFFFFFFFUL >> (32 - size);

		if (rt == 0)
			return;

		switch (op & 0x3f) {
		case 0x0: // ext
			if (pos + size > 32) {
				Comp_Generic(op);
				return;
			}
			if (gpr.IsImm(rs)) {
				gpr.SetImm(rt, (gpr.GetImm(rs) >> pos) & mask);
				return;
			}
			gpr.MapDirtyIn(rt, rs);
			RLWINM(gpr.R(rt), gpr.R(rs), (32 - pos) & 0x1f, 32 - size, 31);
			break;

		case 0x4: // ins
			{
				int msb = _SIZE;
				int actualSize = (msb - pos) + 1;
				if (actualSize <= 0 || pos + actualSize > 32) {
					Comp_Generic(op);
					return;
				}
				u32 sourcemask = 0xFFFFFFFFUL >> (32 - actualSize);
				u32 destmask = sourcemask << pos;
				if (gpr.IsImm(rs)) {
					u32 inserted = (gpr.GetImm(rs) & sourcemask) << pos;
					if (gpr.IsImm(rt)) {
						gpr.SetImm(rt, (gpr.GetImm(rt) & ~destmask) | inserted);
						return;
					}
				}
				gpr.MapDirtyIn(rt, rs, false);
				RLWIMI(gpr.R(rt), gpr.R(rs), pos, 31 - msb, 31 - pos);
			}
			break;

		default:
			Comp_Generic(op);
			break;
		}
	}

}

#endif
