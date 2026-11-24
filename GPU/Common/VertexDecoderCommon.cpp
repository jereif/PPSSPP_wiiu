// Copyright (c) 2013- PPSSPP Project.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2.0 or later versions.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License 2.0 for more details.

// A copy of the GPL 2.0 should have been included with the program.
// If not, see http://www.gnu.org/licenses/

// Official git repository and contact information can be found at
// https://github.com/hrydgard/ppsspp and http://www.ppsspp.org/.

#include <algorithm>
#include <cstdio>

#include "ppsspp_config.h"

#include "Common/Log.h"
#include "Common/CPUDetect.h"
#include "Common/ColorConv.h"
#include "Common/Common.h"
#include "Core/Config.h"
#include "Core/ConfigValues.h"
#include "Core/MemMap.h"
#include "Core/HDRemaster.h"
#include "Core/Reporting.h"
#include "Core/MIPS/JitCommon/JitCommon.h"
#include "Core/Util/AudioFormat.h"  // for clamp_u8
#include "GPU/Common/ShaderCommon.h"
#include "GPU/GPUState.h"
#include "GPU/ge_constants.h"
#include "GPU/Math3D.h"
#include "GPU/Common/VertexDecoderCommon.h"

static const u8 tcsize[4] = { 0, 2, 4, 8 }, tcalign[4] = { 0, 1, 2, 4 };
static const u8 colsize[8] = { 0, 0, 0, 0, 2, 2, 2, 4 }, colalign[8] = { 0, 0, 0, 0, 2, 2, 2, 4 };
static const u8 nrmsize[4] = { 0, 3, 6, 12 }, nrmalign[4] = { 0, 1, 2, 4 };
static const u8 possize[4] = { 3, 3, 6, 12 }, posalign[4] = { 1, 1, 2, 4 };
static const u8 wtsize[4] = { 0, 1, 2, 4 }, wtalign[4] = { 0, 1, 2, 4 };

// When software skinning. This array is only used when non-jitted - when jitted, the matrix
// is kept in registers.
alignas(16) static float skinMatrix[12];

// L1-cached 8-bit -> float LUTs: eliminates PowerPC int-to-float stack store-forwarding stalls.
static const float *GetU8Div128Lut() {
	static float lut[256];
	static bool inited = false;
	if (!inited) {
		for (int i = 0; i < 256; ++i)
			lut[i] = (float)i * (1.0f / 128.0f);
		inited = true;
	}
	return lut;
}

static const float *GetS8Div128Lut() {
	static float lut[256];
	static bool inited = false;
	if (!inited) {
		for (int i = 0; i < 256; ++i)
			lut[i] = (float)(s8)i * (1.0f / 128.0f);
		inited = true;
	}
	return lut;
}

#ifdef __wiiu__
// Fused in-register PowerPC 750CL Paired-Singles software skinning:
// Accumulates all non-zero bone matrices into fr0..fr7 (ZERO skinMatrix memory traffic)
// and transforms both position and optional normal directly from registers.
static inline void SkinPosNormPS(const float *weights, int nw, const float *boneBase,
	const float *inPos, float *outPos, const float *inNrm, float *outNrm) {
	const float zero = 0.0f;
	const float *wPtr = weights;
	const float *bPtr = boneBase;
	asm volatile(
		".machine push\n\t"
		".machine gekko\n\t"
		"ps_merge00 0, %[z], %[z]\n\t"
		"ps_merge00 1, %[z], %[z]\n\t"
		"ps_merge00 2, %[z], %[z]\n\t"
		"ps_merge00 3, %[z], %[z]\n\t"
		"ps_merge00 4, %[z], %[z]\n\t"
		"ps_merge00 5, %[z], %[z]\n\t"
		"ps_merge00 6, %[z], %[z]\n\t"
		"ps_merge00 7, %[z], %[z]\n\t"
		"mtctr     %[nw]\n\t"
		"1:\n\t"
		"lfs       12, 0(%[w])\n\t"
		"addi      %[w], %[w], 4\n\t"
		"fcmpu     0, 12, %[z]\n\t"
		"beq       0, 2f\n\t"
		"psq_l      8,  0(%[b]), 0, 0\n\t"
		"lfs        9,  8(%[b])\n\t"
		"psq_l     10, 12(%[b]), 0, 0\n\t"
		"lfs       11, 20(%[b])\n\t"
		"ps_madds0  0,  8, 12, 0\n\t"
		"ps_madds0  1,  9, 12, 1\n\t"
		"ps_madds0  2, 10, 12, 2\n\t"
		"ps_madds0  3, 11, 12, 3\n\t"
		"psq_l      8, 24(%[b]), 0, 0\n\t"
		"lfs        9, 32(%[b])\n\t"
		"psq_l     10, 36(%[b]), 0, 0\n\t"
		"lfs       11, 44(%[b])\n\t"
		"ps_madds0  4,  8, 12, 4\n\t"
		"ps_madds0  5,  9, 12, 5\n\t"
		"ps_madds0  6, 10, 12, 6\n\t"
		"ps_madds0  7, 11, 12, 7\n\t"
		"2:\n\t"
		"addi      %[b], %[b], 48\n\t"
		"bdnz      1b\n\t"
		"lfs       10, 0(%[ip])\n\t"
		"lfs       11, 4(%[ip])\n\t"
		"lfs       12, 8(%[ip])\n\t"
		"ps_madds0  8, 0, 10, 6\n\t"
		"ps_madds0  9, 1, 10, 7\n\t"
		"ps_madds0  8, 2, 11, 8\n\t"
		"ps_madds0  9, 3, 11, 9\n\t"
		"ps_madds0  8, 4, 12, 8\n\t"
		"ps_madds0  9, 5, 12, 9\n\t"
		"psq_st     8, 0(%[op]), 0, 0\n\t"
		"stfs       9, 8(%[op])\n\t"
		"cmpwi     0, %[in], 0\n\t"
		"beq       0, 3f\n\t"
		"lfs       10, 0(%[in])\n\t"
		"lfs       11, 4(%[in])\n\t"
		"lfs       12, 8(%[in])\n\t"
		"ps_muls0   8, 0, 10\n\t"
		"ps_muls0   9, 1, 10\n\t"
		"ps_madds0  8, 2, 11, 8\n\t"
		"ps_madds0  9, 3, 11, 9\n\t"
		"ps_madds0  8, 4, 12, 8\n\t"
		"ps_madds0  9, 5, 12, 9\n\t"
		"psq_st     8, 0(%[on]), 0, 0\n\t"
		"stfs       9, 8(%[on])\n\t"
		"3:\n\t"
		".machine pop\n\t"
		: [w] "+b"(wPtr), [b] "+b"(bPtr)
		: [nw] "r"(nw), [z] "f"(zero),
		  [ip] "b"(inPos), [op] "b"(outPos),
		  [in] "b"(inNrm), [on] "b"(outNrm)
		: "fr0", "fr1", "fr2", "fr3", "fr4", "fr5", "fr6", "fr7",
		  "fr8", "fr9", "fr10", "fr11", "fr12", "cr0", "ctr", "memory");
}

// Direct u8-weight fused in-register Paired-Singles software skinning:
// Skips zero weights in 2 integer cycles (no stack weights[8], no LUT load, no FPU compare),
// receives pos/nrm directly in FPRs (zero stack round-trip), accumulates bones in fr0..fr7.
template <bool HAS_NRM>
static inline void SkinPosNormU8PS(
	const u8 *wPtr, int nw, const float *u8Lut, const float *boneBase,
	float px, float py, float pz, float *outPos,
	float nx, float ny, float nz, float *outNrm) {
	const float zero = 0.0f;
	const u8 *w = wPtr;
	const float *b = boneBase;
	u32 tmp = 0;
	if (HAS_NRM) {
		asm volatile(
			".machine push\n\t"
			".machine gekko\n\t"
			"ps_merge00 0, %[z], %[z]\n\t"
			"ps_merge00 1, %[z], %[z]\n\t"
			"ps_merge00 2, %[z], %[z]\n\t"
			"ps_merge00 3, %[z], %[z]\n\t"
			"ps_merge00 4, %[z], %[z]\n\t"
			"ps_merge00 5, %[z], %[z]\n\t"
			"ps_merge00 6, %[z], %[z]\n\t"
			"ps_merge00 7, %[z], %[z]\n\t"
			"mtctr     %[nw]\n\t"
			"1:\n\t"
			"lbz       %[tmp], 0(%[w])\n\t"
			"addi      %[w], %[w], 1\n\t"
			"cmpwi     0, %[tmp], 0\n\t"
			"beq       0, 2f\n\t"
			"slwi      %[tmp], %[tmp], 2\n\t"
			"lfsx      12, %[lut], %[tmp]\n\t"
			"psq_l      8,  0(%[b]), 0, 0\n\t"
			"lfs        9,  8(%[b])\n\t"
			"psq_l     10, 12(%[b]), 0, 0\n\t"
			"lfs       11, 20(%[b])\n\t"
			"ps_madds0  0,  8, 12, 0\n\t"
			"ps_madds0  1,  9, 12, 1\n\t"
			"ps_madds0  2, 10, 12, 2\n\t"
			"ps_madds0  3, 11, 12, 3\n\t"
			"psq_l      8, 24(%[b]), 0, 0\n\t"
			"lfs        9, 32(%[b])\n\t"
			"psq_l     10, 36(%[b]), 0, 0\n\t"
			"lfs       11, 44(%[b])\n\t"
			"ps_madds0  4,  8, 12, 4\n\t"
			"ps_madds0  5,  9, 12, 5\n\t"
			"ps_madds0  6, 10, 12, 6\n\t"
			"ps_madds0  7, 11, 12, 7\n\t"
			"2:\n\t"
			"addi      %[b], %[b], 48\n\t"
			"bdnz      1b\n\t"
			"ps_madds0  8, 0, %[px], 6\n\t"
			"ps_madds0  9, 1, %[px], 7\n\t"
			"ps_madds0  8, 2, %[py], 8\n\t"
			"ps_madds0  9, 3, %[py], 9\n\t"
			"ps_madds0  8, 4, %[pz], 8\n\t"
			"ps_madds0  9, 5, %[pz], 9\n\t"
			"psq_st     8, 0(%[op]), 0, 0\n\t"
			"stfs       9, 8(%[op])\n\t"
			"ps_muls0   8, 0, %[nx]\n\t"
			"ps_muls0   9, 1, %[nx]\n\t"
			"ps_madds0  8, 2, %[ny], 8\n\t"
			"ps_madds0  9, 3, %[ny], 9\n\t"
			"ps_madds0  8, 4, %[nz], 8\n\t"
			"ps_madds0  9, 5, %[nz], 9\n\t"
			"psq_st     8, 0(%[on]), 0, 0\n\t"
			"stfs       9, 8(%[on])\n\t"
			".machine pop\n\t"
			: [w] "+b"(w), [b] "+b"(b), [tmp] "=&r"(tmp)
			: [nw] "r"(nw), [lut] "b"(u8Lut), [z] "f"(zero),
			  [px] "f"(px), [py] "f"(py), [pz] "f"(pz), [op] "b"(outPos),
			  [nx] "f"(nx), [ny] "f"(ny), [nz] "f"(nz), [on] "b"(outNrm)
			: "fr0", "fr1", "fr2", "fr3", "fr4", "fr5", "fr6", "fr7",
			  "fr8", "fr9", "fr10", "fr11", "fr12", "cr0", "ctr", "memory");
	} else {
		asm volatile(
			".machine push\n\t"
			".machine gekko\n\t"
			"ps_merge00 0, %[z], %[z]\n\t"
			"ps_merge00 1, %[z], %[z]\n\t"
			"ps_merge00 2, %[z], %[z]\n\t"
			"ps_merge00 3, %[z], %[z]\n\t"
			"ps_merge00 4, %[z], %[z]\n\t"
			"ps_merge00 5, %[z], %[z]\n\t"
			"ps_merge00 6, %[z], %[z]\n\t"
			"ps_merge00 7, %[z], %[z]\n\t"
			"mtctr     %[nw]\n\t"
			"1:\n\t"
			"lbz       %[tmp], 0(%[w])\n\t"
			"addi      %[w], %[w], 1\n\t"
			"cmpwi     0, %[tmp], 0\n\t"
			"beq       0, 2f\n\t"
			"slwi      %[tmp], %[tmp], 2\n\t"
			"lfsx      12, %[lut], %[tmp]\n\t"
			"psq_l      8,  0(%[b]), 0, 0\n\t"
			"lfs        9,  8(%[b])\n\t"
			"psq_l     10, 12(%[b]), 0, 0\n\t"
			"lfs       11, 20(%[b])\n\t"
			"ps_madds0  0,  8, 12, 0\n\t"
			"ps_madds0  1,  9, 12, 1\n\t"
			"ps_madds0  2, 10, 12, 2\n\t"
			"ps_madds0  3, 11, 12, 3\n\t"
			"psq_l      8, 24(%[b]), 0, 0\n\t"
			"lfs        9, 32(%[b])\n\t"
			"psq_l     10, 36(%[b]), 0, 0\n\t"
			"lfs       11, 44(%[b])\n\t"
			"ps_madds0  4,  8, 12, 4\n\t"
			"ps_madds0  5,  9, 12, 5\n\t"
			"ps_madds0  6, 10, 12, 6\n\t"
			"ps_madds0  7, 11, 12, 7\n\t"
			"2:\n\t"
			"addi      %[b], %[b], 48\n\t"
			"bdnz      1b\n\t"
			"ps_madds0  8, 0, %[px], 6\n\t"
			"ps_madds0  9, 1, %[px], 7\n\t"
			"ps_madds0  8, 2, %[py], 8\n\t"
			"ps_madds0  9, 3, %[py], 9\n\t"
			"ps_madds0  8, 4, %[pz], 8\n\t"
			"ps_madds0  9, 5, %[pz], 9\n\t"
			"psq_st     8, 0(%[op]), 0, 0\n\t"
			"stfs       9, 8(%[op])\n\t"
			".machine pop\n\t"
			: [w] "+b"(w), [b] "+b"(b), [tmp] "=&r"(tmp)
			: [nw] "r"(nw), [lut] "b"(u8Lut), [z] "f"(zero),
			  [px] "f"(px), [py] "f"(py), [pz] "f"(pz), [op] "b"(outPos)
			: "fr0", "fr1", "fr2", "fr3", "fr4", "fr5", "fr6", "fr7",
			  "fr8", "fr9", "fr10", "fr11", "fr12", "cr0", "ctr", "memory");
	}
}

// Zero-branch specialized inner loop for hot u8-weight software-skinned vertex formats.
template <int TC, int COL, int NRM, int POS>
static void DecodeSkinnedU8T(
	const u8 * __restrict p, u8 * __restrict d, int count,
	int inSize, int stride, int nw,
	int tcOff, int uvOff, float uMul, float vMul, float uAdd, float vAdd,
	int cOff, int dc0Off, int nOff, int dnOff, int pOff, int dpOff,
	bool &fullAlpha) {
	const float * __restrict u8Lut = GetU8Div128Lut();
	const float * __restrict s8Lut = (NRM == 1 || POS == 1) ? GetS8Div128Lut() : nullptr;
	const float * __restrict boneBase = gstate.boneMatrix;
	for (; count > 0; --count) {
		if (TC == 2) {
			float *uv = (float *)(d + uvOff);
			const u16_le *uv16 = (const u16_le *)(p + tcOff);
			uv[0] = (float)uv16[0] * uMul + uAdd;
			uv[1] = (float)uv16[1] * vMul + vAdd;
		} else if (TC == 1) {
			float *uv = (float *)(d + uvOff);
			uv[0] = u8Lut[p[tcOff]] * uMul + uAdd;
			uv[1] = u8Lut[p[tcOff + 1]] * vMul + vAdd;
		} else if (TC == 3) {
			float *uv = (float *)(d + uvOff);
			const float_le *uvf = (const float_le *)(p + tcOff);
			uv[0] = uvf[0] * uMul + uAdd;
			uv[1] = uvf[1] * vMul + vAdd;
		}
		if (COL == 7) {
			u32 cdata = *(const u32_le *)(p + cOff);
			fullAlpha &= ((cdata >> 24) == 0xFF);
			*(u32_le *)(d + dc0Off) = cdata;
		}
		float nx = 0.0f, ny = 0.0f, nz = 0.0f;
		if (NRM == 1) {
			const u8 *sv = p + nOff;
			nx = s8Lut[sv[0]];
			ny = s8Lut[sv[1]];
			nz = s8Lut[sv[2]];
		} else if (NRM == 2) {
			const s16_le *sv = (const s16_le *)(p + nOff);
			nx = (float)sv[0] * (1.0f / 32768.0f);
			ny = (float)sv[1] * (1.0f / 32768.0f);
			nz = (float)sv[2] * (1.0f / 32768.0f);
		} else if (NRM == 3) {
			const float_le *fv = (const float_le *)(p + nOff);
			nx = fv[0];
			ny = fv[1];
			nz = fv[2];
		}
		float px, py, pz;
		if (POS == 2) {
			const s16_le *pv = (const s16_le *)(p + pOff);
			px = (float)pv[0] * (1.0f / 32768.0f);
			py = (float)pv[1] * (1.0f / 32768.0f);
			pz = (float)pv[2] * (1.0f / 32768.0f);
		} else if (POS == 3) {
			const float_le *pv = (const float_le *)(p + pOff);
			px = pv[0];
			py = pv[1];
			pz = pv[2];
		} else {
			const u8 *pv = p + pOff;
			px = s8Lut[pv[0]];
			py = s8Lut[pv[1]];
			pz = s8Lut[pv[2]];
		}
		SkinPosNormU8PS<NRM != 0>(p, nw, u8Lut, boneBase,
			px, py, pz, (float *)(d + dpOff),
			nx, ny, nz, (float *)(d + dnOff));
		p += inSize;
		d += stride;
	}
}
#endif

// Zero-branch specialized inner loop (2x unrolled to hide PPC int->float store-forwarding stalls).
template <int TC, int COL, int NRM, int POS>
static void DecodeNonSkinnedT(
	const u8 * __restrict p, u8 * __restrict d, int count,
	int inSize, int stride,
	int tcOff, int uvOff, float uMul, float vMul, float uAdd, float vAdd,
	int cOff, int dc0Off, int nOff, int dnOff, int pOff, int dpOff,
	bool &fullAlpha) {
	const float * __restrict s8Lut = (NRM == 1 || POS == 1) ? GetS8Div128Lut() : nullptr;
	const float * __restrict u8Lut = (TC == 1) ? GetU8Div128Lut() : nullptr;
	auto decodeOne = [&](const u8 * __restrict sp, u8 * __restrict sd) {
		if (TC == 2) {
			float *uv = (float *)(sd + uvOff);
			const u16_le *uv16 = (const u16_le *)(sp + tcOff);
			uv[0] = (float)uv16[0] * uMul + uAdd;
			uv[1] = (float)uv16[1] * vMul + vAdd;
		} else if (TC == 1) {
			float *uv = (float *)(sd + uvOff);
			uv[0] = u8Lut[sp[tcOff]] * uMul + uAdd;
			uv[1] = u8Lut[sp[tcOff + 1]] * vMul + vAdd;
		} else if (TC == 3) {
			float *uv = (float *)(sd + uvOff);
			const float_le *uvf = (const float_le *)(sp + tcOff);
			uv[0] = uvf[0] * uMul + uAdd;
			uv[1] = uvf[1] * vMul + vAdd;
		}
		if (COL == 7) {
			u32 cdata = *(const u32_le *)(sp + cOff);
			fullAlpha &= ((cdata >> 24) == 0xFF);
			*(u32_le *)(sd + dc0Off) = cdata;
		} else if (COL == 6) {
			u16 cdata = *(const u16_le *)(sp + cOff);
			fullAlpha &= ((cdata >> 12) == 0xF);
			*(u32_le *)(sd + dc0Off) = Convert4To8(cdata & 0xF) |
				(Convert4To8((cdata >> 4) & 0xF) << 8) |
				(Convert4To8((cdata >> 8) & 0xF) << 16) |
				(Convert4To8((cdata >> 12) & 0xF) << 24);
		} else if (COL == 5) {
			u16 cdata = *(const u16_le *)(sp + cOff);
			fullAlpha &= ((cdata >> 15) != 0);
			*(u32_le *)(sd + dc0Off) = Convert5To8(cdata & 0x1f) |
				(Convert5To8((cdata >> 5) & 0x1f) << 8) |
				(Convert5To8((cdata >> 10) & 0x1f) << 16) |
				((cdata >> 15) ? (255u << 24) : 0);
		} else if (COL == 4) {
			u16 cdata = *(const u16_le *)(sp + cOff);
			*(u32_le *)(sd + dc0Off) = Convert5To8(cdata & 0x1f) |
				(Convert6To8((cdata >> 5) & 0x3f) << 8) |
				(Convert5To8((cdata >> 11) & 0x1f) << 16) |
				(255u << 24);
		}
		if (NRM == 1) {
			float *no = (float *)(sd + dnOff);
			const u8 *sv = sp + nOff;
			no[0] = s8Lut[sv[0]];
			no[1] = s8Lut[sv[1]];
			no[2] = s8Lut[sv[2]];
		} else if (NRM == 2) {
			s16 *no = (s16 *)(sd + dnOff);
			const s16_le *sv = (const s16_le *)(sp + nOff);
			no[0] = sv[0];
			no[1] = sv[1];
			no[2] = sv[2];
			no[3] = 0;
		} else if (NRM == 3) {
			u32 *no = (u32 *)(sd + dnOff);
			const u32_le *fv = (const u32_le *)(sp + nOff);
			no[0] = fv[0];
			no[1] = fv[1];
			no[2] = fv[2];
		}
		if (POS == 2) {
			float *po = (float *)(sd + dpOff);
			const s16_le *pv = (const s16_le *)(sp + pOff);
			po[0] = (float)pv[0] * (1.0f / 32768.0f);
			po[1] = (float)pv[1] * (1.0f / 32768.0f);
			po[2] = (float)pv[2] * (1.0f / 32768.0f);
		} else if (POS == 3) {
			u32 *po = (u32 *)(sd + dpOff);
			const u32_le *pv = (const u32_le *)(sp + pOff);
			po[0] = pv[0];
			po[1] = pv[1];
			po[2] = pv[2];
		} else {
			float *po = (float *)(sd + dpOff);
			const u8 *pv = sp + pOff;
			po[0] = s8Lut[pv[0]];
			po[1] = s8Lut[pv[1]];
			po[2] = s8Lut[pv[2]];
		}
	};
	for (; count >= 2; count -= 2) {
		decodeOne(p, d);
		decodeOne(p + inSize, d + stride);
		p += inSize * 2;
		d += stride * 2;
	}
	if (count) {
		decodeOne(p, d);
	}
}

inline int align(int n, int align) {
	return (n + (align - 1)) & ~(align - 1);
}

int TranslateNumBones(int bones) {
	if (!bones) return 0;
	if (bones < 4) return 4;
	// if (bones < 8) return 8;   I get drawing problems in FF:CC with this!
	return bones;
}

int DecFmtSize(u8 fmt) {
	switch (fmt) {
	case DEC_NONE: return 0;
	case DEC_FLOAT_1: return 4;
	case DEC_FLOAT_2: return 8;
	case DEC_FLOAT_3: return 12;
	case DEC_FLOAT_4: return 16;
	case DEC_S8_3: return 4;
	case DEC_S16_3: return 8;
	case DEC_U8_1: return 4;
	case DEC_U8_2: return 4;
	case DEC_U8_3: return 4;
	case DEC_U8_4: return 4;
	case DEC_U16_1: return 4;
	case DEC_U16_2: return 4;
	case DEC_U16_3: return 8;
	case DEC_U16_4: return 8;
	default:
		return 0;
	}
}

void DecVtxFormat::ComputeID() {
	id = w0fmt | (w1fmt << 4) | (uvfmt << 8) | (c0fmt << 12) | (c1fmt << 16) | (nrmfmt << 20) | (posfmt << 24);
}

void DecVtxFormat::InitializeFromID(uint32_t id) {
	this->id = id;
	w0fmt = ((id) & 0xF);
	w1fmt = ((id >> 4) & 0xF);
	uvfmt = ((id >> 8) & 0xF);
	c0fmt = ((id >> 12) & 0xF);
	c1fmt = ((id >> 16) & 0xF);
	nrmfmt = ((id >> 20) & 0xF);
	posfmt = ((id >> 24) & 0xF);
	w0off = 0;
	w1off = w0off + DecFmtSize(w0fmt);
	uvoff = w1off + DecFmtSize(w1fmt);
	c0off = uvoff + DecFmtSize(uvfmt);
	c1off = c0off + DecFmtSize(c0fmt);
	nrmoff = c1off + DecFmtSize(c1fmt);
	posoff = nrmoff + DecFmtSize(nrmfmt);
	stride = posoff + DecFmtSize(posfmt);
}

void GetIndexBounds(const void *inds, int count, u32 vertType, u16 *indexLowerBound, u16 *indexUpperBound) {
	// Find index bounds. Could cache this in display lists.
	// Also, this could be greatly sped up with SSE2/NEON, although rarely a bottleneck.
	int lowerBound = 0x7FFFFFFF;
	int upperBound = 0;
	u32 idx = vertType & GE_VTYPE_IDX_MASK;
	if (idx == GE_VTYPE_IDX_8BIT) {
		const u8 *ind8 = (const u8 *)inds;
		int i = 0;
		for (; i + 4 <= count; i += 4) {
			int v0 = ind8[i + 0];
			int v1 = ind8[i + 1];
			int v2 = ind8[i + 2];
			int v3 = ind8[i + 3];
			int lo01 = v0 < v1 ? v0 : v1;
			int hi01 = v0 > v1 ? v0 : v1;
			int lo23 = v2 < v3 ? v2 : v3;
			int hi23 = v2 > v3 ? v2 : v3;
			int lo = lo01 < lo23 ? lo01 : lo23;
			int hi = hi01 > hi23 ? hi01 : hi23;
			if (lo < lowerBound) lowerBound = lo;
			if (hi > upperBound) upperBound = hi;
		}
		for (; i < count; i++) {
			int value = ind8[i];
			if (value > upperBound) upperBound = value;
			if (value < lowerBound) lowerBound = value;
		}
	} else if (idx == GE_VTYPE_IDX_16BIT) {
		const u16_le *ind16 = (const u16_le *)inds;
		int i = 0;
		for (; i + 4 <= count; i += 4) {
			int v0 = ind16[i + 0];
			int v1 = ind16[i + 1];
			int v2 = ind16[i + 2];
			int v3 = ind16[i + 3];
			int lo01 = v0 < v1 ? v0 : v1;
			int hi01 = v0 > v1 ? v0 : v1;
			int lo23 = v2 < v3 ? v2 : v3;
			int hi23 = v2 > v3 ? v2 : v3;
			int lo = lo01 < lo23 ? lo01 : lo23;
			int hi = hi01 > hi23 ? hi01 : hi23;
			if (lo < lowerBound) lowerBound = lo;
			if (hi > upperBound) upperBound = hi;
		}
		for (; i < count; i++) {
			int value = ind16[i];
			if (value > upperBound) upperBound = value;
			if (value < lowerBound) lowerBound = value;
		}
	} else if (idx == GE_VTYPE_IDX_32BIT) {
		WARN_LOG_REPORT_ONCE(indexBounds32, G3D, "GetIndexBounds: Decoding 32-bit indexes");
		const u32_le *ind32 = (const u32_le *)inds;
		for (int i = 0; i < count; i++) {
			u16 value = (u16)ind32[i];
			// These aren't documented and should be rare.  Let's bounds check each one.
			if (ind32[i] != value) {
				ERROR_LOG_REPORT_ONCE(indexBounds32Bounds, G3D, "GetIndexBounds: Index outside 16-bit range");
			}
			if (value > upperBound)
				upperBound = value;
			if (value < lowerBound)
				lowerBound = value;
		}
	} else {
		lowerBound = 0;
		upperBound = count - 1;
	}
	*indexLowerBound = (u16)lowerBound;
	*indexUpperBound = (u16)upperBound;
}

void PrintDecodedVertex(VertexReader &vtx) {
	if (vtx.hasNormal()) {
		float nrm[3];
		vtx.ReadNrm(nrm);
		printf("N: %f %f %f\n", nrm[0], nrm[1], nrm[2]);
	}
	if (vtx.hasUV()) {
		float uv[2];
		vtx.ReadUV(uv);
		printf("TC: %f %f\n", uv[0], uv[1]);
	}
	if (vtx.hasColor0()) {
		float col0[4];
		vtx.ReadColor0(col0);
		printf("C0: %f %f %f %f\n", col0[0], col0[1], col0[2], col0[3]);
	}
	if (vtx.hasColor1()) {
		float col1[3];
		vtx.ReadColor1(col1);
		printf("C1: %f %f %f\n", col1[0], col1[1], col1[2]);
	}
	// Etc..
	float pos[3];
	vtx.ReadPos(pos);
	printf("P: %f %f %f\n", pos[0], pos[1], pos[2]);
}

VertexDecoder::VertexDecoder() : decoded_(nullptr), ptr_(nullptr), jitted_(0), jittedSize_(0) {
}

void VertexDecoder::Step_WeightsU8() const
{
	u8 *wt = (u8 *)(decoded_ + decFmt.w0off);
	const u8 *wdata = (const u8*)(ptr_);
	int j;
	for (j = 0; j < nweights; j++)
		wt[j] = wdata[j];
	while (j & 3)   // Zero additional weights rounding up to 4.
		wt[j++] = 0;
}

void VertexDecoder::Step_WeightsU16() const
{
	u16 *wt = (u16 *)(decoded_ + decFmt.w0off);
	const u16_le *wdata = (const u16_le*)(ptr_);
	int j;
	for (j = 0; j < nweights; j++)
		wt[j] = wdata[j];
	while (j & 3)   // Zero additional weights rounding up to 4.
		wt[j++] = 0;
}

void VertexDecoder::Step_WeightsU8ToFloat() const
{
	float *wt = (float *)(decoded_ + decFmt.w0off);
	const u8 *wdata = (const u8*)(ptr_);
	int j;
	for (j = 0; j < nweights; j++) {
		wt[j] = (float)wdata[j] * (1.0f / 128.0f);
	}
	while (j & 3)   // Zero additional weights rounding up to 4.
		wt[j++] = 0;
}

void VertexDecoder::Step_WeightsU16ToFloat() const
{
	float *wt = (float *)(decoded_ + decFmt.w0off);
	const u16_le *wdata = (const u16_le*)(ptr_);
	int j;
	for (j = 0; j < nweights; j++) {
		wt[j] = (float)wdata[j] * (1.0f / 32768.0f);
	}
	while (j & 3)   // Zero additional weights rounding up to 4.
		wt[j++] = 0;
}

// Float weights should be uncommon, we can live with having to multiply these by 2.0
// to avoid special checks in the vertex shader generator.
// (PSP uses 0.0-2.0 fixed point numbers for weights)
void VertexDecoder::Step_WeightsFloat() const
{
	float *wt = (float *)(decoded_ + decFmt.w0off);
	const float_le *wdata = (const float_le*)(ptr_);
	int j;
	for (j = 0; j < nweights; j++) {
		wt[j] = wdata[j];
	}
	while (j & 3)   // Zero additional weights rounding up to 4.
		wt[j++] = 0.0f;
}

void VertexDecoder::ComputeSkinMatrix(const float weights[8]) const {
#ifdef __wiiu__
	bool first = true;
	for (int j = 0; j < nweights; j++) {
		const float w = weights[j];
		if (w != 0.0f) {
			const float *bone = &gstate.boneMatrix[j * 12];
			if (first) {
				first = false;
				asm volatile(
					".machine push\n\t"
					".machine gekko\n\t"
					"psq_l   0,  0(%0), 0, 0\n\t"
					"psq_l   1,  8(%0), 0, 0\n\t"
					"psq_l   2, 16(%0), 0, 0\n\t"
					"psq_l   3, 24(%0), 0, 0\n\t"
					"psq_l   4, 32(%0), 0, 0\n\t"
					"psq_l   5, 40(%0), 0, 0\n\t"
					"ps_muls0 0, 0, %2\n\t"
					"ps_muls0 1, 1, %2\n\t"
					"ps_muls0 2, 2, %2\n\t"
					"ps_muls0 3, 3, %2\n\t"
					"ps_muls0 4, 4, %2\n\t"
					"ps_muls0 5, 5, %2\n\t"
					"psq_st  0,  0(%1), 0, 0\n\t"
					"psq_st  1,  8(%1), 0, 0\n\t"
					"psq_st  2, 16(%1), 0, 0\n\t"
					"psq_st  3, 24(%1), 0, 0\n\t"
					"psq_st  4, 32(%1), 0, 0\n\t"
					"psq_st  5, 40(%1), 0, 0\n\t"
					".machine pop\n\t"
					:
					: "b"(bone), "b"(skinMatrix), "f"(w)
					: "fr0", "fr1", "fr2", "fr3", "fr4", "fr5", "memory");
			} else {
				asm volatile(
					".machine push\n\t"
					".machine gekko\n\t"
					"psq_l   0,  0(%0), 0, 0\n\t"
					"psq_l   1,  8(%0), 0, 0\n\t"
					"psq_l   2, 16(%0), 0, 0\n\t"
					"psq_l   3, 24(%0), 0, 0\n\t"
					"psq_l   4, 32(%0), 0, 0\n\t"
					"psq_l   5, 40(%0), 0, 0\n\t"
					"psq_l   6,  0(%1), 0, 0\n\t"
					"psq_l   7,  8(%1), 0, 0\n\t"
					"psq_l   8, 16(%1), 0, 0\n\t"
					"psq_l   9, 24(%1), 0, 0\n\t"
					"psq_l  10, 32(%1), 0, 0\n\t"
					"psq_l  11, 40(%1), 0, 0\n\t"
					"ps_madds0 6, 0, %2, 6\n\t"
					"ps_madds0 7, 1, %2, 7\n\t"
					"ps_madds0 8, 2, %2, 8\n\t"
					"ps_madds0 9, 3, %2, 9\n\t"
					"ps_madds0 10, 4, %2, 10\n\t"
					"ps_madds0 11, 5, %2, 11\n\t"
					"psq_st  6,  0(%1), 0, 0\n\t"
					"psq_st  7,  8(%1), 0, 0\n\t"
					"psq_st  8, 16(%1), 0, 0\n\t"
					"psq_st  9, 24(%1), 0, 0\n\t"
					"psq_st 10, 32(%1), 0, 0\n\t"
					"psq_st 11, 40(%1), 0, 0\n\t"
					".machine pop\n\t"
					:
					: "b"(bone), "b"(skinMatrix), "f"(w)
					: "fr0", "fr1", "fr2", "fr3", "fr4", "fr5",
					  "fr6", "fr7", "fr8", "fr9", "fr10", "fr11", "memory");
			}
		}
	}
	if (first) {
		memset(skinMatrix, 0, sizeof(skinMatrix));
	}
#else
	memset(skinMatrix, 0, sizeof(skinMatrix));
	for (int j = 0; j < nweights; j++) {
		const float *bone = &gstate.boneMatrix[j * 12];
		if (weights[j] != 0.0f) {
			for (int i = 0; i < 12; i++) {
				skinMatrix[i] += weights[j] * bone[i];
			}
		}
	}
#endif
}

void VertexDecoder::Step_WeightsU8Skin() const {
	const u8 *wdata = (const u8*)(ptr_);
	float weights[8];
	for (int j = 0; j < nweights; j++)
		weights[j] = wdata[j] ? (wdata[j] * (1.0f / 128.0f)) : 0.0f;
	ComputeSkinMatrix(weights);
}

void VertexDecoder::Step_WeightsU16Skin() const {
	const u16_le *wdata = (const u16_le*)(ptr_);
	float weights[8];
	for (int j = 0; j < nweights; j++) {
		u16 w = wdata[j];
		weights[j] = w ? (w * (1.0f / 32768.0f)) : 0.0f;
	}
	ComputeSkinMatrix(weights);
}

void VertexDecoder::Step_WeightsFloatSkin() const {
	const float_le *wdata = (const float_le*)(ptr_);
	float weights[8];
	for (int j = 0; j < nweights; j++)
		weights[j] = wdata[j];
	ComputeSkinMatrix(weights);
}

void VertexDecoder::Step_TcU8ToFloat() const
{
	// u32 to write two bytes of zeroes for free.
	float *uv = (float *)(decoded_ + decFmt.uvoff);
	const u8 *uvdata = (const u8*)(ptr_ + tcoff);
	uv[0] = uvdata[0] * (1.0f / 128.0f);
	uv[1] = uvdata[1] * (1.0f / 128.0f);
}

void VertexDecoder::Step_TcU16ToFloat() const
{
	float *uv = (float *)(decoded_ + decFmt.uvoff);
	const u16_le *uvdata = (const u16_le*)(ptr_ + tcoff);
	uv[0] = uvdata[0] * (1.0f / 32768.0f);
	uv[1] = uvdata[1] * (1.0f / 32768.0f);
}

void VertexDecoder::Step_TcU16DoubleToFloat() const
{
	float *uv = (float*)(decoded_ + decFmt.uvoff);
	const u16_le *uvdata = (const u16_le*)(ptr_ + tcoff);
	uv[0] = uvdata[0] * (1.0f / 16384.0f);
	uv[1] = uvdata[1] * (1.0f / 16384.0f);
}

void VertexDecoder::Step_TcU16ThroughToFloat() const
{
	float *uv = (float *)(decoded_ + decFmt.uvoff);
	const u16_le *uvdata = (const u16_le*)(ptr_ + tcoff);
	uv[0] = uvdata[0];
	uv[1] = uvdata[1];

	gstate_c.vertBounds.minU = std::min(gstate_c.vertBounds.minU, (u16)uvdata[0]);
	gstate_c.vertBounds.maxU = std::max(gstate_c.vertBounds.maxU, (u16)uvdata[0]);
	gstate_c.vertBounds.minV = std::min(gstate_c.vertBounds.minV, (u16)uvdata[1]);
	gstate_c.vertBounds.maxV = std::max(gstate_c.vertBounds.maxV, (u16)uvdata[1]);
}

void VertexDecoder::Step_TcU16ThroughDoubleToFloat() const
{
	float *uv = (float *)(decoded_ + decFmt.uvoff);
	const u16_le *uvdata = (const u16_le*)(ptr_ + tcoff);
	uv[0] = uvdata[0] * 2;
	uv[1] = uvdata[1] * 2;
}

void VertexDecoder::Step_TcFloat() const
{
	float *uv = (float *)(decoded_ + decFmt.uvoff);
	const float_le *uvdata = (const float_le*)(ptr_ + tcoff);
	uv[0] = uvdata[0];
	uv[1] = uvdata[1];
}

void VertexDecoder::Step_TcFloatThrough() const
{
	float *uv = (float *)(decoded_ + decFmt.uvoff);
	const float_le *uvdata = (const float_le*)(ptr_ + tcoff);
	uv[0] = uvdata[0];
	uv[1] = uvdata[1];

	gstate_c.vertBounds.minU = std::min(gstate_c.vertBounds.minU, (u16)uvdata[0]);
	gstate_c.vertBounds.maxU = std::max(gstate_c.vertBounds.maxU, (u16)uvdata[0]);
	gstate_c.vertBounds.minV = std::min(gstate_c.vertBounds.minV, (u16)uvdata[1]);
	gstate_c.vertBounds.maxV = std::max(gstate_c.vertBounds.maxV, (u16)uvdata[1]);
}

void VertexDecoder::Step_TcU8Prescale() const {
	float *uv = (float *)(decoded_ + decFmt.uvoff);
	const u8 *uvdata = (const u8 *)(ptr_ + tcoff);
	uv[0] = (float)uvdata[0] * (1.f / 128.f) * gstate_c.uv.uScale + gstate_c.uv.uOff;
	uv[1] = (float)uvdata[1] * (1.f / 128.f) * gstate_c.uv.vScale + gstate_c.uv.vOff;
}

void VertexDecoder::Step_TcU16Prescale() const {
	float *uv = (float *)(decoded_ + decFmt.uvoff);
	const u16_le *uvdata = (const u16_le *)(ptr_ + tcoff);
	uv[0] = (float)uvdata[0] * (1.f / 32768.f) * gstate_c.uv.uScale + gstate_c.uv.uOff;
	uv[1] = (float)uvdata[1] * (1.f / 32768.f) * gstate_c.uv.vScale + gstate_c.uv.vOff;
}

void VertexDecoder::Step_TcU16DoublePrescale() const {
	float *uv = (float *)(decoded_ + decFmt.uvoff);
	const u16_le *uvdata = (const u16_le *)(ptr_ + tcoff);
	uv[0] = (float)uvdata[0] * (1.f / 16384.f) * gstate_c.uv.uScale + gstate_c.uv.uOff;
	uv[1] = (float)uvdata[1] * (1.f / 16384.f) * gstate_c.uv.vScale + gstate_c.uv.vOff;
}

void VertexDecoder::Step_TcFloatPrescale() const {
	float *uv = (float *)(decoded_ + decFmt.uvoff);
	const float_le *uvdata = (const float_le*)(ptr_ + tcoff);
	uv[0] = uvdata[0] * gstate_c.uv.uScale + gstate_c.uv.uOff;
	uv[1] = uvdata[1] * gstate_c.uv.vScale + gstate_c.uv.vOff;
}

void VertexDecoder::Step_TcU8MorphToFloat() const {
	float uv[2] = { 0, 0 };
	for (int n = 0; n < morphcount; n++) {
		float w = gstate_c.morphWeights[n];
		const u8 *uvdata = (const u8 *)(ptr_ + onesize_*n + tcoff);

		uv[0] += (float)uvdata[0] * (1.f / 128.f) * w;
		uv[1] += (float)uvdata[1] * (1.f / 128.f) * w;
	}

	float *out = (float *)(decoded_ + decFmt.uvoff);
	out[0] = uv[0];
	out[1] = uv[1];
}

void VertexDecoder::Step_TcU16MorphToFloat() const {
	float uv[2] = { 0, 0 };
	for (int n = 0; n < morphcount; n++) {
		float w = gstate_c.morphWeights[n];
		const u16_le *uvdata = (const u16_le *)(ptr_ + onesize_*n + tcoff);

		uv[0] += (float)uvdata[0] * (1.f / 32768.f) * w;
		uv[1] += (float)uvdata[1] * (1.f / 32768.f) * w;
	}

	float *out = (float *)(decoded_ + decFmt.uvoff);
	out[0] = uv[0];
	out[1] = uv[1];
}

void VertexDecoder::Step_TcU16DoubleMorphToFloat() const {
	float uv[2] = { 0, 0 };
	for (int n = 0; n < morphcount; n++) {
		float w = gstate_c.morphWeights[n];
		const u16_le *uvdata = (const u16_le *)(ptr_ + onesize_*n + tcoff);

		uv[0] += (float)uvdata[0] * (1.f / 16384.f) * w;
		uv[1] += (float)uvdata[1] * (1.f / 16384.f) * w;
	}

	float *out = (float *)(decoded_ + decFmt.uvoff);
	out[0] = uv[0];
	out[1] = uv[1];
}

void VertexDecoder::Step_TcFloatMorph() const {
	float uv[2] = { 0, 0 };
	for (int n = 0; n < morphcount; n++) {
		float w = gstate_c.morphWeights[n];
		const float_le *uvdata = (const float_le *)(ptr_ + onesize_*n + tcoff);

		uv[0] += (float)uvdata[0] * w;
		uv[1] += (float)uvdata[1] * w;
	}

	float *out = (float *)(decoded_ + decFmt.uvoff);
	out[0] = uv[0];
	out[1] = uv[1];
}

void VertexDecoder::Step_TcU8PrescaleMorph() const {
	float uv[2] = { 0, 0 };
	for (int n = 0; n < morphcount; n++) {
		float w = gstate_c.morphWeights[n];
		const u8 *uvdata = (const u8 *)(ptr_ + onesize_*n + tcoff);

		uv[0] += (float)uvdata[0] * (1.f / 128.f) * w;
		uv[1] += (float)uvdata[1] * (1.f / 128.f) * w;
	}

	float *out = (float *)(decoded_ + decFmt.uvoff);
	out[0] = uv[0] * gstate_c.uv.uScale + gstate_c.uv.uOff;
	out[1] = uv[1] * gstate_c.uv.vScale + gstate_c.uv.vOff;
}

void VertexDecoder::Step_TcU16PrescaleMorph() const {
	float uv[2] = { 0, 0 };
	for (int n = 0; n < morphcount; n++) {
		float w = gstate_c.morphWeights[n];
		const u16_le *uvdata = (const u16_le *)(ptr_ + onesize_*n + tcoff);

		uv[0] += (float)uvdata[0] * (1.f / 32768.f) * w;
		uv[1] += (float)uvdata[1] * (1.f / 32768.f) * w;
	}

	float *out = (float *)(decoded_ + decFmt.uvoff);
	out[0] = uv[0] * gstate_c.uv.uScale + gstate_c.uv.uOff;
	out[1] = uv[1] * gstate_c.uv.vScale + gstate_c.uv.vOff;
}

void VertexDecoder::Step_TcU16DoublePrescaleMorph() const {
	float uv[2] = { 0, 0 };
	for (int n = 0; n < morphcount; n++) {
		float w = gstate_c.morphWeights[n];
		const u16_le *uvdata = (const u16_le *)(ptr_ + onesize_*n + tcoff);

		uv[0] += (float)uvdata[0] * (1.f / 16384.f) * w;
		uv[1] += (float)uvdata[1] * (1.f / 16384.f) * w;
	}

	float *out = (float *)(decoded_ + decFmt.uvoff);
	out[0] = uv[0] * gstate_c.uv.uScale + gstate_c.uv.uOff;
	out[1] = uv[1] * gstate_c.uv.vScale + gstate_c.uv.vOff;
}

void VertexDecoder::Step_TcFloatPrescaleMorph() const {
	float uv[2] = { 0, 0 };
	for (int n = 0; n < morphcount; n++) {
		float w = gstate_c.morphWeights[n];
		const float_le *uvdata = (const float_le *)(ptr_ + onesize_*n + tcoff);

		uv[0] += (float)uvdata[0] * w;
		uv[1] += (float)uvdata[1] * w;
	}

	float *out = (float *)(decoded_ + decFmt.uvoff);
	out[0] = uv[0] * gstate_c.uv.uScale + gstate_c.uv.uOff;
	out[1] = uv[1] * gstate_c.uv.vScale + gstate_c.uv.vOff;
}

void VertexDecoder::Step_ColorInvalid() const
{
	// Do nothing.  This is only here to prevent crashes.
}

void VertexDecoder::Step_Color565() const
{
	u32_le *c = (u32_le *)(decoded_ + decFmt.c0off);
	u16 cdata = *(u16_le *)(ptr_ + coloff);
	u32 col = Convert5To8(cdata & 0x1f);
	col |= Convert6To8((cdata >> 5) & 0x3f) << 8;
	col |= Convert5To8((cdata >> 11) & 0x1f) << 16;
	col |= 255u << 24;
	*c = col;
	// Always full alpha.
}

void VertexDecoder::Step_Color5551() const
{
	u32_le *c = (u32_le *)(decoded_ + decFmt.c0off);
	u16 cdata = *(u16_le *)(ptr_ + coloff);
	gstate_c.vertexFullAlpha = gstate_c.vertexFullAlpha && (cdata >> 15) != 0;
	u32 col = Convert5To8(cdata & 0x1f);
	col |= Convert5To8((cdata >> 5) & 0x1f) << 8;
	col |= Convert5To8((cdata >> 10) & 0x1f) << 16;
	col |= (cdata >> 15) ? (255u << 24) : 0;
	*c = col;
}

void VertexDecoder::Step_Color4444() const
{
	u32_le *c = (u32_le *)(decoded_ + decFmt.c0off);
	u16 cdata = *(u16_le *)(ptr_ + coloff);
	gstate_c.vertexFullAlpha = gstate_c.vertexFullAlpha && (cdata >> 12) == 0xF;
	u32 col = 0;
	for (int j = 0; j < 4; j++)
		col |= Convert4To8((cdata >> (j * 4)) & 0xF) << (j * 8);
	*c = col;
}

void VertexDecoder::Step_Color8888() const
{
	u32_le *c = (u32_le *)(decoded_ + decFmt.c0off);
	u32 cdata = *(u32_le*)(ptr_ + coloff);
	gstate_c.vertexFullAlpha = gstate_c.vertexFullAlpha && (cdata >> 24) == 0xFF;
	*c = cdata;
}

void VertexDecoder::Step_Color565Morph() const
{
	float col[3] = { 0 };
	for (int n = 0; n < morphcount; n++) {
		float w = gstate_c.morphWeights[n];
		u16 cdata = *(u16_le *)(ptr_ + onesize_*n + coloff);
		col[0] += w * (cdata & 0x1f) * (255.0f / 31.0f);
		col[1] += w * ((cdata >> 5) & 0x3f) * (255.0f / 63.0f);
		col[2] += w * ((cdata >> 11) & 0x1f) * (255.0f / 31.0f);
	}
	u8 *c = decoded_ + decFmt.c0off;
	for (int i = 0; i < 3; i++) {
		c[i] = clamp_u8((int)col[i]);
	}
	c[3] = 255;
	// Always full alpha.
}

void VertexDecoder::Step_Color5551Morph() const
{
	float col[4] = { 0 };
	for (int n = 0; n < morphcount; n++) {
		float w = gstate_c.morphWeights[n];
		u16 cdata = *(u16_le *)(ptr_ + onesize_*n + coloff);
		col[0] += w * (cdata & 0x1f) * (255.0f / 31.0f);
		col[1] += w * ((cdata >> 5) & 0x1f) * (255.0f / 31.0f);
		col[2] += w * ((cdata >> 10) & 0x1f) * (255.0f / 31.0f);
		col[3] += w * ((cdata >> 15) ? 255.0f : 0.0f);
	}
	u8 *c = decoded_ + decFmt.c0off;
	for (int i = 0; i < 4; i++) {
		c[i] = clamp_u8((int)col[i]);
	}
	gstate_c.vertexFullAlpha = gstate_c.vertexFullAlpha && (int)col[3] >= 255;
}

void VertexDecoder::Step_Color4444Morph() const
{
	float col[4] = { 0 };
	for (int n = 0; n < morphcount; n++) {
		float w = gstate_c.morphWeights[n];
		u16 cdata = *(u16_le *)(ptr_ + onesize_*n + coloff);
		for (int j = 0; j < 4; j++)
			col[j] += w * ((cdata >> (j * 4)) & 0xF) * (255.0f / 15.0f);
	}
	u8 *c = decoded_ + decFmt.c0off;
	for (int i = 0; i < 4; i++) {
		c[i] = clamp_u8((int)col[i]);
	}
	gstate_c.vertexFullAlpha = gstate_c.vertexFullAlpha && (int)col[3] >= 255;
}

void VertexDecoder::Step_Color8888Morph() const
{
	float col[4] = { 0 };
	for (int n = 0; n < morphcount; n++) {
		float w = gstate_c.morphWeights[n];
		const u8 *cdata = (const u8*)(ptr_ + onesize_*n + coloff);
		for (int j = 0; j < 4; j++)
			col[j] += w * cdata[j];
	}
	u8 *c = decoded_ + decFmt.c0off;
	for (int i = 0; i < 4; i++) {
		c[i] = clamp_u8((int)col[i]);
	}
	gstate_c.vertexFullAlpha = gstate_c.vertexFullAlpha && (int)col[3] >= 255;
}

void VertexDecoder::Step_NormalS8() const
{
	s8 *normal = (s8 *)(decoded_ + decFmt.nrmoff);
	const s8 *sv = (const s8*)(ptr_ + nrmoff);
	for (int j = 0; j < 3; j++)
		normal[j] = sv[j];
	normal[3] = 0;
}

void VertexDecoder::Step_NormalS8ToFloat() const
{
	float *normal = (float *)(decoded_ + decFmt.nrmoff);
	const s8 *sv = (const s8*)(ptr_ + nrmoff);
	normal[0] = sv[0] * (1.0f / 128.0f);
	normal[1] = sv[1] * (1.0f / 128.0f);
	normal[2] = sv[2] * (1.0f / 128.0f);
}

void VertexDecoder::Step_NormalS16() const
{
	s16 *normal = (s16 *)(decoded_ + decFmt.nrmoff);
	const s16_le *sv = (const s16_le*)(ptr_ + nrmoff);
	for (int j = 0; j < 3; j++)
		normal[j] = sv[j];
	normal[3] = 0;
}

void VertexDecoder::Step_NormalFloat() const
{
	u32 *normal = (u32 *)(decoded_ + decFmt.nrmoff);
	const u32_le *fv = (const u32_le*)(ptr_ + nrmoff);
	for (int j = 0; j < 3; j++)
		normal[j] = fv[j];
}

void VertexDecoder::Step_NormalS8Skin() const
{
	float *normal = (float *)(decoded_ + decFmt.nrmoff);
	const s8 *sv = (const s8*)(ptr_ + nrmoff);
	const float fn[3] = { sv[0] * (1.0f / 128.0f), sv[1] * (1.0f / 128.0f), sv[2] * (1.0f / 128.0f) };
	Norm3ByMatrix43(normal, fn, skinMatrix);
}

void VertexDecoder::Step_NormalS16Skin() const
{
	float *normal = (float *)(decoded_ + decFmt.nrmoff);
	const s16_le *sv = (const s16_le*)(ptr_ + nrmoff);
	const float fn[3] = { (float)sv[0] * (1.0f / 32768.0f), (float)sv[1] * (1.0f / 32768.0f), (float)sv[2] * (1.0f / 32768.0f) };
	Norm3ByMatrix43(normal, fn, skinMatrix);
}

void VertexDecoder::Step_NormalFloatSkin() const
{
	float *normal = (float *)(decoded_ + decFmt.nrmoff);
	const float_le *sv = (const float_le*)(ptr_ + nrmoff);
	const float fn[3] = { sv[0], sv[1], sv[2] };
	Norm3ByMatrix43(normal, fn, skinMatrix);
}

void VertexDecoder::Step_NormalS8Morph() const
{
	float *normal = (float *)(decoded_ + decFmt.nrmoff);
	memset(normal, 0, sizeof(float) * 3);
	for (int n = 0; n < morphcount; n++) {
		const s8 *bv = (const s8*)(ptr_ + onesize_*n + nrmoff);
		const float multiplier = gstate_c.morphWeights[n] * (1.0f / 128.0f);
		for (int j = 0; j < 3; j++)
			normal[j] += bv[j] * multiplier;
	}
}

void VertexDecoder::Step_NormalS16Morph() const
{
	float *normal = (float *)(decoded_ + decFmt.nrmoff);
	memset(normal, 0, sizeof(float) * 3);
	for (int n = 0; n < morphcount; n++) {
		const s16_le *sv = (const s16_le *)(ptr_ + onesize_*n + nrmoff);
		const float multiplier = gstate_c.morphWeights[n] * (1.0f / 32768.0f);
		for (int j = 0; j < 3; j++)
			normal[j] += sv[j] * multiplier;
	}
}

void VertexDecoder::Step_NormalFloatMorph() const
{
	float *normal = (float *)(decoded_ + decFmt.nrmoff);
	memset(normal, 0, sizeof(float) * 3);
	for (int n = 0; n < morphcount; n++) {
		float multiplier = gstate_c.morphWeights[n];
		const float_le *fv = (const float_le*)(ptr_ + onesize_*n + nrmoff);
		for (int j = 0; j < 3; j++)
			normal[j] += fv[j] * multiplier;
	}
}

void VertexDecoder::Step_NormalS8MorphSkin() const {
	float *normal = (float *)(decoded_ + decFmt.nrmoff);
	float nrm[3]{};
	for (int n = 0; n < morphcount; n++) {
		const s8 *bv = (const s8*)(ptr_ + onesize_ * n + nrmoff);
		const float multiplier = gstate_c.morphWeights[n] * (1.0f / 128.0f);
		for (int j = 0; j < 3; j++)
			nrm[j] += bv[j] * multiplier;
	}
	Norm3ByMatrix43(normal, nrm, skinMatrix);
}

void VertexDecoder::Step_NormalS16MorphSkin() const {
	float *normal = (float *)(decoded_ + decFmt.nrmoff);
	float nrm[3]{};
	for (int n = 0; n < morphcount; n++) {
		const s16_le *sv = (const s16_le *)(ptr_ + onesize_ * n + nrmoff);
		const float multiplier = gstate_c.morphWeights[n] * (1.0f / 32768.0f);
		for (int j = 0; j < 3; j++)
			nrm[j] += sv[j] * multiplier;
	}
	Norm3ByMatrix43(normal, nrm, skinMatrix);
}

void VertexDecoder::Step_NormalFloatMorphSkin() const {
	float *normal = (float *)(decoded_ + decFmt.nrmoff);
	float nrm[3]{};
	for (int n = 0; n < morphcount; n++) {
		float multiplier = gstate_c.morphWeights[n];
		const float_le *fv = (const float_le*)(ptr_ + onesize_ * n + nrmoff);
		for (int j = 0; j < 3; j++)
			nrm[j] += fv[j] * multiplier;
	}
	Norm3ByMatrix43(normal, nrm, skinMatrix);
}

void VertexDecoder::Step_PosS8() const
{
	float *pos = (float *)(decoded_ + decFmt.posoff);
	const s8 *sv = (const s8*)(ptr_ + posoff);
	for (int j = 0; j < 3; j++)
		pos[j] = sv[j] * (1.0f / 128.0f);
}

void VertexDecoder::Step_PosS16() const
{
	float *pos = (float *)(decoded_ + decFmt.posoff);
	const s16_le *sv = (const s16_le *)(ptr_ + posoff);
	for (int j = 0; j < 3; j++)
		pos[j] = sv[j] * (1.0f / 32768.0f);
}

void VertexDecoder::Step_PosFloat() const
{
	float *pos = (float *)(decoded_ + decFmt.posoff);
	const float_le *fv = (const float_le *)(ptr_ + posoff);
	for (int j = 0; j < 3; j++)
		pos[j] = fv[j];
}

void VertexDecoder::Step_PosS8Skin() const
{
	float *pos = (float *)(decoded_ + decFmt.posoff);
	const s8 *sv = (const s8*)(ptr_ + posoff);
	const float fn[3] = { sv[0] * (1.0f / 128.0f), sv[1] * (1.0f / 128.0f), sv[2] * (1.0f / 128.0f) };
	Vec3ByMatrix43(pos, fn, skinMatrix);
}

void VertexDecoder::Step_PosS16Skin() const
{
	float *pos = (float *)(decoded_ + decFmt.posoff);
	const s16_le *sv = (const s16_le *)(ptr_ + posoff);
	const float fn[3] = { (float)sv[0] * (1.0f / 32768.0f), (float)sv[1] * (1.0f / 32768.0f), (float)sv[2] * (1.0f / 32768.0f) };
	Vec3ByMatrix43(pos, fn, skinMatrix);
}

void VertexDecoder::Step_PosFloatSkin() const
{
	float *pos = (float *)(decoded_ + decFmt.posoff);
	const float_le *sv = (const float_le*)(ptr_ + posoff);
	const float fn[3] = { sv[0], sv[1], sv[2] };
	Vec3ByMatrix43(pos, fn, skinMatrix);
}

void VertexDecoder::Step_PosS8Through() const
{
	float *v = (float *)(decoded_ + decFmt.posoff);
	const s8 *sv = (const s8*)(ptr_ + posoff);
	v[0] = sv[0];
	v[1] = sv[1];
	v[2] = sv[2];
}

void VertexDecoder::Step_PosS16Through() const
{
	float *v = (float *)(decoded_ + decFmt.posoff);
	const s16_le *sv = (const s16_le *)(ptr_ + posoff);
	const u16_le *uv = (const u16_le *)(ptr_ + posoff);
	v[0] = sv[0];
	v[1] = sv[1];
	v[2] = uv[2];
}

void VertexDecoder::Step_PosFloatThrough() const
{
	float *v = (float *)(decoded_ + decFmt.posoff);
	const float_le *fv = (const float_le*)(ptr_ + posoff);
	v[0] = fv[0];
	v[1] = fv[1];
	v[2] = fv[2];
}

void VertexDecoder::Step_PosS8Morph() const
{
	float *v = (float *)(decoded_ + decFmt.posoff);
	memset(v, 0, sizeof(float) * 3);
	for (int n = 0; n < morphcount; n++) {
		const float multiplier = 1.0f / 128.0f;
		const s8 *sv = (const s8*)(ptr_ + onesize_*n + posoff);
		for (int j = 0; j < 3; j++)
			v[j] += (float)sv[j] * (multiplier * gstate_c.morphWeights[n]);
	}
}

void VertexDecoder::Step_PosS16Morph() const
{
	float *v = (float *)(decoded_ + decFmt.posoff);
	memset(v, 0, sizeof(float) * 3);
	for (int n = 0; n < morphcount; n++) {
		const float multiplier = 1.0f / 32768.0f;
		const s16_le *sv = (const s16_le*)(ptr_ + onesize_*n + posoff);
		for (int j = 0; j < 3; j++)
			v[j] += (float)sv[j] * (multiplier * gstate_c.morphWeights[n]);
	}
}

void VertexDecoder::Step_PosFloatMorph() const
{
	float *v = (float *)(decoded_ + decFmt.posoff);
	memset(v, 0, sizeof(float) * 3);
	for (int n = 0; n < morphcount; n++) {
		const float_le *fv = (const float_le*)(ptr_ + onesize_*n + posoff);
		for (int j = 0; j < 3; j++)
			v[j] += fv[j] * gstate_c.morphWeights[n];
	}
}

void VertexDecoder::Step_PosS8MorphSkin() const {
	float *v = (float *)(decoded_ + decFmt.posoff);
	float pos[3]{};
	for (int n = 0; n < morphcount; n++) {
		const float multiplier = 1.0f / 128.0f;
		const s8 *sv = (const s8*)(ptr_ + onesize_ * n + posoff);
		for (int j = 0; j < 3; j++)
			pos[j] += (float)sv[j] * (multiplier * gstate_c.morphWeights[n]);
	}
	Vec3ByMatrix43(v, pos, skinMatrix);
}

void VertexDecoder::Step_PosS16MorphSkin() const {
	float *v = (float *)(decoded_ + decFmt.posoff);
	float pos[3]{};
	for (int n = 0; n < morphcount; n++) {
		const float multiplier = 1.0f / 32768.0f;
		const s16_le *sv = (const s16_le*)(ptr_ + onesize_ * n + posoff);
		for (int j = 0; j < 3; j++)
			pos[j] += (float)sv[j] * (multiplier * gstate_c.morphWeights[n]);
	}
	Vec3ByMatrix43(v, pos, skinMatrix);
}

void VertexDecoder::Step_PosFloatMorphSkin() const {
	float *v = (float *)(decoded_ + decFmt.posoff);
	float pos[3]{};
	for (int n = 0; n < morphcount; n++) {
		const float_le *fv = (const float_le*)(ptr_ + onesize_ * n + posoff);
		for (int j = 0; j < 3; j++)
			pos[j] += fv[j] * gstate_c.morphWeights[n];
	}
	Vec3ByMatrix43(v, pos, skinMatrix);
}

static const StepFunction wtstep[4] = {
	0,
	&VertexDecoder::Step_WeightsU8,
	&VertexDecoder::Step_WeightsU16,
	&VertexDecoder::Step_WeightsFloat,
};

static const StepFunction wtstepToFloat[4] = {
	0,
	&VertexDecoder::Step_WeightsU8ToFloat,
	&VertexDecoder::Step_WeightsU16ToFloat,
	&VertexDecoder::Step_WeightsFloat,
};

// TODO: Morph weights correctly! This is missing. Not sure if any game actually
// use this functionality at all.

static const StepFunction wtstep_skin[4] = {
	0,
	&VertexDecoder::Step_WeightsU8Skin,
	&VertexDecoder::Step_WeightsU16Skin,
	&VertexDecoder::Step_WeightsFloatSkin,
};

static const StepFunction tcstepToFloat[4] = {
	0,
	&VertexDecoder::Step_TcU8ToFloat,
	&VertexDecoder::Step_TcU16ToFloat,
	&VertexDecoder::Step_TcFloat,
};

static const StepFunction tcstep_prescale[4] = {
	0,
	&VertexDecoder::Step_TcU8Prescale,
	&VertexDecoder::Step_TcU16Prescale,
	&VertexDecoder::Step_TcFloatPrescale,
};

static const StepFunction tcstep_prescale_remaster[4] = {
	0,
	&VertexDecoder::Step_TcU8Prescale,
	&VertexDecoder::Step_TcU16DoublePrescale,
	&VertexDecoder::Step_TcFloatPrescale,
};

static const StepFunction tcstep_prescale_morph[4] = {
	0,
	&VertexDecoder::Step_TcU8PrescaleMorph,
	&VertexDecoder::Step_TcU16PrescaleMorph,
	&VertexDecoder::Step_TcFloatPrescaleMorph,
};

static const StepFunction tcstep_prescale_morph_remaster[4] = {
	0,
	&VertexDecoder::Step_TcU8PrescaleMorph,
	&VertexDecoder::Step_TcU16DoublePrescaleMorph,
	&VertexDecoder::Step_TcFloatPrescaleMorph,
};

static const StepFunction tcstep_morphToFloat[4] = {
	0,
	&VertexDecoder::Step_TcU8MorphToFloat,
	&VertexDecoder::Step_TcU16MorphToFloat,
	&VertexDecoder::Step_TcFloatMorph,
};

static const StepFunction tcstep_morph_remasterToFloat[4] = {
	0,
	&VertexDecoder::Step_TcU8MorphToFloat,
	&VertexDecoder::Step_TcU16DoubleMorphToFloat,
	&VertexDecoder::Step_TcFloatMorph,
};

static const StepFunction tcstep_throughToFloat[4] = {
	0,
	&VertexDecoder::Step_TcU8ToFloat,
	&VertexDecoder::Step_TcU16ThroughToFloat,
	&VertexDecoder::Step_TcFloatThrough,
};

static const StepFunction tcstep_remasterToFloat[4] = {
	0,
	&VertexDecoder::Step_TcU8ToFloat,
	&VertexDecoder::Step_TcU16DoubleToFloat,
	&VertexDecoder::Step_TcFloat,
};

static const StepFunction tcstep_through_remasterToFloat[4] = {
	0,
	&VertexDecoder::Step_TcU8ToFloat,
	&VertexDecoder::Step_TcU16ThroughDoubleToFloat,
	&VertexDecoder::Step_TcFloatThrough,
};

static const StepFunction colstep[8] = {
	0,
	&VertexDecoder::Step_ColorInvalid,
	&VertexDecoder::Step_ColorInvalid,
	&VertexDecoder::Step_ColorInvalid,
	&VertexDecoder::Step_Color565,
	&VertexDecoder::Step_Color5551,
	&VertexDecoder::Step_Color4444,
	&VertexDecoder::Step_Color8888,
};

static const StepFunction colstep_morph[8] = {
	0,
	&VertexDecoder::Step_ColorInvalid,
	&VertexDecoder::Step_ColorInvalid,
	&VertexDecoder::Step_ColorInvalid,
	&VertexDecoder::Step_Color565Morph,
	&VertexDecoder::Step_Color5551Morph,
	&VertexDecoder::Step_Color4444Morph,
	&VertexDecoder::Step_Color8888Morph,
};

static const StepFunction nrmstep[4] = {
	0,
	&VertexDecoder::Step_NormalS8,
	&VertexDecoder::Step_NormalS16,
	&VertexDecoder::Step_NormalFloat,
};

static const StepFunction nrmstep8BitToFloat[4] = {
	0,
	&VertexDecoder::Step_NormalS8ToFloat,
	&VertexDecoder::Step_NormalS16,
	&VertexDecoder::Step_NormalFloat,
};

static const StepFunction nrmstep_skin[4] = {
	0,
	&VertexDecoder::Step_NormalS8Skin,
	&VertexDecoder::Step_NormalS16Skin,
	&VertexDecoder::Step_NormalFloatSkin,
};

static const StepFunction nrmstep_morph[4] = {
	0,
	&VertexDecoder::Step_NormalS8Morph,
	&VertexDecoder::Step_NormalS16Morph,
	&VertexDecoder::Step_NormalFloatMorph,
};

static const StepFunction nrmstep_morphskin[4] = {
	0,
	&VertexDecoder::Step_NormalS8MorphSkin,
	&VertexDecoder::Step_NormalS16MorphSkin,
	&VertexDecoder::Step_NormalFloatMorphSkin,
};

static const StepFunction posstep[4] = {
	&VertexDecoder::Step_PosS8,
	&VertexDecoder::Step_PosS8,
	&VertexDecoder::Step_PosS16,
	&VertexDecoder::Step_PosFloat,
};

static const StepFunction posstep_skin[4] = {
	&VertexDecoder::Step_PosS8Skin,
	&VertexDecoder::Step_PosS8Skin,
	&VertexDecoder::Step_PosS16Skin,
	&VertexDecoder::Step_PosFloatSkin,
};

static const StepFunction posstep_morph[4] = {
	&VertexDecoder::Step_PosS8Morph,
	&VertexDecoder::Step_PosS8Morph,
	&VertexDecoder::Step_PosS16Morph,
	&VertexDecoder::Step_PosFloatMorph,
};

static const StepFunction posstep_morph_skin[4] = {
	&VertexDecoder::Step_PosS8MorphSkin,
	&VertexDecoder::Step_PosS8MorphSkin,
	&VertexDecoder::Step_PosS16MorphSkin,
	&VertexDecoder::Step_PosFloatMorphSkin,
};

static const StepFunction posstep_through[4] = {
	&VertexDecoder::Step_PosS8Through,
	&VertexDecoder::Step_PosS8Through,
	&VertexDecoder::Step_PosS16Through,
	&VertexDecoder::Step_PosFloatThrough,
};

void VertexDecoder::SetVertexType(u32 fmt, const VertexDecoderOptions &options, VertexDecoderJitCache *jitCache) {
	fmt_ = fmt;
	throughmode = (fmt & GE_VTYPE_THROUGH) != 0;
	numSteps_ = 0;

	biggest = 0;
	size = 0;

	tc = fmt & 0x3;
	col = (fmt >> 2) & 0x7;
	nrm = (fmt >> 5) & 0x3;
	pos = (fmt >> 7) & 0x3;
	weighttype = (fmt >> 9) & 0x3;
	idx = (fmt >> 11) & 0x3;
	morphcount = ((fmt >> 18) & 0x7) + 1;
	nweights = ((fmt >> 14) & 0x7) + 1;

	int decOff = 0;
	memset(&decFmt, 0, sizeof(decFmt));

	if (morphcount > 1) {
		DEBUG_LOG_REPORT_ONCE(vtypeM, G3D, "VTYPE with morph used: THRU=%i TC=%i COL=%i POS=%i NRM=%i WT=%i NW=%i IDX=%i MC=%i", (int)throughmode, tc, col, pos, nrm, weighttype, nweights, idx, morphcount);
	} else {
		DEBUG_LOG(G3D, "VTYPE: THRU=%i TC=%i COL=%i POS=%i NRM=%i WT=%i NW=%i IDX=%i MC=%i", (int)throughmode, tc, col, pos, nrm, weighttype, nweights, idx, morphcount);
	}

	bool skinInDecode = weighttype != 0 && g_Config.bSoftwareSkinning;

	if (weighttype) { // && nweights?
		weightoff = size;
		//size = align(size, wtalign[weighttype]);	unnecessary
		size += wtsize[weighttype] * nweights;
		if (wtalign[weighttype] > biggest)
			biggest = wtalign[weighttype];

		if (skinInDecode) {
			// No visible output, computes a matrix that is passed through the skinMatrix variable
			// to the "nrm" and "pos" steps.
			// Technically we should support morphing the weights too, but I have a hard time
			// imagining that any game would use that.. but you never know.
			steps_[numSteps_++] = wtstep_skin[weighttype];
		} else {
			int fmtBase = DEC_FLOAT_1;
			if (options.expandAllWeightsToFloat) {
				steps_[numSteps_++] = wtstepToFloat[weighttype];
				fmtBase = DEC_FLOAT_1;
			} else {
				steps_[numSteps_++] = wtstep[weighttype];
				if (weighttype == GE_VTYPE_WEIGHT_8BIT >> GE_VTYPE_WEIGHT_SHIFT) {
					fmtBase = DEC_U8_1;
				} else if (weighttype == GE_VTYPE_WEIGHT_16BIT >> GE_VTYPE_WEIGHT_SHIFT) {
					fmtBase = DEC_U16_1;
				} else if (weighttype == GE_VTYPE_WEIGHT_FLOAT >> GE_VTYPE_WEIGHT_SHIFT) {
					fmtBase = DEC_FLOAT_1;
				}
			}

			int numWeights = TranslateNumBones(nweights);

			if (numWeights <= 4) {
				decFmt.w0off = decOff;
				decFmt.w0fmt = fmtBase + numWeights - 1;
				decOff += DecFmtSize(decFmt.w0fmt);
			} else {
				decFmt.w0off = decOff;
				decFmt.w0fmt = fmtBase + 3;
				decOff += DecFmtSize(decFmt.w0fmt);
				decFmt.w1off = decOff;
				decFmt.w1fmt = fmtBase + numWeights - 5;
				decOff += DecFmtSize(decFmt.w1fmt);
			}
		}
	}

	if (tc) {
		size = align(size, tcalign[tc]);
		tcoff = size;
		size += tcsize[tc];
		if (tcalign[tc] > biggest)
			biggest = tcalign[tc];

		// NOTE: That we check getUVGenMode here means that we must include it in the decoder ID!
		// throughmode is automatically included though, because it's part of the vertType.
		if (!throughmode && (gstate.getUVGenMode() == GE_TEXMAP_TEXTURE_COORDS || gstate.getUVGenMode() == GE_TEXMAP_UNKNOWN)) {
			if (g_DoubleTextureCoordinates)
				steps_[numSteps_++] = morphcount == 1 ? tcstep_prescale_remaster[tc] : tcstep_prescale_morph_remaster[tc];
			else
				steps_[numSteps_++] = morphcount == 1 ? tcstep_prescale[tc] : tcstep_prescale_morph[tc];
			decFmt.uvfmt = DEC_FLOAT_2;
		} else {
			// We now always expand UV to float.
			if (morphcount != 1 && !throughmode)
				steps_[numSteps_++] = g_DoubleTextureCoordinates ? tcstep_morph_remasterToFloat[tc] : tcstep_morphToFloat[tc];
			else if (g_DoubleTextureCoordinates)
				steps_[numSteps_++] = throughmode ? tcstep_through_remasterToFloat[tc] : tcstep_remasterToFloat[tc];
			else
				steps_[numSteps_++] = throughmode ? tcstep_throughToFloat[tc] : tcstepToFloat[tc];
			decFmt.uvfmt = DEC_FLOAT_2;
		}

		decFmt.uvoff = decOff;
		decOff += DecFmtSize(decFmt.uvfmt);
	}

	if (col) {
		size = align(size, colalign[col]);
		coloff = size;
		size += colsize[col];
		if (colalign[col] > biggest)
			biggest = colalign[col];

		steps_[numSteps_++] = morphcount == 1 ? colstep[col] : colstep_morph[col];

		// All color formats decode to DEC_U8_4 currently.
		// They can become floats later during transform though.
		decFmt.c0fmt = DEC_U8_4;
		decFmt.c0off = decOff;
		decOff += DecFmtSize(decFmt.c0fmt);
	} else {
		coloff = 0;
	}

	if (nrm) {
		size = align(size, nrmalign[nrm]);
		nrmoff = size;
		size += nrmsize[nrm];
		if (nrmalign[nrm] > biggest)
			biggest = nrmalign[nrm];

		if (skinInDecode) {
			steps_[numSteps_++] = morphcount == 1 ? nrmstep_skin[nrm] : nrmstep_morphskin[nrm];
			// After skinning, we always have three floats.
			decFmt.nrmfmt = DEC_FLOAT_3;
		} else {
			if (morphcount == 1) {
				// The 8-bit and 16-bit normal formats match GL formats nicely, and the 16-bit normal format matches a D3D format so let's use them where possible.
				switch (nrm) {
				case GE_VTYPE_NRM_8BIT >> GE_VTYPE_NRM_SHIFT:
					if (options.expand8BitNormalsToFloat) {
						decFmt.nrmfmt = DEC_FLOAT_3;
						steps_[numSteps_++] = nrmstep8BitToFloat[nrm];
					} else {
						decFmt.nrmfmt = DEC_S8_3;
						steps_[numSteps_++] = nrmstep[nrm];
					}
					break;
				case GE_VTYPE_NRM_16BIT >> GE_VTYPE_NRM_SHIFT:
					decFmt.nrmfmt = DEC_S16_3;
					steps_[numSteps_++] = nrmstep[nrm];
					break;
				case GE_VTYPE_NRM_FLOAT >> GE_VTYPE_NRM_SHIFT:
					decFmt.nrmfmt = DEC_FLOAT_3;
					steps_[numSteps_++] = nrmstep[nrm];
					break;
				}
			} else {
				decFmt.nrmfmt = DEC_FLOAT_3;
				steps_[numSteps_++] = nrmstep_morph[nrm];
			}
		}
		decFmt.nrmoff = decOff;
		decOff += DecFmtSize(decFmt.nrmfmt);
	}

	bool reportNoPos = false;
	if (!pos) {
		reportNoPos = true;
		pos = 1;
	}
	if (pos) { // there's always a position
		size = align(size, posalign[pos]);
		posoff = size;
		size += possize[pos];
		if (posalign[pos] > biggest)
			biggest = posalign[pos];

		if (throughmode) {
			steps_[numSteps_++] = posstep_through[pos];
			decFmt.posfmt = DEC_FLOAT_3;
		} else {
			if (skinInDecode) {
				steps_[numSteps_++] = morphcount == 1 ? posstep_skin[pos] : posstep_morph_skin[pos];
				decFmt.posfmt = DEC_FLOAT_3;
			} else {
				steps_[numSteps_++] = morphcount == 1 ? posstep[pos] : posstep_morph[pos];
				decFmt.posfmt = DEC_FLOAT_3;
			}
		}
		decFmt.posoff = decOff;
		decOff += DecFmtSize(decFmt.posfmt);
	}

	decFmt.stride = decOff;

	decFmt.ComputeID();

	size = align(size, biggest);
	onesize_ = size;
	size *= morphcount;
	DEBUG_LOG(G3D, "SVT : size = %i, aligned to biggest %i", size, biggest);

	if (reportNoPos) {
		char temp[256]{};
		ToString(temp);
		ERROR_LOG_REPORT(G3D, "Vertices without position found: (%08x) %s", fmt_, temp);
	}

	// Attempt to JIT as well. But only do that if the main CPU JIT is enabled, in order to aid
	// debugging attempts - if the main JIT doesn't work, this one won't do any better, probably.
	if (jitCache && g_Config.bVertexDecoderJit && g_Config.iCpuCore == (int)CPUCore::JIT) {
		jitted_ = jitCache->Compile(*this, &jittedSize_);
		if (!jitted_) {
			WARN_LOG(G3D, "Vertex decoder JIT failed! fmt = %08x (%s)", fmt_, GetString(SHADER_STRING_SHORT_DESC).c_str());
		}
	}
}

void VertexDecoder::DecodeVerts(u8 *decodedptr, const void *verts, int indexLowerBound, int indexUpperBound) const {
	// Decode the vertices within the found bounds, once each
	// decoded_ and ptr_ are used in the steps, so can't be turned into locals for speed.
	decoded_ = decodedptr;
	ptr_ = (const u8*)verts + indexLowerBound * size;

	int count = indexUpperBound - indexLowerBound + 1;
	int stride = decFmt.stride;

	// Check alignment before running the decoder, as we may crash if it's bad (as should the real PSP but doesn't always)
	if (((uintptr_t)verts & (biggest - 1)) != 0) {
		// Bad alignment. Not really sure what to do here... zero the verts to be safe?
		memset(decodedptr, 0, count * stride);
		return;
	}

	if (jitted_) {
		// We've compiled the steps into optimized machine code, so just jump!
		jitted_(ptr_, decoded_, count);
	} else if (!throughmode && morphcount == 1 && !g_DoubleTextureCoordinates &&
		(!weighttype || g_Config.bSoftwareSkinning || (decFmt.w0fmt >= DEC_FLOAT_1 && decFmt.w0fmt <= DEC_FLOAT_4)) &&
		(!nrm || (weighttype && g_Config.bSoftwareSkinning) || nrm != 1 || decFmt.nrmfmt == DEC_FLOAT_3)) {
		// Inline fast-path (including software skinning): eliminates 3-5 pointer-to-member calls per vertex.
		const u8 * __restrict p = ptr_;
		u8 * __restrict d = decoded_;
		const int inSize = size;
		const int wt = weighttype;
		const bool swSkin = wt && g_Config.bSoftwareSkinning;
		const int nw = nweights;
		const int numBones = (wt && !swSkin) ? TranslateNumBones(nw) : 0;
		const int tcFmt = tc;
		const int tcOff = tcoff;
		const int uvOff = decFmt.uvoff;
		const bool prescale = tcFmt && (gstate.getUVGenMode() == GE_TEXMAP_TEXTURE_COORDS || gstate.getUVGenMode() == GE_TEXMAP_UNKNOWN);
		const float tcBase = (tcFmt == 1) ? (1.0f / 128.0f) : (tcFmt == 2 ? (1.0f / 32768.0f) : 1.0f);
		const float uMul = prescale ? (tcBase * gstate_c.uv.uScale) : tcBase;
		const float vMul = prescale ? (tcBase * gstate_c.uv.vScale) : tcBase;
		const float uAdd = prescale ? gstate_c.uv.uOff : 0.0f;
		const float vAdd = prescale ? gstate_c.uv.vOff : 0.0f;
		const int colFmt = col;
		const int cOff = coloff;
		const int dc0Off = decFmt.c0off;
		const int nrmFmt = nrm;
		const int nOff = nrmoff;
		const int dnOff = decFmt.nrmoff;
		const int posFmt = pos;
		const int pOff = posoff;
		const int dpOff = decFmt.posoff;
		bool fullAlpha = gstate_c.vertexFullAlpha;
		const float * __restrict s8Lut = GetS8Div128Lut();
		const float * __restrict u8Lut = GetU8Div128Lut();

		// Zero-branch specialized dispatch for the hottest u8-weight software-skinned formats.
#ifdef __wiiu__
		if (swSkin && wt == 1 && (colFmt == 0 || colFmt == 7) && (posFmt == 2 || posFmt == 3)) {
			const float tcUMul = (tcFmt == 1) ? (prescale ? gstate_c.uv.uScale : 1.0f) : uMul;
			const float tcVMul = (tcFmt == 1) ? (prescale ? gstate_c.uv.vScale : 1.0f) : vMul;
			const u32 key = (tcFmt << 8) | (colFmt << 4) | (nrmFmt << 2) | posFmt;
			bool handled = true;
			switch (key) {
#define DISPATCH_SK(T, C, N, P) \
	case ((T << 8) | (C << 4) | (N << 2) | P): \
		DecodeSkinnedU8T<T, C, N, P>(p, d, count, inSize, stride, nw, tcOff, uvOff, tcUMul, tcVMul, uAdd, vAdd, cOff, dc0Off, nOff, dnOff, pOff, dpOff, fullAlpha); \
		break
			DISPATCH_SK(2, 0, 1, 2);
			DISPATCH_SK(2, 7, 1, 2);
			DISPATCH_SK(2, 0, 0, 2);
			DISPATCH_SK(2, 7, 0, 2);
			DISPATCH_SK(2, 0, 2, 2);
			DISPATCH_SK(2, 0, 1, 3);
			DISPATCH_SK(3, 0, 3, 3);
			DISPATCH_SK(3, 0, 1, 3);
			DISPATCH_SK(0, 0, 1, 2);
			DISPATCH_SK(0, 0, 0, 2);
#undef DISPATCH_SK
			default:
				handled = false;
				break;
			}
			if (handled) {
				gstate_c.vertexFullAlpha = fullAlpha;
				return;
			}
		}
#endif

		// Zero-branch specialized dispatch for the hottest non-skinned PSP vertex formats.
		if (!wt && (colFmt == 0 || colFmt >= 4)) {
			const float tcUMul = (tcFmt == 1) ? (prescale ? gstate_c.uv.uScale : 1.0f) : uMul;
			const float tcVMul = (tcFmt == 1) ? (prescale ? gstate_c.uv.vScale : 1.0f) : vMul;
			const u32 key = (tcFmt << 8) | (colFmt << 4) | (nrmFmt << 2) | posFmt;
			bool handled = true;
			switch (key) {
#define DISPATCH_NS(T, C, N, P) \
	case ((T << 8) | (C << 4) | (N << 2) | P): \
		DecodeNonSkinnedT<T, C, N, P>(p, d, count, inSize, stride, tcOff, uvOff, tcUMul, tcVMul, uAdd, vAdd, cOff, dc0Off, nOff, dnOff, pOff, dpOff, fullAlpha); \
		break
			DISPATCH_NS(2, 0, 1, 2);
			DISPATCH_NS(2, 7, 1, 2);
			DISPATCH_NS(2, 0, 0, 2);
			DISPATCH_NS(2, 7, 0, 2);
			DISPATCH_NS(2, 6, 0, 2);
			DISPATCH_NS(2, 5, 0, 2);
			DISPATCH_NS(2, 4, 0, 2);
			DISPATCH_NS(2, 6, 1, 2);
			DISPATCH_NS(2, 5, 1, 2);
			DISPATCH_NS(2, 4, 1, 2);
			DISPATCH_NS(2, 0, 2, 2);
			DISPATCH_NS(2, 7, 2, 2);
			DISPATCH_NS(2, 0, 1, 3);
			DISPATCH_NS(2, 7, 1, 3);
			DISPATCH_NS(3, 0, 3, 3);
			DISPATCH_NS(3, 7, 3, 3);
			DISPATCH_NS(3, 0, 0, 3);
			DISPATCH_NS(3, 7, 0, 3);
			DISPATCH_NS(3, 0, 1, 3);
			DISPATCH_NS(3, 7, 1, 3);
			DISPATCH_NS(0, 7, 0, 2);
			DISPATCH_NS(0, 7, 0, 3);
			DISPATCH_NS(0, 6, 0, 2);
			DISPATCH_NS(0, 5, 0, 2);
			DISPATCH_NS(0, 4, 0, 2);
			DISPATCH_NS(0, 0, 1, 2);
			DISPATCH_NS(0, 0, 0, 2);
			DISPATCH_NS(0, 0, 0, 3);
			DISPATCH_NS(1, 7, 0, 2);
			DISPATCH_NS(1, 0, 1, 2);
			DISPATCH_NS(1, 7, 0, 1);
			DISPATCH_NS(2, 0, 1, 1);
			DISPATCH_NS(2, 7, 1, 1);
			DISPATCH_NS(2, 7, 0, 1);
#undef DISPATCH_NS
			default:
				handled = false;
				break;
			}
			if (handled) {
				gstate_c.vertexFullAlpha = fullAlpha;
				return;
			}
		}

		for (; count > 0; --count) {
			float weights[8];
			if (wt) {
				if (swSkin) {
					if (wt == 1) {
						for (int j = 0; j < nw; ++j)
							weights[j] = u8Lut[p[j]];
					} else if (wt == 2) {
						const u16_le *w16 = (const u16_le *)p;
						for (int j = 0; j < nw; ++j) {
							u16 w = w16[j];
							weights[j] = w ? (w * (1.0f / 32768.0f)) : 0.0f;
						}
					} else {
						const float_le *wf = (const float_le *)p;
						for (int j = 0; j < nw; ++j)
							weights[j] = wf[j];
					}
#ifndef __wiiu__
					ComputeSkinMatrix(weights);
#endif
				} else {
					float *outW = (float *)d;
					if (wt == 1) {
						for (int j = 0; j < nw; ++j)
							outW[j] = u8Lut[p[j]];
					} else if (wt == 2) {
						const u16_le *w16 = (const u16_le *)p;
						for (int j = 0; j < nw; ++j)
							outW[j] = w16[j] * (1.0f / 32768.0f);
					} else {
						const float_le *wf = (const float_le *)p;
						for (int j = 0; j < nw; ++j)
							outW[j] = wf[j];
					}
					for (int j = nw; j < numBones; ++j)
						outW[j] = 0.0f;
				}
			}
			if (tcFmt) {
				float *uv = (float *)(d + uvOff);
				if (tcFmt == 2) {
					const u16_le *uv16 = (const u16_le *)(p + tcOff);
					uv[0] = (float)uv16[0] * uMul + uAdd;
					uv[1] = (float)uv16[1] * vMul + vAdd;
				} else if (tcFmt == 1) {
					const u8 *uv8 = p + tcOff;
					uv[0] = u8Lut[uv8[0]] * (prescale ? gstate_c.uv.uScale : 1.0f) + uAdd;
					uv[1] = u8Lut[uv8[1]] * (prescale ? gstate_c.uv.vScale : 1.0f) + vAdd;
				} else {
					const float_le *uvf = (const float_le *)(p + tcOff);
					uv[0] = uvf[0] * uMul + uAdd;
					uv[1] = uvf[1] * vMul + vAdd;
				}
			}
			if (colFmt) {
				u32_le *cOut = (u32_le *)(d + dc0Off);
				if (colFmt == 7) {
					u32 cdata = *(const u32_le *)(p + cOff);
					fullAlpha &= ((cdata >> 24) == 0xFF);
					*cOut = cdata;
				} else if (colFmt == 6) {
					u16 cdata = *(const u16_le *)(p + cOff);
					fullAlpha &= ((cdata >> 12) == 0xF);
					u32 c = Convert4To8(cdata & 0xF) |
						(Convert4To8((cdata >> 4) & 0xF) << 8) |
						(Convert4To8((cdata >> 8) & 0xF) << 16) |
						(Convert4To8((cdata >> 12) & 0xF) << 24);
					*cOut = c;
				} else if (colFmt == 5) {
					u16 cdata = *(const u16_le *)(p + cOff);
					fullAlpha &= ((cdata >> 15) != 0);
					u32 c = Convert5To8(cdata & 0x1f) |
						(Convert5To8((cdata >> 5) & 0x1f) << 8) |
						(Convert5To8((cdata >> 10) & 0x1f) << 16) |
						((cdata >> 15) ? (255u << 24) : 0);
					*cOut = c;
				} else {
					u16 cdata = *(const u16_le *)(p + cOff);
					u32 c = Convert5To8(cdata & 0x1f) |
						(Convert6To8((cdata >> 5) & 0x3f) << 8) |
						(Convert5To8((cdata >> 11) & 0x1f) << 16) |
						(255u << 24);
					*cOut = c;
				}
			}
			if (swSkin) {
				float inNrm[3];
				float *outNrm = nullptr;
				if (nrmFmt) {
					outNrm = (float *)(d + dnOff);
					if (nrmFmt == 1) {
						const u8 *sv = p + nOff;
						inNrm[0] = s8Lut[sv[0]];
						inNrm[1] = s8Lut[sv[1]];
						inNrm[2] = s8Lut[sv[2]];
					} else if (nrmFmt == 2) {
						const s16_le *sv = (const s16_le *)(p + nOff);
						inNrm[0] = (float)sv[0] * (1.0f / 32768.0f);
						inNrm[1] = (float)sv[1] * (1.0f / 32768.0f);
						inNrm[2] = (float)sv[2] * (1.0f / 32768.0f);
					} else {
						const float_le *fv = (const float_le *)(p + nOff);
						inNrm[0] = fv[0];
						inNrm[1] = fv[1];
						inNrm[2] = fv[2];
					}
				}
				float inPos[3];
				float *outPos = (float *)(d + dpOff);
				if (posFmt == 2) {
					const s16_le *pv = (const s16_le *)(p + pOff);
					inPos[0] = (float)pv[0] * (1.0f / 32768.0f);
					inPos[1] = (float)pv[1] * (1.0f / 32768.0f);
					inPos[2] = (float)pv[2] * (1.0f / 32768.0f);
				} else if (posFmt == 3) {
					const float_le *pv = (const float_le *)(p + pOff);
					inPos[0] = pv[0];
					inPos[1] = pv[1];
					inPos[2] = pv[2];
				} else {
					const u8 *pv = p + pOff;
					inPos[0] = s8Lut[pv[0]];
					inPos[1] = s8Lut[pv[1]];
					inPos[2] = s8Lut[pv[2]];
				}
#ifdef __wiiu__
				SkinPosNormPS(weights, nw, gstate.boneMatrix, inPos, outPos, outNrm ? inNrm : nullptr, outNrm);
#else
				if (outNrm)
					Norm3ByMatrix43(outNrm, inNrm, skinMatrix);
				Vec3ByMatrix43(outPos, inPos, skinMatrix);
#endif
			} else {
				if (nrmFmt) {
					if (nrmFmt == 1) {
						float *no = (float *)(d + dnOff);
						const u8 *sv = p + nOff;
						no[0] = s8Lut[sv[0]];
						no[1] = s8Lut[sv[1]];
						no[2] = s8Lut[sv[2]];
					} else if (nrmFmt == 2) {
						s16 *no = (s16 *)(d + dnOff);
						const s16_le *sv = (const s16_le *)(p + nOff);
						no[0] = sv[0];
						no[1] = sv[1];
						no[2] = sv[2];
						no[3] = 0;
					} else {
						u32 *no = (u32 *)(d + dnOff);
						const u32_le *fv = (const u32_le *)(p + nOff);
						no[0] = fv[0];
						no[1] = fv[1];
						no[2] = fv[2];
					}
				}
				if (posFmt == 2) {
					float *po = (float *)(d + dpOff);
					const s16_le *pv = (const s16_le *)(p + pOff);
					po[0] = pv[0] * (1.0f / 32768.0f);
					po[1] = pv[1] * (1.0f / 32768.0f);
					po[2] = pv[2] * (1.0f / 32768.0f);
				} else if (posFmt == 3) {
					u32 *po = (u32 *)(d + dpOff);
					const u32_le *pv = (const u32_le *)(p + pOff);
					po[0] = pv[0];
					po[1] = pv[1];
					po[2] = pv[2];
				} else {
					float *po = (float *)(d + dpOff);
					const u8 *pv = p + pOff;
					po[0] = s8Lut[pv[0]];
					po[1] = s8Lut[pv[1]];
					po[2] = s8Lut[pv[2]];
				}
			}
			p += inSize;
			d += stride;
		}
		gstate_c.vertexFullAlpha = fullAlpha;
	} else {
		// Interpret the decode steps
		for (; count; count--) {
			for (int i = 0; i < numSteps_; i++) {
				((*this).*steps_[i])();
			}
			ptr_ += size;
			decoded_ += stride;
		}
	}
}

static const char *posnames[4] = { "?", "s8", "s16", "f" };
static const char *nrmnames[4] = { "", "s8", "s16", "f" };
static const char *tcnames[4] = { "", "u8", "u16", "f" };
static const char *idxnames[4] = { "-", "u8", "u16", "?" };
static const char *weightnames[4] = { "-", "u8", "u16", "f" };
static const char *colnames[8] = { "", "?", "?", "?", "565", "5551", "4444", "8888" };

int VertexDecoder::ToString(char *output) const {
	char * start = output;
	output += sprintf(output, "P: %s ", posnames[pos]);
	if (nrm)
		output += sprintf(output, "N: %s ", nrmnames[nrm]);
	if (col)
		output += sprintf(output, "C: %s ", colnames[col]);
	if (tc)
		output += sprintf(output, "T: %s ", tcnames[tc]);
	if (weighttype)
		output += sprintf(output, "W: %s (%ix) ", weightnames[weighttype], nweights);
	if (idx)
		output += sprintf(output, "I: %s ", idxnames[idx]);
	if (morphcount > 1)
		output += sprintf(output, "Morph: %i ", morphcount);
	if (throughmode)
		output += sprintf(output, " (through)");

	output += sprintf(output, " (size: %i)", VertexSize());
	return output - start;
}

std::string VertexDecoder::GetString(DebugShaderStringType stringType) {
	char buffer[256];
	switch (stringType) {
	case SHADER_STRING_SHORT_DESC:
		ToString(buffer);
		return std::string(buffer);
	case SHADER_STRING_SOURCE_CODE:
		{
			if (!jitted_)
				return "Not compiled";
			std::vector<std::string> lines;
#if PPSSPP_ARCH(ARM64)
			lines = DisassembleArm64((const u8 *)jitted_, jittedSize_);
#elif PPSSPP_ARCH(ARM)
			lines = DisassembleArm2((const u8 *)jitted_, jittedSize_);
#elif PPSSPP_ARCH(MIPS) || PPSSPP_ARCH(MIPS64)
			// No MIPS disassembler defined
#elif defined(__wiiu__)
			lines = DisassemblePPC((const u8 *)jitted_, jittedSize_);
#else
			lines = DisassembleX86((const u8 *)jitted_, jittedSize_);
#endif
			std::string buffer;
			for (auto line : lines) {
				buffer += line;
				buffer += "\n";
			}
			return buffer;
		}

	default:
		return "N/A";
	}
}

VertexDecoderJitCache::VertexDecoderJitCache()
#if PPSSPP_ARCH(ARM64)
 : fp(this)
#endif
{
#ifndef __wiiu__
	// 256k should be enough.
	AllocCodeSpace(1024 * 64 * 4);
#endif

	// Add some random code to "help" MSVC's buggy disassembler :(
#if defined(_WIN32) && (defined(_M_IX86) || defined(_M_X64))
	using namespace Gen;
	for (int i = 0; i < 100; i++) {
		MOV(32, R(EAX), R(EBX));
		RET();
	}
#elif defined(ARM)
	BKPT(0);
	BKPT(0);
#endif
}

void VertexDecoderJitCache::Clear() {
	ClearCodeSpace(0);
}
