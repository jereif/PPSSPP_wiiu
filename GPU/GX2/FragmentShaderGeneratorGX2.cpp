// Copyright (c) 2017- PPSSPP Project.

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

#include "GPU/Common/ShaderCommon.h"
#include "GPU/Common/ShaderUniforms.h"
#include "GPU/Common/GPUStateUtils.h"
#include "GPU/ge_constants.h"

#include "GPU/GX2/FragmentShaderGeneratorGX2.h"
#include "GPU/GX2/ShaderManagerGX2.h"

#include <wiiu/gx2/shader_emitter.hpp>
#include <wiiu/gx2/shader_disasm.hpp>
#include <wiiu/gx2/shader_info.h>

#include "GPU/Vulkan/FragmentShaderGeneratorVulkan.h"
#include <wiiu/os/debug.h>

using namespace GX2Gen;

// Diagnostic variants of the pixel shader, cycled by DrawEngineGX2 on traced frames.
int g_gx2FsDbg = 0;

class FragmentShaderGeneratorGX2 : private GX2PixelShaderEmitter {
public:
	FragmentShaderGeneratorGX2() {}
	bool Supported(const FShaderID &id);
	void Emit(const FShaderID &id, GX2PixelShader *ps);
};

bool FragmentShaderGeneratorGX2::Supported(const FShaderID &id) {
	FShaderID unsupported;
	unsupported.SetBit(FS_BIT_SHADER_DEPAL);
	unsupported.SetBit(FS_BIT_COLOR_TEST);
	unsupported.SetBits(FS_BIT_COLOR_TEST_FUNC, 2, -1);
	unsupported.SetBit(FS_BIT_COLOR_AGAINST_ZERO);
	//	unsupported.SetBit(FS_BIT_DO_TEXTURE_PROJ);
	//	unsupported.SetBits(FS_BIT_STENCIL_TO_ALPHA, 2, -1);
	//	unsupported.SetBits(FS_BIT_REPLACE_ALPHA_WITH_STENCIL_TYPE, 4, -1);
	unsupported.SetBits(FS_BIT_REPLACE_LOGIC_OP_TYPE, 2, -1);
	unsupported.SetBit(FS_BIT_FLATSHADE);
	unsupported.SetBit(FS_BIT_NO_DEPTH_CANNOT_DISCARD_STENCIL);

	if ((unsupported.d[0] & id.d[0]) || (unsupported.d[1] & id.d[1]))
		return false;

	// Blend replacement that only needs shader math is supported; reading the framebuffer is not.
	ReplaceBlendType rb = (ReplaceBlendType)id.Bits(FS_BIT_REPLACE_BLEND, 3);
	if (rb == REPLACE_BLEND_COPY_FBO || rb > REPLACE_BLEND_COPY_FBO)
		return false;
	if (rb == REPLACE_BLEND_PRE_SRC || rb == REPLACE_BLEND_PRE_SRC_2X_ALPHA) {
		switch ((GEBlendSrcFactor)id.Bits(FS_BIT_BLENDFUNC_A, 4)) {
		case GE_SRCBLEND_SRCALPHA:
		case GE_SRCBLEND_INVSRCALPHA:
		case GE_SRCBLEND_DOUBLESRCALPHA:
		case GE_SRCBLEND_DOUBLEINVSRCALPHA:
		case GE_SRCBLEND_FIXA:
			break;
		default:
			return false;
		}
	}
	return true;
}

void FragmentShaderGeneratorGX2::Emit(const FShaderID &id, GX2PixelShader *ps) {
	GEComparison alphaTestFunc = (GEComparison)id.Bits(FS_BIT_ALPHA_TEST_FUNC, 3);
	//	GEComparison colorTestFunc = (GEComparison)id.Bits(FS_BIT_COLOR_TEST_FUNC, 2);
	GETexFunc texFunc = (GETexFunc)id.Bits(FS_BIT_TEXFUNC, 3);
	const int dbg = g_gx2FsDbg;
	if (dbg == 4)
		texFunc = GE_TEXFUNC_REPLACE;
	ReplaceBlendType replaceBlend = (ReplaceBlendType)(id.Bits(FS_BIT_REPLACE_BLEND, 3));
	ReplaceAlphaType stencilToAlpha = (ReplaceAlphaType)(id.Bits(FS_BIT_STENCIL_TO_ALPHA, 2));
	if (g_gx2FsDbg == 3)
		stencilToAlpha = REPLACE_ALPHA_NO;
	GEBlendSrcFactor replaceBlendFuncA = (GEBlendSrcFactor)id.Bits(FS_BIT_BLENDFUNC_A, 4);
	//	GEBlendDstFactor replaceBlendFuncB = (GEBlendDstFactor)id.Bits(FS_BIT_BLENDFUNC_B, 4);
	//	GEBlendMode replaceBlendEq = (GEBlendMode)id.Bits(FS_BIT_BLENDEQ, 3);
	StencilValueType replaceAlphaWithStencilType = (StencilValueType)id.Bits(FS_BIT_REPLACE_ALPHA_WITH_STENCIL_TYPE, 4);

	bool clearMode = id.Bit(FS_BIT_CLEARMODE);
	bool doTexture = !clearMode && id.Bit(FS_BIT_DO_TEXTURE);
	bool doLMode = !clearMode && id.Bit(FS_BIT_LMODE);
	bool doFog = !clearMode && id.Bit(FS_BIT_ENABLE_FOG);
	bool testForceToZero = id.Bit(FS_BIT_TEST_DISCARD_TO_ZERO);

	// All import registers must be allocated before any general-purpose register.
	Reg color = allocImportReg(PSInput::COLOR0);
	Reg color1 = doLMode ? allocImportReg(PSInput::COLOR1) : Reg(0);
	Reg fogDepth = doFog ? allocImportReg(PSInput::FOGDEPTH) : Reg(0);
	Reg coords = doTexture ? allocImportReg(PSInput::COORDS) : Reg(0);

	bool needsTmp = !clearMode && (
		(doTexture && id.Bit(FS_BIT_SHADER_TEX_CLAMP)) ||
		(doTexture && ((texFunc == GE_TEXFUNC_DECAL && id.Bit(FS_BIT_TEXALPHA)) || texFunc == GE_TEXFUNC_BLEND)) ||
		(id.Bit(FS_BIT_ALPHA_TEST) && !id.Bit(FS_BIT_ALPHA_AGAINST_ZERO)) ||
		((replaceBlend == REPLACE_BLEND_PRE_SRC || replaceBlend == REPLACE_BLEND_PRE_SRC_2X_ALPHA) &&
			(replaceBlendFuncA == GE_SRCBLEND_INVSRCALPHA || replaceBlendFuncA == GE_SRCBLEND_DOUBLESRCALPHA || replaceBlendFuncA == GE_SRCBLEND_DOUBLEINVSRCALPHA)) ||
		doFog);
	Reg tmp = (needsTmp || stencilToAlpha == REPLACE_ALPHA_DUALSOURCE) ? allocReg() : Reg(0);

	if (!clearMode) {
		if (doTexture && dbg != 5) {
			if (id.Bit(FS_BIT_DO_TEXTURE_PROJ)) {
				RECIP_IEEE(___(x), coords(z));
				ALU_LAST();
				MUL(coords(x), coords(x), PS);
				MUL(coords(y), coords(y), PS);
				ALU_LAST();
			}
			if (id.Bit(FS_BIT_SHADER_TEX_CLAMP)) {
				// Clamp/wrap inside a larger surface (texclamp = wrap xy, half texel xy).
				KCacheReg texClamp(UB_Bindings::Base, offsetof(UB_VS_FS_Base, texClamp), this);
				if (id.Bit(FS_BIT_CLAMP_S)) {
					MULADD(tmp(x), texClamp(z), C(-1.0f), texClamp(x));
					ALU_LAST();
					MAX(coords(x), coords(x), texClamp(z));
					ALU_LAST();
					MIN(coords(x), coords(x), tmp(x));
					ALU_LAST();
				} else {
					RECIP_IEEE(___(x), texClamp(x));
					ALU_LAST();
					MUL(___(x), coords(x), PS);
					ALU_LAST();
					FRACT(___(x), PV(x));
					ALU_LAST();
					MUL(coords(x), PV(x), texClamp(x));
					ALU_LAST();
				}
				if (id.Bit(FS_BIT_CLAMP_T)) {
					MULADD(tmp(x), texClamp(w), C(-1.0f), texClamp(y));
					ALU_LAST();
					MAX(coords(y), coords(y), texClamp(w));
					ALU_LAST();
					MIN(coords(y), coords(y), tmp(x));
					ALU_LAST();
				} else {
					RECIP_IEEE(___(x), texClamp(y));
					ALU_LAST();
					MUL(___(x), coords(y), PS);
					ALU_LAST();
					FRACT(___(x), PV(x));
					ALU_LAST();
					MUL(coords(y), PV(x), texClamp(y));
					ALU_LAST();
				}
				if (id.Bit(FS_BIT_TEXTURE_AT_OFFSET)) {
					KCacheReg texOff(UB_Bindings::Base, offsetof(UB_VS_FS_Base, texClampOffset), this);
					ADD(coords(x), coords(x), texOff(x));
					ADD(coords(y), coords(y), texOff(y));
					ALU_LAST();
				}
			}
			Reg sample = coords;
			SAMPLE(sample, coords(x, y, _0_, _0_), 0, 0);
			if (id.Bit(FS_BIT_BGRA_TEXTURE))
				sample = sample(b, g, r, a);

			switch (texFunc) {
			case GE_TEXFUNC_REPLACE:
				MOV(color(r), sample(r));
				MOV(color(g), sample(g));
				MOV(color(b), sample(b));
				if (id.Bit(FS_BIT_TEXALPHA))
					MOV(color(a), sample(a));
				ALU_LAST();
				break;
			case GE_TEXFUNC_DECAL:
				if (id.Bit(FS_BIT_TEXALPHA)) {
					MULADD(tmp(r), color(r), C(-1.0f), sample(r));
					MULADD(tmp(g), color(g), C(-1.0f), sample(g));
					MULADD(tmp(b), color(b), C(-1.0f), sample(b));
					ALU_LAST();
					MULADD(color(r), tmp(r), sample(a), color(r));
					MULADD(color(g), tmp(g), sample(a), color(g));
					MULADD(color(b), tmp(b), sample(a), color(b));
					ALU_LAST();
				} else {
					MOV(color(r), sample(r));
					MOV(color(g), sample(g));
					MOV(color(b), sample(b));
					ALU_LAST();
				}
				break;
			case GE_TEXFUNC_BLEND: {
				KCacheReg texEnv(UB_Bindings::Base, offsetof(UB_VS_FS_Base, texEnvColor), this);
				MULADD(tmp(r), color(r), C(-1.0f), texEnv(r));
				MULADD(tmp(g), color(g), C(-1.0f), texEnv(g));
				MULADD(tmp(b), color(b), C(-1.0f), texEnv(b));
				ALU_LAST();
				MULADD(color(r), tmp(r), sample(r), color(r));
				MULADD(color(g), tmp(g), sample(g), color(g));
				MULADD(color(b), tmp(b), sample(b), color(b));
				if (id.Bit(FS_BIT_TEXALPHA))
					MUL(color(a), color(a), sample(a));
				ALU_LAST();
				break;
			}
			case GE_TEXFUNC_MODULATE:
				MUL(color(r), color(r), sample(r));
				MUL(color(g), color(g), sample(g));
				MUL(color(b), color(b), sample(b));
				if (id.Bit(FS_BIT_TEXALPHA))
					MUL(color(a), color(a), sample(a));
				ALU_LAST();
				break;
			default:
			case GE_TEXFUNC_ADD:
				ADD(color(r), color(r), sample(r));
				ADD(color(g), color(g), sample(g));
				ADD(color(b), color(b), sample(b));
				if (id.Bit(FS_BIT_TEXALPHA))
					MUL(color(a), color(a), sample(a));
				ALU_LAST();
				MIN(color(r), color(r), C(1.0f));
				MIN(color(g), color(g), C(1.0f));
				MIN(color(b), color(b), C(1.0f));
				ALU_LAST();
				break;
			}
		}

		if (doLMode && dbg != 1) {
			ADD(color(r), color(r), color1(r));
			ADD(color(g), color(g), color1(g));
			ADD(color(b), color(b), color1(b));
			ALU_LAST();
			MIN(color(r), color(r), C(1.0f));
			MIN(color(g), color(g), C(1.0f));
			MIN(color(b), color(b), C(1.0f));
			ALU_LAST();
		}

		if (doTexture && id.Bit(FS_BIT_COLOR_DOUBLE) && dbg != 2) {
			MOVx2(color(r), color(r));
			MOVx2(color(g), color(g));
			MOVx2(color(b), color(b));
			ALU_LAST();
			MIN(color(r), color(r), C(1.0f));
			MIN(color(g), color(g), C(1.0f));
			MIN(color(b), color(b), C(1.0f));
			ALU_LAST();
		}

		if (id.Bit(FS_BIT_ALPHA_TEST)) {
			if (id.Bit(FS_BIT_ALPHA_AGAINST_ZERO)) {
				if (alphaTestFunc == GE_COMP_NOTEQUAL || alphaTestFunc == GE_COMP_GREATER) {
					if (testForceToZero) {
						ADD(___(x), color(a), C(-0.002f));
						ALU_LAST();
						CNDGT(color(a), PV(x), color(a), C(0.0f));
						ALU_LAST();
					} else {
						KILLGT(x, C(0.002f), color(a));
						ALU_LAST();
					}
				} else if (alphaTestFunc != GE_COMP_NEVER && alphaTestFunc != GE_COMP_ALWAYS) {
					if (testForceToZero) {
						ADD(___(x), color(a), C(-0.002f));
						ALU_LAST();
						CNDGT(color(a), PV(x), C(0.0f), color(a));
						ALU_LAST();
					} else {
						KILLGT(x, color(a), C(0.002f));
						ALU_LAST();
					}
				} else if (alphaTestFunc == GE_COMP_NEVER) {
					if (testForceToZero) {
						MOV(color(a), C(0.0f));
						ALU_LAST();
					} else {
						KILLE(x, color(a), color(a));
						ALU_LAST();
					}
				}
			} else if (alphaTestFunc == GE_COMP_NEVER) {
				if (testForceToZero) {
					MOV(color(a), C(0.0f));
					ALU_LAST();
				} else {
					KILLE(x, color(a), color(a));
					ALU_LAST();
				}
			} else if (alphaTestFunc != GE_COMP_ALWAYS) {
				MIN(tmp(x), color(a), C(1.0f));
				ALU_LAST();
				MAX(tmp(x), tmp(x), C(0.0f));
				ALU_LAST();
				MULADD(tmp(x), tmp(x), C(255.0f), C(0.5f));
				ALU_LAST();
				FLOOR(tmp(x), tmp(x));
				ALU_LAST();
				FLT_TO_INT(tmp(x), tmp(x));
				ALU_LAST();
				KCacheChannel alphaMask(UB_Bindings::Base, offsetof(UB_VS_FS_Base, colorTestMask) + 12, this);
				KCacheChannel alphaRef(UB_Bindings::Base, offsetof(UB_VS_FS_Base, alphaColorRef) + 12, this);
				AND_INT(tmp(x), tmp(x), alphaMask);
				ALU_LAST();
				if (testForceToZero) {
					SUB_INT(___(x), tmp(x), alphaRef);
					ALU_LAST();
					switch (alphaTestFunc) {
					case GE_COMP_EQUAL:    CNDE_INT(color(a), PV(x), color(a), C(0.0f)); break;
					case GE_COMP_NOTEQUAL: CNDE_INT(color(a), PV(x), C(0.0f), color(a)); break;
					case GE_COMP_LESS:     CNDGE_INT(color(a), PV(x), C(0.0f), color(a)); break;
					case GE_COMP_LEQUAL:   CNDGT_INT(color(a), PV(x), C(0.0f), color(a)); break;
					case GE_COMP_GREATER:  CNDGT_INT(color(a), PV(x), color(a), C(0.0f)); break;
					case GE_COMP_GEQUAL:   CNDGE_INT(color(a), PV(x), color(a), C(0.0f)); break;
					default: break;
					}
					ALU_LAST();
				} else {
					switch (alphaTestFunc) {
					case GE_COMP_EQUAL:    KILLNE_INT(x, tmp(x), alphaRef); break;
					case GE_COMP_NOTEQUAL: KILLE_INT(x, tmp(x), alphaRef); break;
					case GE_COMP_LESS:     KILLGE_INT(x, tmp(x), alphaRef); break;
					case GE_COMP_LEQUAL:   KILLGT_INT(x, tmp(x), alphaRef); break;
					case GE_COMP_GREATER:  KILLGE_INT(x, alphaRef, tmp(x)); break;
					case GE_COMP_GEQUAL:   KILLGT_INT(x, alphaRef, tmp(x)); break;
					default: break;
					}
					ALU_LAST();
				}
			}
		}

		if (doFog) {
			KCacheReg fogColor(UB_Bindings::Base, offsetof(UB_VS_FS_Base, fogColor), this);
			MIN(fogDepth(x), fogDepth(x), C(1.0f));
			ALU_LAST();
			MAX(fogDepth(x), fogDepth(x), C(0.0f));
			ALU_LAST();
			MULADD(tmp(r), fogColor(r), C(-1.0f), color(r));
			MULADD(tmp(g), fogColor(g), C(-1.0f), color(g));
			MULADD(tmp(b), fogColor(b), C(-1.0f), color(b));
			ALU_LAST();
			MULADD(color(r), tmp(r), fogDepth(x), fogColor(r));
			MULADD(color(g), tmp(g), fogDepth(x), fogColor(g));
			MULADD(color(b), tmp(b), fogDepth(x), fogColor(b));
			ALU_LAST();
		}

		if (replaceBlend == REPLACE_BLEND_2X_SRC) {
			MOVx2(color(r), color(r));
			MOVx2(color(g), color(g));
			MOVx2(color(b), color(b));
			ALU_LAST();
		}

		if (replaceBlend == REPLACE_BLEND_PRE_SRC || replaceBlend == REPLACE_BLEND_PRE_SRC_2X_ALPHA) {
			switch (replaceBlendFuncA) {
			case GE_SRCBLEND_SRCALPHA:
				MUL(color(r), color(r), color(a));
				MUL(color(g), color(g), color(a));
				MUL(color(b), color(b), color(a));
				ALU_LAST();
				break;
			case GE_SRCBLEND_FIXA: {
				KCacheReg fixA(UB_Bindings::Base, offsetof(UB_VS_FS_Base, blendFixA), this);
				MUL(color(r), color(r), fixA(r));
				MUL(color(g), color(g), fixA(g));
				MUL(color(b), color(b), fixA(b));
				ALU_LAST();
				break;
			}
			default:
				if (replaceBlendFuncA == GE_SRCBLEND_INVSRCALPHA) {
					MULADD(tmp(x), color(a), C(-1.0f), C(1.0f));
				} else if (replaceBlendFuncA == GE_SRCBLEND_DOUBLESRCALPHA) {
					MULADD(tmp(x), color(a), C(2.0f), C(0.0f));
				} else { // GE_SRCBLEND_DOUBLEINVSRCALPHA
					MULADD(tmp(x), color(a), C(-2.0f), C(1.0f));
				}
				ALU_LAST();
				MUL(color(r), color(r), tmp(x));
				MUL(color(g), color(g), tmp(x));
				MUL(color(b), color(b), tmp(x));
				ALU_LAST();
				break;
			}
		}

		if (replaceBlend == REPLACE_BLEND_2X_ALPHA || replaceBlend == REPLACE_BLEND_PRE_SRC_2X_ALPHA) {
			MOVx2(color(a), color(a));
			ALU_LAST();
		}
	}

	SrcChannel replacedAlpha = C(0.0f);
	if (stencilToAlpha != REPLACE_ALPHA_NO) {
		switch (replaceAlphaWithStencilType) {
		case STENCIL_VALUE_UNIFORM: {
			replacedAlpha = KCacheChannel(UB_Bindings::Base, offsetof(UB_VS_FS_Base, stencil), this);
			break;
		}

		case STENCIL_VALUE_ZERO: replacedAlpha = C(0.0f); break;

		case STENCIL_VALUE_ONE:
		case STENCIL_VALUE_INVERT:
			// In invert, we subtract by one, but we want to output one here.
			replacedAlpha = C(1.0f);
			break;

		case STENCIL_VALUE_INCR_4:
		case STENCIL_VALUE_DECR_4:
			// We're adding/subtracting, just by the smallest value in 4-bit.
			replacedAlpha = C(1.0f / 15.0f);
			break;

		case STENCIL_VALUE_INCR_8:
		case STENCIL_VALUE_DECR_8:
			// We're adding/subtracting, just by the smallest value in 8-bit.
			replacedAlpha = C(1.0f / 255.0f);
			break;

		case STENCIL_VALUE_KEEP:
			// Do nothing. We'll mask out the alpha using color mask.
			break;
		}
	}

	if (dbg == 6) {
		MOV(color(r), C(1.0f));
		MOV(color(g), C(0.0f));
		MOV(color(b), C(0.0f));
		ALU_LAST();
	}

	switch (stencilToAlpha) {
	case REPLACE_ALPHA_DUALSOURCE: {
		MOV(tmp(r), color(r));
		MOV(tmp(g), color(g));
		MOV(tmp(b), color(b));
		MOV(tmp(a), replacedAlpha);
		ALU_LAST();
		EXP(PIX0, tmp);
		EXP(PIX1, color(_0_, _0_, _0_, a));
		break;
	}

	case REPLACE_ALPHA_YES:
		MOV(color(a), replacedAlpha);
		ALU_LAST();
		EXP(PIX0, color);
		break;

	case REPLACE_ALPHA_NO: EXP(PIX0, color); break;

	default:
		ERROR_LOG(G3D, "Bad stencil-to-alpha type, corrupt ID?");
		EXP(PIX0, color);
		break;
	}

	END_OF_PROGRAM(ps);
}

void GenerateFragmentShaderGX2(const FShaderID &id, GX2PixelShader *ps) {
	FragmentShaderGeneratorGX2 fsGen;
	if (fsGen.Supported(id)) {
		fsGen.Emit(id, ps);
#if 0
		char buffer[0x20000];
		printf("\n### GPU Regs ###\n");
		GX2PixelShaderInfo(ps, buffer);
		puts(buffer);

		printf("\n### ASM ###\n%s\n", FragmentShaderDesc(id).c_str());
		DisassembleGX2Shader(ps->program, ps->size, buffer);
		puts(buffer);

		printf("\n### glsl ###\n");
		GenerateVulkanGLSLFragmentShader(id, buffer, 0);
		puts(buffer);
#endif
	} else {
		NOTICE_LOG(G3D, "unsupported FShaderID: \"%s\" (d0=%08x d1=%08x)", FragmentShaderDesc(id).c_str(), id.d[0], id.d[1]);
		*ps = PShaderAllGX2;
	}
}
