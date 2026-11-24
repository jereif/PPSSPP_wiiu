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

#include "GPU/GPUState.h"
#include "GPU/Common/ShaderCommon.h"
#include "GPU/Common/ShaderUniforms.h"
#include "GPU/ge_constants.h"

#include "GPU/GX2/VertexShaderGeneratorGX2.h"
#include "GPU/GX2/ShaderManagerGX2.h"

#include <wiiu/gx2/shader_emitter.hpp>
#include <wiiu/gx2/shader_disasm.hpp>
#include <wiiu/gx2/shader_info.h>

using namespace GX2Gen;

namespace {

enum DoLightComputation {
	LIGHT_OFF,
	LIGHT_SHADE,
	LIGHT_FULL,
};

class VertexShaderGeneratorGX2 : private GX2VertexShaderEmitter {
public:
	VertexShaderGeneratorGX2() {}
	bool Supported(const VShaderID &id);
	void Emit(const VShaderID &id, GX2VertexShader *vs);

private:
	// Writes dot(vec4(a.xyz, aw), vec4(b.xyz, bw)) into dst using scalar MUL/ADD (avoids R700 DOT4/inline-const quirks).
	template <typename A, typename B>
	void emitDot4(DstChannel dst, A a, SrcChannel aw, B b, SrcChannel bw) {
		MUL(___(x), a(x), b(x));
		MUL(___(y), a(y), b(y));
		MUL(___(z), a(z), b(z));
		MUL(___(w), aw, bw);
		ALU_LAST();
		ADD(___(x), PV(x), PV(y));
		ADD(___(y), PV(z), PV(w));
		ALU_LAST();
		ADD(dst, PV(x), PV(y));
		ALU_LAST();
	}

	template <typename A, typename B>
	void emitDot3ToPV(A a, B b) {
		MUL(___(x), a(x), b(x));
		MUL(___(y), a(y), b(y));
		MUL(___(z), a(z), b(z));
		ALU_LAST();
		ADD(___(x), PV(x), PV(y));
		MOV(___(y), PV(z));
		ALU_LAST();
		ADD(___(x), PV(x), PV(y));
		ALU_LAST();
	}

	// dst.xyz = vec4(v.xyz, wCh) * mat3x4 (must not alias dst with v).
	void emitMulMat3x4Vec4(Reg dst, Reg v, SrcChannel wCh, KCacheRegs mat) {
		emitDot4(dst(x), v, wCh, mat[0], mat[0](w));
		emitDot4(dst(y), v, wCh, mat[1], mat[1](w));
		emitDot4(dst(z), v, wCh, mat[2], mat[2](w));
	}

	// dst.xyzw = v.x*mat[0] + v.y*mat[1] + v.z*mat[2] + wCh*mat[3].
	void emitMulMat4Vec4(Reg dst, Reg v, SrcChannel wCh, KCacheRegs mat) {
		MUL(___(x), v(x), mat[0](x));
		MUL(___(y), v(x), mat[0](y));
		MUL(___(z), v(x), mat[0](z));
		MUL(___(w), v(x), mat[0](w));
		ALU_LAST();

		MULADD(___(x), v(y), mat[1](x), PV(x));
		MULADD(___(y), v(y), mat[1](y), PV(y));
		MULADD(___(z), v(y), mat[1](z), PV(z));
		MULADD(___(w), v(y), mat[1](w), PV(w));
		ALU_LAST();

		MULADD(___(x), v(z), mat[2](x), PV(x));
		MULADD(___(y), v(z), mat[2](y), PV(y));
		MULADD(___(z), v(z), mat[2](z), PV(z));
		MULADD(___(w), v(z), mat[2](w), PV(w));
		ALU_LAST();

		if (wCh.id == ALU_SRC_1) {
			ADD(dst(x), mat[3](x), PV(x));
			ADD(dst(y), mat[3](y), PV(y));
			ADD(dst(z), mat[3](z), PV(z));
			ADD(dst(w), mat[3](w), PV(w));
		} else {
			MULADD(dst(x), wCh, mat[3](x), PV(x));
			MULADD(dst(y), wCh, mat[3](y), PV(y));
			MULADD(dst(z), wCh, mat[3](z), PV(z));
			MULADD(dst(w), wCh, mat[3](w), PV(w));
		}
		ALU_LAST();
	}

	// dst.xyz = normalize(v.xyz), or (0,0,0) when length == 0.
	void emitNormalize3Safe(Reg dst, Reg v, Reg tmpScalar) {
		emitDot3ToPV(v, v);
		MOV(tmpScalar(x), PV(x));
		ALU_LAST();

		RECIPSQRT_IEEE(___(x), tmpScalar(x));
		ALU_LAST();

		MUL(___(x), v(x), PS);
		MUL(___(y), v(y), PS);
		MUL(___(z), v(z), PS);
		ALU_LAST();

		CNDGT(dst(x), tmpScalar(x), PV(x), C(0.0f));
		CNDGT(dst(y), tmpScalar(x), PV(y), C(0.0f));
		CNDGT(dst(z), tmpScalar(x), PV(z), C(0.0f));
		ALU_LAST();
	}

	// dst(dstCh) = (expSrc <= 0.0f) ? 1.0f : ((base > 0.0f) ? pow(base, expSrc) : 0.0f)
	void emitPowSafe(Reg dst, Channel dstCh, SrcChannel baseSrc, SrcChannel expSrc) {
		LOG_CLAMPED(___(x), baseSrc);
		ALU_LAST();

		MUL(___(x), PS, expSrc);
		ALU_LAST();

		EXP_IEEE(___(x), PV(x));
		ALU_LAST();

		CNDGT(___(x), baseSrc, PS, C(0.0f));
		ALU_LAST();

		CNDGT(dst(dstCh), expSrc, PV(x), C(1.0f));
		ALU_LAST();
	}

	void emitDepthRoundZVP(Reg outPos) {
		// z = outPos.z / outPos.w
		RECIP_IEEE(___(x), outPos(w));
		ALU_LAST();

		KCacheReg depthRange(UB_Bindings::Base, offsetof(UB_VS_FS_Base, depthRange), this);
		MUL(___(x), outPos(z), PS);
		ALU_LAST();

		MULADD(___(x), PV(x), depthRange(x), depthRange(y));
		ALU_LAST();

		FLOOR(___(x), PV(x));
		ALU_LAST();

		MULADD(___(x), depthRange(z), C(-1.0f), PV(x));
		ALU_LAST();

		MUL(___(x), PV(x), depthRange(w));
		ALU_LAST();

		MUL(outPos(z), PV(x), outPos(w));
		ALU_LAST();
	}
};

} // namespace

// Set from WiiUMain when sd:/ppsspp/nolight.txt exists (kill switch for the hardware lighting path).
bool g_gx2NoLighting = false;

bool VertexShaderGeneratorGX2::Supported(const VShaderID &id) {
	// Hardware spline/bezier tessellation is not generated here; fall back if ever requested.
	if (id.Bit(VS_BIT_BEZIER) || id.Bit(VS_BIT_SPLINE))
		return false;
	if (id.Bit(VS_BIT_LIGHTING_ENABLE) && g_gx2NoLighting)
		return false;
	return true;
}

void VertexShaderGeneratorGX2::Emit(const VShaderID &id, GX2VertexShader *vs) {
	bool isModeThrough = id.Bit(VS_BIT_IS_THROUGH);
	bool lmode = id.Bit(VS_BIT_LMODE);
	bool doTexture = id.Bit(VS_BIT_DO_TEXTURE);
	bool doTextureTransform = id.Bit(VS_BIT_DO_TEXTURE_TRANSFORM);

	GETexMapMode uvGenMode = static_cast<GETexMapMode>(id.Bits(VS_BIT_UVGEN_MODE, 2));
	GETexProjMapMode uvProjMode = static_cast<GETexProjMapMode>(id.Bits(VS_BIT_UVPROJ_MODE, 2));
	bool doShadeMapping = uvGenMode == GE_TEXMAP_ENVIRONMENT_MAP;

	bool useHWTransform = id.Bit(VS_BIT_USE_HW_TRANSFORM);
	bool hasColor = id.Bit(VS_BIT_HAS_COLOR) || !useHWTransform;
	bool hasNormal = id.Bit(VS_BIT_HAS_NORMAL) && useHWTransform;
	bool hasTexcoord = id.Bit(VS_BIT_HAS_TEXCOORD) || !useHWTransform;
	bool enableFog = id.Bit(VS_BIT_ENABLE_FOG);
	bool flipNormal = id.Bit(VS_BIT_NORM_REVERSE);
	int ls0 = id.Bits(VS_BIT_LS0, 2);
	int ls1 = id.Bits(VS_BIT_LS1, 2);
	bool enableBones = id.Bit(VS_BIT_ENABLE_BONES);
	bool enableLighting = id.Bit(VS_BIT_LIGHTING_ENABLE);
	int matUpdate = id.Bits(VS_BIT_MATERIAL_UPDATE, 3);

	int numBoneWeights = enableBones ? (1 + id.Bits(VS_BIT_BONES, 3)) : 0;

	DoLightComputation doLight[4] = { LIGHT_OFF, LIGHT_OFF, LIGHT_OFF, LIGHT_OFF };
	if (useHWTransform) {
		int shadeLight0 = doShadeMapping ? ls0 : -1;
		int shadeLight1 = doShadeMapping ? ls1 : -1;
		for (int i = 0; i < 4; i++) {
			if (i == shadeLight0 || i == shadeLight1)
				doLight[i] = LIGHT_SHADE;
			if (enableLighting && id.Bit(VS_BIT_LIGHT0_ENABLE + i))
				doLight[i] = LIGHT_FULL;
		}
	}

	// All import registers must be allocated before any temporary registers.
	Reg pos = allocImportReg(VSInput::POSITION);
	Reg w1(0), w2(0), nrm(0), coord(0), col0(0), col1(0);

	bool texcoordInVec3 = !useHWTransform && doTextureTransform && !isModeThrough;
	if (!useHWTransform) {
		if (doTexture && hasTexcoord)
			coord = allocImportReg(VSInput::COORDS);
		if (hasColor) {
			col0 = allocImportReg(VSInput::COLOR0);
			if (lmode)
				col1 = allocImportReg(VSInput::COLOR1);
		}
	} else {
		if (enableBones) {
			w1 = allocImportReg(VSInput::WEIGHT0);
			if (numBoneWeights > 4)
				w2 = allocImportReg(VSInput::WEIGHT1);
		}
		if (hasNormal)
			nrm = allocImportReg(VSInput::NORMAL);
		if (doTexture && hasTexcoord)
			coord = allocImportReg(VSInput::COORDS);
		if (hasColor)
			col0 = allocImportReg(VSInput::COLOR0);
	}

	Reg outPos = allocReg();
	Reg outCol0 = allocReg();
	Reg outCol1 = lmode ? allocReg() : Reg(0);
	Reg outCoords = doTexture ? allocReg() : Reg(0);
	Reg outFog = enableFog ? allocReg() : Reg(0);

	if (!useHWTransform) {
		KCacheRegs proj = isModeThrough
			? KCacheRegs(UB_Bindings::Base, offsetof(UB_VS_FS_Base, proj_through), this)
			: KCacheRegs(UB_Bindings::Base, offsetof(UB_VS_FS_Base, proj), this);
		emitMulMat4Vec4(outPos, pos, C(1.0f), proj);

		if (!isModeThrough && gstate_c.Supports(GPU_ROUND_DEPTH_TO_16BIT)) {
			FLUSH_ALU();
			emitDepthRoundZVP(outPos);
		}

		if (doTexture) {
			if (hasTexcoord) {
				MOV(outCoords(x), coord(x));
				MOV(outCoords(y), coord(y));
				MOV(outCoords(z), texcoordInVec3 ? (SrcChannel)coord(z) : C(1.0f));
			} else {
				MOV(outCoords(x), C(0.0f));
				MOV(outCoords(y), C(0.0f));
				MOV(outCoords(z), C(1.0f));
			}
			ALU_LAST();
		}

		if (hasColor) {
			MOV(outCol0(x), col0(x));
			MOV(outCol0(y), col0(y));
			MOV(outCol0(z), col0(z));
			MOV(outCol0(w), col0(w));
			ALU_LAST();
			if (lmode) {
				MOV(outCol1(x), col1(x));
				MOV(outCol1(y), col1(y));
				MOV(outCol1(z), col1(z));
				ALU_LAST();
			}
		} else {
			FLUSH_ALU();
			KCacheReg matAmb(UB_Bindings::Base, offsetof(UB_VS_FS_Base, matAmbient), this);
			MOV(outCol0(x), matAmb(x));
			MOV(outCol0(y), matAmb(y));
			MOV(outCol0(z), matAmb(z));
			MOV(outCol0(w), matAmb(w));
			ALU_LAST();
			if (lmode) {
				MOV(outCol1(x), C(0.0f));
				MOV(outCol1(y), C(0.0f));
				MOV(outCol1(z), C(0.0f));
				ALU_LAST();
			}
		}

		if (enableFog) {
			MOV(outFog(x), pos(w));
			ALU_LAST();
		}
	} else {
		Reg worldPos = allocReg();
		Reg worldNormal = allocReg();
		Reg viewPos = allocReg();
		Reg tmp0 = allocReg();
		Reg tmp1 = allocReg();
		Reg tmp2 = allocReg();

		// Step 1: Skinning (in ascending bone order so KCACHE bank locking never fails).
		if (!enableBones) {
			FLUSH_ALU();
			KCacheRegs worldMtx(UB_Bindings::Base, offsetof(UB_VS_FS_Base, world), this);
			emitMulMat3x4Vec4(worldPos, pos, C(1.0f), worldMtx);

			if (hasNormal) {
				if (flipNormal) {
					MUL(tmp0(x), nrm(x), C(-1.0f));
					MUL(tmp0(y), nrm(y), C(-1.0f));
					MUL(tmp0(z), nrm(z), C(-1.0f));
					ALU_LAST();
					emitMulMat3x4Vec4(tmp1, tmp0, C(0.0f), worldMtx);
				} else {
					emitMulMat3x4Vec4(tmp1, nrm, C(0.0f), worldMtx);
				}
				emitNormalize3Safe(worldNormal, tmp1, tmp2);
			} else {
				MOV(worldNormal(x), C(0.0f));
				MOV(worldNormal(y), C(0.0f));
				MOV(worldNormal(z), C(1.0f));
				ALU_LAST();
			}
		} else {
			// DrawEngineGX2 sets expandAllWeightsToFloat = true, so Step_Weights*ToFloat already scales weights to [0, 2).
			float weightFactor = 1.0f;

			Reg sm0 = tmp0;
			Reg sm1 = tmp1;
			Reg sm2 = tmp2;

			FLUSH_ALU();
			for (int i = 0; i < numBoneWeights; i++) {
				KCacheRegs boneMtx(UB_Bindings::Bones, offsetof(UB_VS_Bones, bones) + i * 12 * sizeof(float), this);
				SrcChannel wi = (i < 4) ? (SrcChannel)w1((Channel)i) : (SrcChannel)w2((Channel)(i - 4));
				if (i == 0) {
					MUL(sm0(x), wi, boneMtx[0](x));
					MUL(sm0(y), wi, boneMtx[0](y));
					MUL(sm0(z), wi, boneMtx[0](z));
					MUL(sm0(w), wi, boneMtx[0](w));
					ALU_LAST();
					MUL(sm1(x), wi, boneMtx[1](x));
					MUL(sm1(y), wi, boneMtx[1](y));
					MUL(sm1(z), wi, boneMtx[1](z));
					MUL(sm1(w), wi, boneMtx[1](w));
					ALU_LAST();
					MUL(sm2(x), wi, boneMtx[2](x));
					MUL(sm2(y), wi, boneMtx[2](y));
					MUL(sm2(z), wi, boneMtx[2](z));
					MUL(sm2(w), wi, boneMtx[2](w));
					ALU_LAST();
				} else {
					MULADD(sm0(x), wi, boneMtx[0](x), sm0(x));
					MULADD(sm0(y), wi, boneMtx[0](y), sm0(y));
					MULADD(sm0(z), wi, boneMtx[0](z), sm0(z));
					MULADD(sm0(w), wi, boneMtx[0](w), sm0(w));
					ALU_LAST();
					MULADD(sm1(x), wi, boneMtx[1](x), sm1(x));
					MULADD(sm1(y), wi, boneMtx[1](y), sm1(y));
					MULADD(sm1(z), wi, boneMtx[1](z), sm1(z));
					MULADD(sm1(w), wi, boneMtx[1](w), sm1(w));
					ALU_LAST();
					MULADD(sm2(x), wi, boneMtx[2](x), sm2(x));
					MULADD(sm2(y), wi, boneMtx[2](y), sm2(y));
					MULADD(sm2(z), wi, boneMtx[2](z), sm2(z));
					MULADD(sm2(w), wi, boneMtx[2](w), sm2(w));
					ALU_LAST();
				}
			}

			FLUSH_ALU();

			Reg skinnedPos = allocReg();
			Reg skinnedNrm = allocReg();

			emitDot4(skinnedPos(x), pos, C(1.0f), sm0, sm0(w));
			emitDot4(skinnedPos(y), pos, C(1.0f), sm1, sm1(w));
			emitDot4(skinnedPos(z), pos, C(1.0f), sm2, sm2(w));
			if (weightFactor != 1.0f) {
				MUL(skinnedPos(x), skinnedPos(x), C(weightFactor));
				MUL(skinnedPos(y), skinnedPos(y), C(weightFactor));
				MUL(skinnedPos(z), skinnedPos(z), C(weightFactor));
				ALU_LAST();
			}

			if (hasNormal) {
				if (flipNormal) {
					MUL(skinnedNrm(x), nrm(x), C(-weightFactor));
					MUL(skinnedNrm(y), nrm(y), C(-weightFactor));
					MUL(skinnedNrm(z), nrm(z), C(-weightFactor));
				} else if (weightFactor != 1.0f) {
					MUL(skinnedNrm(x), nrm(x), C(weightFactor));
					MUL(skinnedNrm(y), nrm(y), C(weightFactor));
					MUL(skinnedNrm(z), nrm(z), C(weightFactor));
				} else {
					MOV(skinnedNrm(x), nrm(x));
					MOV(skinnedNrm(y), nrm(y));
					MOV(skinnedNrm(z), nrm(z));
				}
				ALU_LAST();
				emitDot3ToPV(skinnedNrm, sm0); MOV(worldNormal(x), PV(x)); ALU_LAST();
				emitDot3ToPV(skinnedNrm, sm1); MOV(worldNormal(y), PV(x)); ALU_LAST();
				emitDot3ToPV(skinnedNrm, sm2); MOV(worldNormal(z), PV(x)); ALU_LAST();
			} else {
				float zSign = flipNormal ? -weightFactor : weightFactor;
				MUL(worldNormal(x), C(zSign), sm0(z));
				MUL(worldNormal(y), C(zSign), sm1(z));
				MUL(worldNormal(z), C(zSign), sm2(z));
				ALU_LAST();
			}

			FLUSH_ALU();
			KCacheRegs worldMtx(UB_Bindings::Base, offsetof(UB_VS_FS_Base, world), this);
			emitMulMat3x4Vec4(worldPos, skinnedPos, C(1.0f), worldMtx);
			emitMulMat3x4Vec4(tmp0, worldNormal, C(0.0f), worldMtx);
			emitNormalize3Safe(worldNormal, tmp0, tmp2);
		}

		// Step 2: View & Projection transform (+ optional depthRoundZVP & fogdepth).
		// Lock Base addr=0 (proj/view) before Base addr=1 (depthRange/fogCoef).
		FLUSH_ALU();
		KCacheRegs viewMtx(UB_Bindings::Base, offsetof(UB_VS_FS_Base, view), this);
		emitMulMat3x4Vec4(viewPos, worldPos, C(1.0f), viewMtx);

		KCacheRegs projMtx(UB_Bindings::Base, offsetof(UB_VS_FS_Base, proj), this);
		emitMulMat4Vec4(outPos, viewPos, C(1.0f), projMtx);

		if (gstate_c.Supports(GPU_ROUND_DEPTH_TO_16BIT)) {
			emitDepthRoundZVP(outPos);
		}

		if (enableFog) {
			KCacheReg fogCoef(UB_Bindings::Base, offsetof(UB_VS_FS_Base, fogCoef), this);
			ADD(___(x), viewPos(z), fogCoef(x));
			ALU_LAST();
			MUL(outFog(x), PV(x), fogCoef(y));
			ALU_LAST();
		}

		// Step 3: Lighting.
		if (enableLighting) {
			bool specularIsZero = true;
			for (int i = 0; i < 4; i++) {
				if (doLight[i] == LIGHT_FULL && static_cast<GELightComputation>(id.Bits(VS_BIT_LIGHT0_COMP + 4 * i, 2)) == GE_LIGHTCOMP_BOTH) {
					specularIsZero = false;
				}
			}

			Reg matAmbReg = ((matUpdate & 1) && hasColor) ? col0 : allocReg();
			if (!((matUpdate & 1) && hasColor)) {
				FLUSH_ALU();
				KCacheReg baseMatAmb(UB_Bindings::Base, offsetof(UB_VS_FS_Base, matAmbient), this);
				MOV(matAmbReg(x), baseMatAmb(x));
				MOV(matAmbReg(y), baseMatAmb(y));
				MOV(matAmbReg(z), baseMatAmb(z));
				MOV(matAmbReg(w), baseMatAmb(w));
				ALU_LAST();
			}

			FLUSH_ALU();
			KCacheReg uAmbient(UB_Bindings::Lights, offsetof(UB_VS_Lights, ambientColor), this);
			KCacheReg matEmissive(UB_Bindings::Lights, offsetof(UB_VS_Lights, materialEmissive), this);
			KCacheReg matDiffuseK(UB_Bindings::Lights, offsetof(UB_VS_Lights, materialDiffuse), this);
			KCacheReg matSpecK(UB_Bindings::Lights, offsetof(UB_VS_Lights, materialSpecular), this);

			Reg lightSum0 = allocReg();
			Reg lightSum1 = (!specularIsZero) ? allocReg() : Reg(0);

			// lightSum0 = u_ambient * ambientMat + vec4(matemissive, 0.0)
			MULADD(lightSum0(x), uAmbient(x), matAmbReg(x), matEmissive(x));
			MULADD(lightSum0(y), uAmbient(y), matAmbReg(y), matEmissive(y));
			MULADD(lightSum0(z), uAmbient(z), matAmbReg(z), matEmissive(z));
			MUL(lightSum0(w), uAmbient(w), matAmbReg(w));
			ALU_LAST();

			if (!specularIsZero) {
				MOV(lightSum1(x), C(0.0f));
				MOV(lightSum1(y), C(0.0f));
				MOV(lightSum1(z), C(0.0f));
				ALU_LAST();
			}

			Reg diffuseMatReg = ((matUpdate & 2) && hasColor) ? col0 : tmp2;
			if (!((matUpdate & 2) && hasColor)) {
				MOV(diffuseMatReg(x), matDiffuseK(x));
				MOV(diffuseMatReg(y), matDiffuseK(y));
				MOV(diffuseMatReg(z), matDiffuseK(z));
				ALU_LAST();
			}

			Reg specMatReg = (!specularIsZero && ((matUpdate & 4) && hasColor)) ? col0 : (!specularIsZero ? allocReg() : Reg(0));
			if (!specularIsZero && !((matUpdate & 4) && hasColor)) {
				MOV(specMatReg(x), matSpecK(x));
				MOV(specMatReg(y), matSpecK(y));
				MOV(specMatReg(z), matSpecK(z));
				ALU_LAST();
			}

			Reg specExpReg = allocReg();
			MOV(specExpReg(x), matSpecK(w)); // light.matspecular.a
			ALU_LAST();

			Reg toLight = tmp0;
			Reg diffuseTmp = tmp1;
			Reg scalarReg = allocReg(); // .x = distance, .y = lightScale, .z = dotOrig, .w = temp

			bool hasSpotLight = false;
			for (int i = 0; i < 4; i++) {
				if (doLight[i] == LIGHT_FULL) {
					GELightType t = static_cast<GELightType>(id.Bits(VS_BIT_LIGHT0_TYPE + 4 * i, 2));
					if (t == GE_LIGHTTYPE_SPOT || t == GE_LIGHTTYPE_UNKNOWN)
						hasSpotLight = true;
				}
			}
			Reg dirTmp = hasSpotLight ? allocReg() : Reg(0);
			Reg spotTmp = hasSpotLight ? allocReg() : Reg(0);
			Reg specTmp = (!specularIsZero) ? allocReg() : Reg(0);

			for (int i = 0; i < 4; i++) {
				if (doLight[i] != LIGHT_FULL)
					continue;

				GELightType type = static_cast<GELightType>(id.Bits(VS_BIT_LIGHT0_TYPE + 4 * i, 2));
				GELightComputation comp = static_cast<GELightComputation>(id.Bits(VS_BIT_LIGHT0_COMP + 4 * i, 2));
				bool doSpecular = comp == GE_LIGHTCOMP_BOTH;
				bool poweredDiffuse = comp == GE_LIGHTCOMP_ONLYPOWDIFFUSE;

				// Start each light in a clean clause locking addr=0 first, then addr=1.
				FLUSH_ALU();
				KCacheReg lpos(UB_Bindings::Lights, offsetof(UB_VS_Lights, lpos) + i * 16, this);
				KCacheReg ldir(UB_Bindings::Lights, offsetof(UB_VS_Lights, ldir) + i * 16, this);
				KCacheReg latt(UB_Bindings::Lights, offsetof(UB_VS_Lights, latt) + i * 16, this);
				KCacheReg angleSpot(UB_Bindings::Lights, offsetof(UB_VS_Lights, lightAngle_SpotCoef) + i * 16, this);
				KCacheReg lAmbient(UB_Bindings::Lights, offsetof(UB_VS_Lights, lightAmbient) + i * 16, this);
				KCacheReg lDiffuse(UB_Bindings::Lights, offsetof(UB_VS_Lights, lightDiffuse) + i * 16, this);
				KCacheReg lSpecular(UB_Bindings::Lights, offsetof(UB_VS_Lights, lightSpecular) + i * 16, this);

				if (type == GE_LIGHTTYPE_DIRECTIONAL) {
					MOV(toLight(x), lpos(x));
					MOV(toLight(y), lpos(y));
					MOV(toLight(z), lpos(z));
					ALU_LAST();
				} else {
					MULADD(toLight(x), worldPos(x), C(-1.0f), lpos(x));
					MULADD(toLight(y), worldPos(y), C(-1.0f), lpos(y));
					MULADD(toLight(z), worldPos(z), C(-1.0f), lpos(z));
					ALU_LAST();

					emitDot3ToPV(toLight, toLight);
					MOV(scalarReg(w), PV(x)); // distSq
					ALU_LAST();

					SQRT_IEEE(___(x), scalarReg(w));
					ALU_LAST();
					MOV(scalarReg(x), PS); // distance
					ALU_LAST();

					RECIPSQRT_IEEE(___(x), scalarReg(w));
					ALU_LAST();
					MUL(___(x), toLight(x), PS);
					MUL(___(y), toLight(y), PS);
					MUL(___(z), toLight(z), PS);
					ALU_LAST();

					CNDGT(toLight(x), scalarReg(w), PV(x), C(0.0f));
					CNDGT(toLight(y), scalarReg(w), PV(y), C(0.0f));
					CNDGT(toLight(z), scalarReg(w), PV(z), C(0.0f));
					ALU_LAST();
				}

				// dotOrig = dot(toLight, worldnormal)
				emitDot3ToPV(toLight, worldNormal);
				MOV(scalarReg(z), PV(x));
				ALU_LAST();

				// Diffuse term factor in diffuseTmp(w)
				if (poweredDiffuse) {
					MAX(diffuseTmp(w), scalarReg(z), C(0.0f));
					ALU_LAST();
					emitPowSafe(diffuseTmp, w, diffuseTmp(w), specExpReg(x));
				} else {
					MAX(diffuseTmp(w), scalarReg(z), C(0.0f));
					ALU_LAST();
				}

				// Attenuation -> scalarReg(y) (uses addr=0 latt/ldir before addr=1 angleSpot/lDiffuse/lAmbient/lSpecular)
				if (type != GE_LIGHTTYPE_DIRECTIONAL) {
					MUL(scalarReg(w), scalarReg(x), scalarReg(x)); // dist^2
					ALU_LAST();

					MULADD(___(x), latt(y), scalarReg(x), latt(x));
					ALU_LAST();
					MULADD(___(x), latt(z), scalarReg(w), PV(x));
					ALU_LAST();

					RECIP_IEEE(___(x), PV(x));
					ALU_LAST();
					MAX(___(x), PS, C(0.0f));
					ALU_LAST();
					MIN(scalarReg(y), PV(x), C(1.0f));
					ALU_LAST();

					if (type == GE_LIGHTTYPE_SPOT || type == GE_LIGHTTYPE_UNKNOWN) {
						MOV(dirTmp(x), ldir(x));
						MOV(dirTmp(y), ldir(y));
						MOV(dirTmp(z), ldir(z));
						ALU_LAST();
						emitNormalize3Safe(dirTmp, dirTmp, spotTmp);

						// spotDot = (len(dir) == 0.0) ? 0.0 : dot(normalize(dir), toLight)
						// (Since dirTmp == (0,0,0) when len == 0, dot(dirTmp, toLight) is already 0.0.)
						emitDot3ToPV(dirTmp, toLight);
						MOV(spotTmp(x), PV(x)); // angle
						ALU_LAST();

						// pow(max(angle, 0.0), angleSpot.y) or 1.0 if angleSpot.y <= 0.0
						MAX(spotTmp(y), spotTmp(x), C(0.0f));
						ALU_LAST();
						emitPowSafe(spotTmp, z, spotTmp(y), angleSpot(y));

						// angle >= angleSpot.x  <=>  (angle - angleSpot.x) >= 0.0
						MULADD(___(x), angleSpot(x), C(-1.0f), spotTmp(x));
						ALU_LAST();
						CNDGE(___(x), PV(x), spotTmp(z), C(0.0f));
						ALU_LAST();
						MUL(scalarReg(y), scalarReg(y), PV(x));
						ALU_LAST();
					}
				}

				// diffuse = (light.diffuse[i] * diffuseMat) * dotFactor
				MUL(___(x), lDiffuse(x), diffuseMatReg(x));
				MUL(___(y), lDiffuse(y), diffuseMatReg(y));
				MUL(___(z), lDiffuse(z), diffuseMatReg(z));
				ALU_LAST();
				MUL(diffuseTmp(x), PV(x), diffuseTmp(w));
				MUL(diffuseTmp(y), PV(y), diffuseTmp(w));
				MUL(diffuseTmp(z), PV(z), diffuseTmp(w));
				ALU_LAST();

				// lightSum0.rgb += (light.ambient[i] * ambientMat.rgb + diffuse) * lightScale
				MULADD(diffuseTmp(x), lAmbient(x), matAmbReg(x), diffuseTmp(x));
				MULADD(diffuseTmp(y), lAmbient(y), matAmbReg(y), diffuseTmp(y));
				MULADD(diffuseTmp(z), lAmbient(z), matAmbReg(z), diffuseTmp(z));
				ALU_LAST();

				if (type == GE_LIGHTTYPE_DIRECTIONAL) {
					ADD(lightSum0(x), lightSum0(x), diffuseTmp(x));
					ADD(lightSum0(y), lightSum0(y), diffuseTmp(y));
					ADD(lightSum0(z), lightSum0(z), diffuseTmp(z));
				} else {
					MULADD(lightSum0(x), diffuseTmp(x), scalarReg(y), lightSum0(x));
					MULADD(lightSum0(y), diffuseTmp(y), scalarReg(y), lightSum0(y));
					MULADD(lightSum0(z), diffuseTmp(z), scalarReg(y), lightSum0(z));
				}
				ALU_LAST();

				if (doSpecular) {
					FLUSH_ALU();
					// halfVec = normalize(toLight + vec3(0, 0, 1))
					MOV(diffuseTmp(x), toLight(x));
					MOV(diffuseTmp(y), toLight(y));
					ADD(diffuseTmp(z), toLight(z), C(1.0f));
					ALU_LAST();
					emitNormalize3Safe(diffuseTmp, diffuseTmp, specTmp);

					// specDot = max(dot(halfVec, worldNormal), 0.0)
					emitDot3ToPV(diffuseTmp, worldNormal);
					MAX(specTmp(x), PV(x), C(0.0f));
					ALU_LAST();

					emitPowSafe(specTmp, x, specTmp(x), specExpReg(x));

					// Only contribute when dotOrig (scalarReg.z) >= 0.0
					CNDGE(specTmp(x), scalarReg(z), specTmp(x), C(0.0f));
					ALU_LAST();

					if (type != GE_LIGHTTYPE_DIRECTIONAL) {
						MUL(specTmp(x), specTmp(x), scalarReg(y));
						ALU_LAST();
					}

					MUL(___(x), lSpecular(x), specMatReg(x));
					MUL(___(y), lSpecular(y), specMatReg(y));
					MUL(___(z), lSpecular(z), specMatReg(z));
					ALU_LAST();
					MULADD(lightSum1(x), PV(x), specTmp(x), lightSum1(x));
					MULADD(lightSum1(y), PV(y), specTmp(x), lightSum1(y));
					MULADD(lightSum1(z), PV(z), specTmp(x), lightSum1(z));
					ALU_LAST();
				}
			}

			// Clamp light sums to [0.0, 1.0].
			MAX(___(x), lightSum0(x), C(0.0f));
			MAX(___(y), lightSum0(y), C(0.0f));
			MAX(___(z), lightSum0(z), C(0.0f));
			MAX(___(w), lightSum0(w), C(0.0f));
			ALU_LAST();
			MIN(outCol0(x), PV(x), C(1.0f));
			MIN(outCol0(y), PV(y), C(1.0f));
			MIN(outCol0(z), PV(z), C(1.0f));
			MIN(outCol0(w), PV(w), C(1.0f));
			ALU_LAST();

			if (lmode) {
				if (specularIsZero) {
					MOV(outCol1(x), C(0.0f));
					MOV(outCol1(y), C(0.0f));
					MOV(outCol1(z), C(0.0f));
					ALU_LAST();
				} else {
					MAX(___(x), lightSum1(x), C(0.0f));
					MAX(___(y), lightSum1(y), C(0.0f));
					MAX(___(z), lightSum1(z), C(0.0f));
					ALU_LAST();
					MIN(outCol1(x), PV(x), C(1.0f));
					MIN(outCol1(y), PV(y), C(1.0f));
					MIN(outCol1(z), PV(z), C(1.0f));
					ALU_LAST();
				}
			} else if (!specularIsZero) {
				ADD(___(x), outCol0(x), lightSum1(x));
				ADD(___(y), outCol0(y), lightSum1(y));
				ADD(___(z), outCol0(z), lightSum1(z));
				ALU_LAST();
				MAX(___(x), PV(x), C(0.0f));
				MAX(___(y), PV(y), C(0.0f));
				MAX(___(z), PV(z), C(0.0f));
				ALU_LAST();
				MIN(outCol0(x), PV(x), C(1.0f));
				MIN(outCol0(y), PV(y), C(1.0f));
				MIN(outCol0(z), PV(z), C(1.0f));
				ALU_LAST();
			}
		} else {
			if (hasColor) {
				MOV(outCol0(x), col0(x));
				MOV(outCol0(y), col0(y));
				MOV(outCol0(z), col0(z));
				MOV(outCol0(w), col0(w));
				ALU_LAST();
			} else {
				FLUSH_ALU();
				KCacheReg matAmb(UB_Bindings::Base, offsetof(UB_VS_FS_Base, matAmbient), this);
				MOV(outCol0(x), matAmb(x));
				MOV(outCol0(y), matAmb(y));
				MOV(outCol0(z), matAmb(z));
				MOV(outCol0(w), matAmb(w));
				ALU_LAST();
			}
			if (lmode) {
				MOV(outCol1(x), C(0.0f));
				MOV(outCol1(y), C(0.0f));
				MOV(outCol1(z), C(0.0f));
				ALU_LAST();
			}
		}

		// Step 4: UV generation
		bool scaleUV = !isModeThrough && (uvGenMode == GE_TEXMAP_TEXTURE_COORDS || uvGenMode == GE_TEXMAP_UNKNOWN);
		if (doTexture) {
			switch (uvGenMode) {
			case GE_TEXMAP_TEXTURE_COORDS:
			case GE_TEXMAP_UNKNOWN:
				if (scaleUV) {
					if (hasTexcoord) {
						FLUSH_ALU();
						KCacheReg uvScaleOffset(UB_Bindings::Base, offsetof(UB_VS_FS_Base, uvScaleOffset), this);
						MULADD(outCoords(x), coord(x), uvScaleOffset(x), uvScaleOffset(z));
						MULADD(outCoords(y), coord(y), uvScaleOffset(y), uvScaleOffset(w));
						MOV(outCoords(z), C(1.0f));
					} else {
						MOV(outCoords(x), C(0.0f));
						MOV(outCoords(y), C(0.0f));
						MOV(outCoords(z), C(1.0f));
					}
					ALU_LAST();
				} else {
					FLUSH_ALU();
					KCacheReg uvScaleOffset(UB_Bindings::Base, offsetof(UB_VS_FS_Base, uvScaleOffset), this);
					if (hasTexcoord) {
						MULADD(outCoords(x), coord(x), uvScaleOffset(x), uvScaleOffset(z));
						MULADD(outCoords(y), coord(y), uvScaleOffset(y), uvScaleOffset(w));
					} else {
						MOV(outCoords(x), uvScaleOffset(z));
						MOV(outCoords(y), uvScaleOffset(w));
					}
					MOV(outCoords(z), C(1.0f));
					ALU_LAST();
				}
				break;

			case GE_TEXMAP_TEXTURE_MATRIX: {
				Reg texSrc = tmp0;
				switch (uvProjMode) {
				case GE_PROJMAP_POSITION:
					texSrc = pos;
					break;
				case GE_PROJMAP_UV:
					if (hasTexcoord) {
						MOV(tmp0(x), coord(x));
						MOV(tmp0(y), coord(y));
						MOV(tmp0(z), C(0.0f));
					} else {
						MOV(tmp0(x), C(0.0f));
						MOV(tmp0(y), C(0.0f));
						MOV(tmp0(z), C(0.0f));
					}
					ALU_LAST();
					break;
				case GE_PROJMAP_NORMALIZED_NORMAL:
					if (hasNormal) {
						if (flipNormal) {
							MUL(tmp0(x), nrm(x), C(-1.0f));
							MUL(tmp0(y), nrm(y), C(-1.0f));
							MUL(tmp0(z), nrm(z), C(-1.0f));
							ALU_LAST();
							emitNormalize3Safe(tmp0, tmp0, tmp2);
						} else {
							emitNormalize3Safe(tmp0, nrm, tmp2);
						}
					} else {
						MOV(tmp0(x), C(0.0f));
						MOV(tmp0(y), C(0.0f));
						MOV(tmp0(z), C(1.0f));
						ALU_LAST();
					}
					break;
				case GE_PROJMAP_NORMAL:
					if (hasNormal) {
						if (flipNormal) {
							MUL(tmp0(x), nrm(x), C(-1.0f));
							MUL(tmp0(y), nrm(y), C(-1.0f));
							MUL(tmp0(z), nrm(z), C(-1.0f));
							ALU_LAST();
						} else {
							texSrc = nrm;
						}
					} else {
						MOV(tmp0(x), C(0.0f));
						MOV(tmp0(y), C(0.0f));
						MOV(tmp0(z), C(1.0f));
						ALU_LAST();
					}
					break;
				}
				// Lock Base addr=0 (tex[0..1]) before Base addr=1 (tex[2], uvScaleOffset).
				FLUSH_ALU();
				KCacheRegs texMtx(UB_Bindings::Base, offsetof(UB_VS_FS_Base, tex), this);
				emitMulMat3x4Vec4(outCoords, texSrc, C(1.0f), texMtx);

				KCacheReg uvScaleOffset(UB_Bindings::Base, offsetof(UB_VS_FS_Base, uvScaleOffset), this);
				MUL(outCoords(x), outCoords(x), uvScaleOffset(x));
				MUL(outCoords(y), outCoords(y), uvScaleOffset(y));
				ALU_LAST();
				break;
			}

			case GE_TEXMAP_ENVIRONMENT_MAP: {
				FLUSH_ALU();
				KCacheReg lpos0(UB_Bindings::Lights, offsetof(UB_VS_Lights, lpos) + ls0 * 16, this);
				MOV(tmp0(x), lpos0(x));
				MOV(tmp0(y), lpos0(y));
				MOV(tmp0(z), lpos0(z));
				ALU_LAST();
				emitNormalize3Safe(tmp0, tmp0, tmp2);
				emitDot3ToPV(tmp0, worldNormal);
				CNDGT(tmp1(x), tmp2(x), PV(x), worldNormal(z));
				ALU_LAST();

				if (ls1 == ls0) {
					MOV(tmp1(y), tmp1(x));
					ALU_LAST();
				} else {
					KCacheReg lpos1(UB_Bindings::Lights, offsetof(UB_VS_Lights, lpos) + ls1 * 16, this);
					MOV(tmp0(x), lpos1(x));
					MOV(tmp0(y), lpos1(y));
					MOV(tmp0(z), lpos1(z));
					ALU_LAST();
					emitNormalize3Safe(tmp0, tmp0, tmp2);
					emitDot3ToPV(tmp0, worldNormal);
					CNDGT(tmp1(y), tmp2(x), PV(x), worldNormal(z));
					ALU_LAST();
				}

				FLUSH_ALU();
				KCacheReg uvScaleOffset(UB_Bindings::Base, offsetof(UB_VS_FS_Base, uvScaleOffset), this);
				ADD(___(x), C(1.0f), tmp1(x));
				ADD(___(y), C(1.0f), tmp1(y));
				ALU_LAST();
				MUL(___(x), PV(x), uvScaleOffset(x));
				MUL(___(y), PV(y), uvScaleOffset(y));
				ALU_LAST();
				MUL(outCoords(x), PV(x), C(0.5f));
				MUL(outCoords(y), PV(y), C(0.5f));
				MOV(outCoords(z), C(1.0f));
				ALU_LAST();
				break;
			}

			default:
				MOV(outCoords(x), C(0.0f));
				MOV(outCoords(y), C(0.0f));
				MOV(outCoords(z), C(1.0f));
				ALU_LAST();
				break;
			}
		}
	}

	EXP_POS(outPos);
	if (doTexture)
		EXP_PARAM(PSInput::COORDS, outCoords(x, y, z, __), NO_BARRIER);
	EXP_PARAM(PSInput::COLOR0, outCol0, NO_BARRIER);
	if (lmode)
		EXP_PARAM(PSInput::COLOR1, outCol1(x, y, z, _0_), NO_BARRIER);
	if (enableFog)
		EXP_PARAM(PSInput::FOGDEPTH, outFog(x, __, __, __), NO_BARRIER);

	END_OF_PROGRAM(vs);
}

void GenerateVertexShaderGX2(const VShaderID &id, GX2VertexShader *vs) {
	VertexShaderGeneratorGX2 vsGen;
	if (vsGen.Supported(id)) {
		vsGen.Emit(id, vs);
	} else {
		NOTICE_LOG(G3D, "unsupported VShaderID: \"%s\"", VertexShaderDesc(id).c_str());
		if (id.Bit(VS_BIT_USE_HW_TRANSFORM))
			*vs = id.Bit(VS_BIT_ENABLE_BONES) ? VShaderHWSkinGX2 : VShaderHWNoSkinGX2;
		else
			*vs = VShaderSWGX2;
	}
}
