// Copyright (c) 2015- PPSSPP Project.

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

#include "ppsspp_config.h"

#include <map>
#include <wiiu/gx2/shaders.h>
#include <wiiu/gx2/shader_disasm.hpp>
#include <wiiu/gx2/shader_info.h>

#include <wiiu/os/time.h>
#include "Common/Math/lin/matrix4x4.h"
#include "Common/Math/math_util.h"
#include "Common/Data/Convert/SmallDataConvert.h"
#include "Common/GPU/thin3d.h"
#include "Common/Data/Encoding/Utf8.h"
#include "Common/Common.h"
#include "Core/Config.h"
#include "Core/Reporting.h"
#include "GPU/Math3D.h"
#include "GPU/GPUState.h"
#include "GPU/ge_constants.h"
#include "GPU/GX2/ShaderManagerGX2.h"
#include "GPU/GX2/FragmentShaderGeneratorGX2.h"
#include "GPU/GX2/VertexShaderGeneratorGX2.h"
#include "GPU/GX2/GX2Util.h"
#include "GPU/Vulkan/FragmentShaderGeneratorVulkan.h"
#include "GPU/Vulkan/VertexShaderGeneratorVulkan.h"

GX2PerfCounters g_gx2Perf = {};

GX2PShader::GX2PShader(FShaderID id) : GX2PixelShader(), id_(id) {
	GenerateFragmentShaderGX2(id, this);
	ub_id = (UB_FSID *)MEM2_alloc(sizeof(UB_FSID), GX2_UNIFORM_BLOCK_ALIGNMENT);
	memset(ub_id, 0, sizeof(UB_FSID));
	ub_id->FS_BIT_CLEARMODE = id.Bit(FS_BIT_CLEARMODE);
	ub_id->FS_BIT_DO_TEXTURE = id.Bit(FS_BIT_DO_TEXTURE);
	ub_id->FS_BIT_TEXFUNC = id.Bits(FS_BIT_TEXFUNC, 3);
	ub_id->FS_BIT_TEXALPHA = id.Bit(FS_BIT_TEXALPHA);
	ub_id->FS_BIT_SHADER_DEPAL = id.Bit(FS_BIT_SHADER_DEPAL);
	ub_id->FS_BIT_SHADER_TEX_CLAMP = id.Bit(FS_BIT_SHADER_TEX_CLAMP);
	ub_id->FS_BIT_CLAMP_S = id.Bit(FS_BIT_CLAMP_S);
	ub_id->FS_BIT_CLAMP_T = id.Bit(FS_BIT_CLAMP_T);
	ub_id->FS_BIT_TEXTURE_AT_OFFSET = id.Bit(FS_BIT_TEXTURE_AT_OFFSET);
	ub_id->FS_BIT_LMODE = id.Bit(FS_BIT_LMODE);
	ub_id->FS_BIT_ALPHA_TEST = id.Bit(FS_BIT_ALPHA_TEST);
	ub_id->FS_BIT_ALPHA_TEST_FUNC = id.Bits(FS_BIT_ALPHA_TEST_FUNC, 3);
	ub_id->FS_BIT_ALPHA_AGAINST_ZERO = id.Bit(FS_BIT_ALPHA_AGAINST_ZERO);
	ub_id->FS_BIT_COLOR_TEST = id.Bit(FS_BIT_COLOR_TEST);
	ub_id->FS_BIT_COLOR_TEST_FUNC = id.Bits(FS_BIT_COLOR_TEST_FUNC, 2);
	ub_id->FS_BIT_COLOR_AGAINST_ZERO = id.Bit(FS_BIT_COLOR_AGAINST_ZERO);
	ub_id->FS_BIT_ENABLE_FOG = id.Bit(FS_BIT_ENABLE_FOG);
	ub_id->FS_BIT_DO_TEXTURE_PROJ = id.Bit(FS_BIT_DO_TEXTURE_PROJ);
	ub_id->FS_BIT_COLOR_DOUBLE = id.Bit(FS_BIT_COLOR_DOUBLE);
	ub_id->FS_BIT_STENCIL_TO_ALPHA = id.Bits(FS_BIT_STENCIL_TO_ALPHA, 2);
	ub_id->FS_BIT_REPLACE_ALPHA_WITH_STENCIL_TYPE = id.Bits(FS_BIT_REPLACE_ALPHA_WITH_STENCIL_TYPE, 4);
	ub_id->FS_BIT_REPLACE_LOGIC_OP_TYPE = id.Bits(FS_BIT_REPLACE_LOGIC_OP_TYPE, 2);
	ub_id->FS_BIT_REPLACE_BLEND = id.Bits(FS_BIT_REPLACE_BLEND, 3);
	ub_id->FS_BIT_BLENDEQ = id.Bits(FS_BIT_BLENDEQ, 3);
	ub_id->FS_BIT_BLENDFUNC_A = id.Bits(FS_BIT_BLENDFUNC_A, 4);
	ub_id->FS_BIT_BLENDFUNC_B = id.Bits(FS_BIT_BLENDFUNC_B, 4);
	ub_id->FS_BIT_FLATSHADE = id.Bit(FS_BIT_FLATSHADE);
	ub_id->FS_BIT_BGRA_TEXTURE = id.Bit(FS_BIT_BGRA_TEXTURE);
	ub_id->GPU_SCALE_DEPTH_FROM_24BIT_TO_16BIT = gstate_c.Supports(GPU_SCALE_DEPTH_FROM_24BIT_TO_16BIT);
	ub_id->GPU_SUPPORTS_DEPTH_CLAMP = gstate_c.Supports(GPU_SUPPORTS_DEPTH_CLAMP);
	ub_id->GPU_ROUND_FRAGMENT_DEPTH_TO_16BIT = gstate_c.Supports(GPU_ROUND_FRAGMENT_DEPTH_TO_16BIT);
	ub_id->GPU_SUPPORTS_ACCURATE_DEPTH = gstate_c.Supports(GPU_SUPPORTS_ACCURATE_DEPTH);
	GX2Invalidate(GX2_INVALIDATE_MODE_CPU_UNIFORM_BLOCK, ub_id, sizeof(UB_FSID));
}

std::string GX2PShader::GetShaderString(DebugShaderStringType type) const {
	switch (type) {
	case SHADER_STRING_SHORT_DESC: return FragmentShaderDesc(id_);
	case SHADER_STRING_SOURCE_CODE: {
		char buffer[0x20000];
		DisassembleGX2Shader(program, size, buffer);
		sprintf(buffer + strlen(buffer), "\n### GPU Regs ###\n");
		GX2PixelShaderInfo(this, buffer + strlen(buffer));
		sprintf(buffer + strlen(buffer), "\n### glsl ###\n");
		GenerateVulkanGLSLFragmentShader(id_, buffer + strlen(buffer), 0);
		_assert_(strlen(buffer) < sizeof(buffer));
		return buffer;
	}
	default: return "N/A";
	}
}

GX2VShader::GX2VShader(VShaderID id) : GX2VertexShader(), id_(id) {
	GenerateVertexShaderGX2(id, this);
	ub_id = (UB_VSID *)MEM2_alloc(sizeof(UB_VSID), GX2_UNIFORM_BLOCK_ALIGNMENT);
	memset(ub_id, 0, sizeof(UB_VSID));
	ub_id->VS_BIT_LMODE = id.Bit(VS_BIT_LMODE);
	ub_id->VS_BIT_IS_THROUGH = id.Bit(VS_BIT_IS_THROUGH);
	ub_id->VS_BIT_ENABLE_FOG = id.Bit(VS_BIT_ENABLE_FOG);
	ub_id->VS_BIT_HAS_COLOR = id.Bit(VS_BIT_HAS_COLOR);
	ub_id->VS_BIT_DO_TEXTURE = id.Bit(VS_BIT_DO_TEXTURE);
	ub_id->VS_BIT_DO_TEXTURE_TRANSFORM = id.Bit(VS_BIT_DO_TEXTURE_TRANSFORM);
	ub_id->VS_BIT_USE_HW_TRANSFORM = id.Bit(VS_BIT_USE_HW_TRANSFORM);
	ub_id->VS_BIT_HAS_NORMAL = id.Bit(VS_BIT_HAS_NORMAL);
	ub_id->VS_BIT_NORM_REVERSE = id.Bit(VS_BIT_NORM_REVERSE);
	ub_id->VS_BIT_HAS_TEXCOORD = id.Bit(VS_BIT_HAS_TEXCOORD);
	ub_id->VS_BIT_HAS_COLOR_TESS = id.Bit(VS_BIT_HAS_COLOR_TESS);
	ub_id->VS_BIT_HAS_TEXCOORD_TESS = id.Bit(VS_BIT_HAS_TEXCOORD_TESS);
	ub_id->VS_BIT_NORM_REVERSE_TESS = id.Bit(VS_BIT_NORM_REVERSE_TESS);
	ub_id->VS_BIT_UVGEN_MODE = id.Bits(VS_BIT_UVGEN_MODE, 2);
	ub_id->VS_BIT_UVPROJ_MODE = id.Bits(VS_BIT_UVPROJ_MODE, 2);
	ub_id->VS_BIT_LS0 = id.Bits(VS_BIT_LS0, 2);
	ub_id->VS_BIT_LS1 = id.Bits(VS_BIT_LS1, 2);
	ub_id->VS_BIT_BONES = id.Bits(VS_BIT_BONES, 3);
	ub_id->VS_BIT_ENABLE_BONES = id.Bit(VS_BIT_ENABLE_BONES);
	ub_id->VS_BIT_LIGHT[0].COMP = id.Bits(VS_BIT_LIGHT0_COMP, 2);
	ub_id->VS_BIT_LIGHT[0].TYPE = id.Bits(VS_BIT_LIGHT0_TYPE, 2);
	ub_id->VS_BIT_LIGHT[1].COMP = id.Bits(VS_BIT_LIGHT1_COMP, 2);
	ub_id->VS_BIT_LIGHT[1].TYPE = id.Bits(VS_BIT_LIGHT1_TYPE, 2);
	ub_id->VS_BIT_LIGHT[2].COMP = id.Bits(VS_BIT_LIGHT2_COMP, 2);
	ub_id->VS_BIT_LIGHT[2].TYPE = id.Bits(VS_BIT_LIGHT2_TYPE, 2);
	ub_id->VS_BIT_LIGHT[3].COMP = id.Bits(VS_BIT_LIGHT3_COMP, 2);
	ub_id->VS_BIT_LIGHT[3].TYPE = id.Bits(VS_BIT_LIGHT3_TYPE, 2);
	ub_id->VS_BIT_MATERIAL_UPDATE = id.Bits(VS_BIT_MATERIAL_UPDATE, 3);
	ub_id->VS_BIT_SPLINE = id.Bit(VS_BIT_SPLINE);
	ub_id->VS_BIT_LIGHT[0].ENABLE = id.Bit(VS_BIT_LIGHT0_ENABLE);
	ub_id->VS_BIT_LIGHT[1].ENABLE = id.Bit(VS_BIT_LIGHT1_ENABLE);
	ub_id->VS_BIT_LIGHT[2].ENABLE = id.Bit(VS_BIT_LIGHT2_ENABLE);
	ub_id->VS_BIT_LIGHT[3].ENABLE = id.Bit(VS_BIT_LIGHT3_ENABLE);
	ub_id->VS_BIT_LIGHTING_ENABLE = id.Bit(VS_BIT_LIGHTING_ENABLE);
	ub_id->VS_BIT_WEIGHT_FMTSCALE = id.Bits(VS_BIT_WEIGHT_FMTSCALE, 2);
	ub_id->VS_BIT_FLATSHADE = id.Bit(VS_BIT_FLATSHADE);
	ub_id->VS_BIT_BEZIER = id.Bit(VS_BIT_BEZIER);
	ub_id->GPU_ROUND_DEPTH_TO_16BIT = gstate_c.Supports(GPU_ROUND_DEPTH_TO_16BIT);
	GX2Invalidate(GX2_INVALIDATE_MODE_CPU_UNIFORM_BLOCK, ub_id, sizeof(UB_VSID));
}

std::string GX2VShader::GetShaderString(DebugShaderStringType type) const {
	switch (type) {
	case SHADER_STRING_SHORT_DESC: return VertexShaderDesc(id_);
	case SHADER_STRING_SOURCE_CODE:
	{
		char buffer[0x20000];
		DisassembleGX2Shader(program, size, buffer);
		sprintf(buffer + strlen(buffer), "\n### GPU Regs ###\n");
		GX2VertexShaderInfo(this, buffer + strlen(buffer));
		sprintf(buffer + strlen(buffer), "\n### glsl ###\n");
		GenerateVulkanGLSLVertexShader(id_, buffer + strlen(buffer));
		_assert_(strlen(buffer) < sizeof(buffer));
		return buffer;
	}
	default: return "N/A";
	}
}

ShaderManagerGX2::ShaderManagerGX2(Draw::DrawContext *draw, GX2ContextState *context) :
	ShaderManagerCommon(draw), lastVShader_(nullptr), lastFShader_(nullptr) {
	memset(&ub_base, 0, sizeof(ub_base));
	memset(&ub_lights, 0, sizeof(ub_lights));
	memset(&ub_bones, 0, sizeof(ub_bones));

	INFO_LOG(G3D, "sizeof(ub_base): %d", (int)sizeof(ub_base));
	INFO_LOG(G3D, "sizeof(ub_lights): %d", (int)sizeof(ub_lights));
	INFO_LOG(G3D, "sizeof(ub_bones): %d", (int)sizeof(ub_bones));
}

ShaderManagerGX2::~ShaderManagerGX2() {
	ClearShaders();
}

void ShaderManagerGX2::Clear() {
	for (auto iter = fsCache_.begin(); iter != fsCache_.end(); ++iter) {
		delete iter->second;
	}
	for (auto iter = vsCache_.begin(); iter != vsCache_.end(); ++iter) {
		delete iter->second;
	}
	fsCache_.clear();
	vsCache_.clear();
	memset(vsFast_, 0, sizeof(vsFast_));
	memset(fsFast_, 0, sizeof(fsFast_));
	lastFSID_.set_invalid();
	lastVSID_.set_invalid();
	gstate_c.Dirty(DIRTY_VERTEXSHADER_STATE | DIRTY_FRAGMENTSHADER_STATE);
}

void ShaderManagerGX2::ClearShaders() {
	Clear();
	DirtyLastShader();
	gstate_c.Dirty(DIRTY_ALL_UNIFORMS);
}

void ShaderManagerGX2::DirtyLastShader() {
	lastFSID_.set_invalid();
	lastVSID_.set_invalid();
	lastVShader_ = nullptr;
	lastFShader_ = nullptr;
	dirtyEpoch_++;
	psBaseBound_ = false;
	gstate_c.Dirty(DIRTY_VERTEXSHADER_STATE | DIRTY_FRAGMENTSHADER_STATE | DIRTY_ALL_UNIFORMS);
}

static inline void CopySwapU32(u32 * __restrict dst, const u32 * __restrict src, size_t count) {
#if defined(__wiiu__) || defined(__PPC__)
	const uintptr_t o4 = 4, o8 = 8, o12 = 12, o16 = 16, o20 = 20, o24 = 24, o28 = 28;
	size_t i = 0;
	if (((uintptr_t)dst & 31) == 0) {
		for (; i + 8 <= count; i += 8, src += 8, dst += 8) {
			// Allocate cold 32-byte ring-buffer cache line directly in L1 (avoids MEM2 Read-For-Ownership).
			asm volatile("dcbz 0, %0" : : "b"(dst) : "memory");
			u32 a = src[0], b = src[1], c = src[2], d = src[3];
			u32 e = src[4], f = src[5], g = src[6], h = src[7];
			asm volatile(
				"stwbrx %0, 0, %8\n\t"
				"stwbrx %1, %8, %9\n\t"
				"stwbrx %2, %8, %10\n\t"
				"stwbrx %3, %8, %11\n\t"
				"stwbrx %4, %8, %12\n\t"
				"stwbrx %5, %8, %13\n\t"
				"stwbrx %6, %8, %14\n\t"
				"stwbrx %7, %8, %15\n\t"
				:
				: "r"(a), "r"(b), "r"(c), "r"(d), "r"(e), "r"(f), "r"(g), "r"(h),
				  "b"(dst), "r"(o4), "r"(o8), "r"(o12), "r"(o16), "r"(o20), "r"(o24), "r"(o28)
				: "memory");
		}
	}
	for (; i + 4 <= count; i += 4, src += 4, dst += 4) {
		u32 a = src[0];
		u32 b = src[1];
		u32 c = src[2];
		u32 d = src[3];
		asm volatile(
			"stwbrx %0, 0, %4\n\t"
			"stwbrx %1, %4, %5\n\t"
			"stwbrx %2, %4, %6\n\t"
			"stwbrx %3, %4, %7\n\t"
			:
			: "r"(a), "r"(b), "r"(c), "r"(d), "b"(dst), "r"(o4), "r"(o8), "r"(o12)
			: "memory");
	}
	for (; i < count; i++, src++, dst++) {
		u32 a = *src;
		asm volatile("stwbrx %0, 0, %1" : : "r"(a), "r"(dst) : "memory");
	}
#else
	size_t i = 0;
	for (; i + 4 <= count; i += 4) {
		u32 a = __builtin_bswap32(src[i + 0]);
		u32 b = __builtin_bswap32(src[i + 1]);
		u32 c = __builtin_bswap32(src[i + 2]);
		u32 d = __builtin_bswap32(src[i + 3]);
		dst[i + 0] = a;
		dst[i + 1] = b;
		dst[i + 2] = c;
		dst[i + 3] = d;
	}
	for (; i < count; i++) {
		dst[i] = __builtin_bswap32(src[i]);
	}
#endif
}

uint64_t ShaderManagerGX2::UpdateUniforms(PushBufferGX2 *push, bool useBufferedRendering) {
	uint64_t dirty = gstate_c.GetDirtyUniforms();
	if (dirty != 0) {
		u64 uboT0 = OSGetSystemTime();
		const bool useLights = !lastVShader_ || lastVShader_->ID().Bit(VS_BIT_LIGHTING_ENABLE) ||
			((GETexMapMode)lastVShader_->ID().Bits(VS_BIT_UVGEN_MODE, 2) == GE_TEXMAP_ENVIRONMENT_MAP);
		const bool useBones = !lastVShader_ || lastVShader_->ID().Bit(VS_BIT_ENABLE_BONES);
		uint64_t deferred = 0;

		u32 *push_base = nullptr;
		u32 *push_lights = nullptr;
		u32 *push_bones = nullptr;
		if (dirty & DIRTY_BASE_UNIFORMS) {
			BaseUpdateUniforms(&ub_base, dirty, true, useBufferedRendering);
			push_base = (u32 *)push->BeginPush(nullptr, sizeof(ub_base));
			CopySwapU32(push_base, (const u32 *)&ub_base, sizeof(ub_base) / 4);
		}
		if (dirty & DIRTY_LIGHT_UNIFORMS) {
			if (useLights) {
				LightUpdateUniforms(&ub_lights, dirty);
				push_lights = (u32 *)push->BeginPush(nullptr, sizeof(ub_lights));
				CopySwapU32(push_lights, (const u32 *)&ub_lights, sizeof(ub_lights) / 4);
			} else {
				deferred |= (dirty & DIRTY_LIGHT_UNIFORMS);
			}
		}
		if (dirty & DIRTY_BONE_UNIFORMS) {
			if (useBones) {
				BoneUpdateUniforms(&ub_bones, dirty);
				push_bones = (u32 *)push->BeginPush(nullptr, sizeof(ub_bones));
				// Always copy all 8 bones: a later draw may use more bones without re-uploading
				// bone matrices (e.g. GTA head / Tekken limbs sharing a skeleton).
				CopySwapU32(push_bones, (const u32 *)&ub_bones, sizeof(ub_bones) / 4);
			} else {
				deferred |= (dirty & DIRTY_BONE_UNIFORMS);
			}
		}
		if (push_base || push_lights || push_bones) {
			// Single DCStoreRange for all updated UBOs in this draw.
			push->EndPush();
			if (push_base) {
				GX2SetVertexUniformBlock(1, sizeof(ub_base), push_base);
				static const uint64_t DIRTY_FS_BASE =
					DIRTY_ALPHACOLORREF | DIRTY_FOGCOLOR | DIRTY_TEXENV | DIRTY_STENCILREPLACEVALUE |
					DIRTY_ALPHACOLORMASK | DIRTY_SHADERBLEND | DIRTY_TEXCLAMP | DIRTY_DEPAL | DIRTY_DEPTHRANGE;
				if (!psBaseBound_ || (dirty & DIRTY_FS_BASE)) {
					GX2SetPixelUniformBlock(1, sizeof(ub_base), push_base);
					psBaseBound_ = true;
				}
			}
			if (push_lights) {
				GX2SetVertexUniformBlock(2, sizeof(ub_lights), push_lights);
			}
			if (push_bones) {
				GX2SetVertexUniformBlock(3, sizeof(ub_bones), push_bones);
			}
		}
		g_gx2Perf.uboTicks += OSGetSystemTime() - uboT0;
		gstate_c.CleanUniforms();
		if (deferred) {
			gstate_c.Dirty(deferred);
		}
		return dirty & ~deferred;
	}
	gstate_c.CleanUniforms();
	return dirty;
}

void ShaderManagerGX2::GetShaders(int prim, u32 vertType, GX2VShader **vshader, GX2PShader **fshader,
											 bool useHWTransform, bool useHWTessellation) {
	VShaderID VSID;
	FShaderID FSID;

	if (gstate_c.IsDirty(DIRTY_VERTEXSHADER_STATE)) {
		gstate_c.Clean(DIRTY_VERTEXSHADER_STATE);
		ComputeVertexShaderID(&VSID, vertType, useHWTransform, useHWTessellation);
	} else {
		VSID = lastVSID_;
	}

	if (gstate_c.IsDirty(DIRTY_FRAGMENTSHADER_STATE)) {
		gstate_c.Clean(DIRTY_FRAGMENTSHADER_STATE);
		ComputeFragmentShaderID(&FSID, draw_->GetBugs());
	} else {
		FSID = lastFSID_;
	}

	// Just update uniforms if this is the same shader as last time.
	if (lastVShader_ != nullptr && lastFShader_ != nullptr && VSID == lastVSID_ && FSID == lastFSID_) {
		*vshader = lastVShader_;
		*fshader = lastFShader_;
		// Already all set, no need to look up in shader maps.
		return;
	}

	GX2VShader *vs;
	if (lastVShader_ != nullptr && VSID == lastVSID_) {
		vs = lastVShader_;
	} else {
		u32 vsSlot = ((VSID.d[0] * 2654435761u) ^ VSID.d[1]) & kShaderFastMask;
		if (vsFast_[vsSlot].shader && vsFast_[vsSlot].id == VSID) {
			vs = vsFast_[vsSlot].shader;
		} else {
			VSCache::iterator vsIter = vsCache_.find(VSID);
			if (vsIter == vsCache_.end()) {
				// Vertex shader not in cache. Let's generate it.
				NOTICE_LOG(G3D, "new VS: %s", VertexShaderDesc(VSID).c_str());
				vs = new GX2VShader(VSID);
				vsCache_[VSID] = vs;
			} else {
				vs = vsIter->second;
			}
			vsFast_[vsSlot].id = VSID;
			vsFast_[vsSlot].shader = vs;
		}
		lastVSID_ = VSID;
	}

	GX2PShader *fs;
	if (lastFShader_ != nullptr && FSID == lastFSID_) {
		fs = lastFShader_;
	} else {
		u32 fsSlot = ((FSID.d[0] * 2654435761u) ^ FSID.d[1]) & kShaderFastMask;
		if (fsFast_[fsSlot].shader && fsFast_[fsSlot].id == FSID) {
			fs = fsFast_[fsSlot].shader;
		} else {
			FSCache::iterator fsIter = fsCache_.find(FSID);
			if (fsIter == fsCache_.end()) {
				// Fragment shader not in cache. Let's generate it.
				NOTICE_LOG(G3D, "new FS: %s", FragmentShaderDesc(FSID).c_str());
				fs = new GX2PShader(FSID);
				fsCache_[FSID] = fs;
			} else {
				fs = fsIter->second;
			}
			fsFast_[fsSlot].id = FSID;
			fsFast_[fsSlot].shader = fs;
		}
		lastFSID_ = FSID;
	}

	lastVShader_ = vs;
	lastFShader_ = fs;

	*vshader = vs;
	*fshader = fs;

	if (vs->IsFallback())
		GX2SetVertexUniformBlock(4, sizeof(UB_VSID), vs->ub_id);
	if (fs->IsFallback())
		GX2SetPixelUniformBlock(5, sizeof(UB_FSID), fs->ub_id);
}

std::vector<std::string> ShaderManagerGX2::DebugGetShaderIDs(DebugShaderType type) {
	std::string id;
	std::vector<std::string> ids;
	switch (type) {
	case SHADER_TYPE_VERTEX: {
		for (auto iter : vsCache_) {
			iter.first.ToString(&id);
			ids.push_back(id);
		}
		break;
	}
	case SHADER_TYPE_FRAGMENT: {
		for (auto iter : fsCache_) {
			iter.first.ToString(&id);
			ids.push_back(id);
		}
		break;
	}
	default: break;
	}
	return ids;
}

std::string ShaderManagerGX2::DebugGetShaderString(std::string id, DebugShaderType type,
																	DebugShaderStringType stringType) {
	ShaderID shaderId;
	shaderId.FromString(id);
	switch (type) {
	case SHADER_TYPE_VERTEX: {
		auto iter = vsCache_.find(VShaderID(shaderId));
		if (iter == vsCache_.end()) {
			return "";
		}
		return iter->second->GetShaderString(stringType);
	}

	case SHADER_TYPE_FRAGMENT: {
		auto iter = fsCache_.find(FShaderID(shaderId));
		if (iter == fsCache_.end()) {
			return "";
		}
		return iter->second->GetShaderString(stringType);
	}
	default: return "N/A";
	}
}
