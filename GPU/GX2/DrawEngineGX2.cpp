// Copyright (c) 2012- PPSSPP Project.

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
#include <vector>

#include "Common/Profiler/Profiler.h"
#include "Common/MemoryUtil.h"
#include "Core/MemMap.h"
#include "Core/Host.h"
#include "Core/System.h"
#include "Core/Reporting.h"
#include "Core/Config.h"
#include "Core/CoreTiming.h"

#include "GPU/Math3D.h"
#include "GPU/GPUState.h"
#include "GPU/ge_constants.h"

#include "GPU/Common/TextureDecoder.h"
#include "GPU/Common/SplineCommon.h"

#include "GPU/Common/TransformCommon.h"
#include "GPU/Common/VertexDecoderCommon.h"
#include "GPU/Common/SoftwareTransformCommon.h"
#include <wiiu/os/time.h>
#include "GPU/GX2/FramebufferManagerGX2.h"
#include "GPU/GX2/TextureCacheGX2.h"
#include "GPU/GX2/DrawEngineGX2.h"
#include "GPU/GX2/ShaderManagerGX2.h"
#include "GPU/GX2/GPU_GX2.h"

static const GX2PrimitiveMode GX2prim[8] = {
	GX2_PRIMITIVE_MODE_POINTS,
	GX2_PRIMITIVE_MODE_LINES,
	GX2_PRIMITIVE_MODE_LINE_STRIP,
	GX2_PRIMITIVE_MODE_TRIANGLES,
	GX2_PRIMITIVE_MODE_TRIANGLE_STRIP,
	GX2_PRIMITIVE_MODE_TRIANGLES,
	GX2_PRIMITIVE_MODE_TRIANGLES,
	GX2_PRIMITIVE_MODE_INVALID
};

#define VERTEXCACHE_DECIMATION_INTERVAL 17

enum { VAI_KILL_AGE = 120, VAI_UNRELIABLE_KILL_AGE = 240, VAI_UNRELIABLE_KILL_MAX = 4 };
enum {
	VERTEX_PUSH_SIZE = 1024 * 1024 * 16,
	INDEX_PUSH_SIZE = 1024 * 1024 * 4,
	UBO_PUSH_SIZE = 1024 * 1024 * 16,
};

static const GX2AttribStream TransformedVertexElements[] = {
	{ 0, 0, 0, GX2_ATTRIB_FORMAT_FLOAT_32_32_32_32, GX2_ATTRIB_INDEX_PER_VERTEX, 0, GX2_COMP_SEL(_x, _y, _z, _w), GX2_ENDIAN_SWAP_DEFAULT },
	{ 1, 0, 16, GX2_ATTRIB_FORMAT_FLOAT_32_32_32, GX2_ATTRIB_INDEX_PER_VERTEX, 0, GX2_COMP_SEL(_x, _y, _z, _1), GX2_ENDIAN_SWAP_DEFAULT },
	{ 2, 0, 28, GX2_ATTRIB_FORMAT_UNORM_8_8_8_8, GX2_ATTRIB_INDEX_PER_VERTEX, 0, GX2_COMP_SEL(_r, _g, _b, _a), GX2_ENDIAN_SWAP_NONE },
	{ 3, 0, 32, GX2_ATTRIB_FORMAT_UNORM_8_8_8_8, GX2_ATTRIB_INDEX_PER_VERTEX, 0, GX2_COMP_SEL(_r, _g, _b, _a), GX2_ENDIAN_SWAP_NONE },
};

DrawEngineGX2::DrawEngineGX2(Draw::DrawContext *draw, GX2ContextState *context)
	: draw_(draw),
	  context_(context),
	  vai_(256),
	  fetchShaderMap_(32),
	  blendCache_(32),
	  depthStencilCache_(64),
	  rasterCache_(4)
{
	decOptions_.expandAllWeightsToFloat = true;
	decOptions_.expand8BitNormalsToFloat = true;

	decimationCounter_ = VERTEXCACHE_DECIMATION_INTERVAL;

	// All this is a LOT of memory, need to see if we can cut down somehow.
	decoded = (u8 *)MEM2_alloc(DECODED_VERTEX_BUFFER_SIZE, GX2_VERTEX_BUFFER_ALIGNMENT);
	decIndex = (u16 *)MEM2_alloc(DECODED_INDEX_BUFFER_SIZE, GX2_INDEX_BUFFER_ALIGNMENT);

	indexGen.Setup(decIndex);

	InitDeviceObjects();
}

DrawEngineGX2::~DrawEngineGX2() {
	DestroyDeviceObjects();
	MEM2_free(decoded);
	MEM2_free(decIndex);
}

void DrawEngineGX2::InitDeviceObjects() {
	pushVerts_ = new PushBufferGX2(VERTEX_PUSH_SIZE, GX2_VERTEX_BUFFER_ALIGNMENT, GX2_INVALIDATE_MODE_CPU_ATTRIBUTE_BUFFER);
	pushInds_ = new PushBufferGX2(INDEX_PUSH_SIZE, GX2_INDEX_BUFFER_ALIGNMENT, GX2_INVALIDATE_MODE_CPU_ATTRIBUTE_BUFFER);
	pushUBO_ = new PushBufferGX2(UBO_PUSH_SIZE, GX2_UNIFORM_BLOCK_ALIGNMENT, GX2_INVALIDATE_MODE_CPU_UNIFORM_BLOCK);

	tessDataTransfer = new TessellationDataTransferGX2(context_);
}

void DrawEngineGX2::ClearTrackedVertexArrays() {
	vai_.Iterate([&](u32 hash, VertexArrayInfoGX2 *vai) { delete vai; });
	vai_.Clear();
}

void DrawEngineGX2::ClearInputLayoutMap() {
	fetchShaderMap_.Iterate([&](const FetchShaderKey &key, GX2FetchShader *il) {
		MEM2_free(il->program);
		delete il;
	});
	fetchShaderMap_.Clear();
	lastBoundFetchShader_ = nullptr;
}

void DrawEngineGX2::Resized() {
	DrawEngineCommon::Resized();
	ClearInputLayoutMap();
}

void DrawEngineGX2::DestroyDeviceObjects() {
	ClearTrackedVertexArrays();
	ClearInputLayoutMap();
	delete tessDataTransfer;
	delete pushVerts_;
	delete pushInds_;
	delete pushUBO_;
	tessDataTransfer = nullptr;
	pushVerts_ = nullptr;
	pushInds_ = nullptr;
	pushUBO_ = nullptr;
	depthStencilCache_.Iterate([&](const u64 &key, GX2DepthStencilControlReg *ds) { free(ds); });
	depthStencilCache_.Clear();
	blendCache_.Iterate([&](const u64 &key, GX2BlendState *bs) { free(bs); });
	blendCache_.Clear();
	rasterCache_.Iterate([&](const u32 &key, GX2RasterizerState *rs) { free(rs); });
	rasterCache_.Clear();
}

struct DeclTypeInfo {
	u32 mask;
	GX2EndianSwapMode endianSwap;
	GX2AttribFormat format;
};

static const DeclTypeInfo VComp[] = {
	{ GX2_COMP_SEL(_0, _0, _0, _0), GX2_ENDIAN_SWAP_DEFAULT, GX2_ATTRIB_FORMAT_FLOAT_32_32_32_32 }, // DEC_NONE,
	{ GX2_COMP_SEL(_x, _0, _0, _1), GX2_ENDIAN_SWAP_DEFAULT, GX2_ATTRIB_FORMAT_FLOAT_32 },          // DEC_FLOAT_1,
	{ GX2_COMP_SEL(_x, _y, _0, _1), GX2_ENDIAN_SWAP_DEFAULT, GX2_ATTRIB_FORMAT_FLOAT_32_32 },       // DEC_FLOAT_2,
	{ GX2_COMP_SEL(_x, _y, _z, _1), GX2_ENDIAN_SWAP_DEFAULT, GX2_ATTRIB_FORMAT_FLOAT_32_32_32 },    // DEC_FLOAT_3,
	{ GX2_COMP_SEL(_x, _y, _z, _w), GX2_ENDIAN_SWAP_DEFAULT, GX2_ATTRIB_FORMAT_FLOAT_32_32_32_32 }, // DEC_FLOAT_4,
	{ GX2_COMP_SEL(_x, _y, _z, _1), GX2_ENDIAN_SWAP_NONE, GX2_ATTRIB_FORMAT_SNORM_8_8_8_8 },     // DEC_S8_3,
	{ GX2_COMP_SEL(_x, _y, _z, _1), GX2_ENDIAN_SWAP_8_IN_16, GX2_ATTRIB_FORMAT_SNORM_16_16_16_16 }, // DEC_S16_3,
	{ GX2_COMP_SEL(_x, _0, _0, _1), GX2_ENDIAN_SWAP_DEFAULT, GX2_ATTRIB_FORMAT_UNORM_8 },           // DEC_U8_1,
	{ GX2_COMP_SEL(_x, _y, _0, _1), GX2_ENDIAN_SWAP_NONE, GX2_ATTRIB_FORMAT_UNORM_8_8 },         // DEC_U8_2,
	{ GX2_COMP_SEL(_x, _y, _z, _1), GX2_ENDIAN_SWAP_NONE, GX2_ATTRIB_FORMAT_UNORM_8_8_8_8 },     // DEC_U8_3,
	{ GX2_COMP_SEL(_x, _y, _z, _w), GX2_ENDIAN_SWAP_NONE, GX2_ATTRIB_FORMAT_UNORM_8_8_8_8 },     // DEC_U8_4,
	{ GX2_COMP_SEL(_x, _0, _0, _1), GX2_ENDIAN_SWAP_8_IN_16, GX2_ATTRIB_FORMAT_UNORM_16 },          // DEC_U16_1,
	{ GX2_COMP_SEL(_x, _y, _0, _1), GX2_ENDIAN_SWAP_8_IN_16, GX2_ATTRIB_FORMAT_UNORM_16_16 },       // DEC_U16_2,
	{ GX2_COMP_SEL(_x, _y, _z, _1), GX2_ENDIAN_SWAP_8_IN_16, GX2_ATTRIB_FORMAT_UNORM_16_16_16_16 }, // DEC_U16_3,
	{ GX2_COMP_SEL(_x, _y, _z, _w), GX2_ENDIAN_SWAP_8_IN_16, GX2_ATTRIB_FORMAT_UNORM_16_16_16_16 }, // DEC_U16_4,
};

static void VertexAttribSetup(GX2AttribStream *VertexElement, u8 fmt, u8 offset, u8 location) {
	VertexElement->location = location;
	VertexElement->buffer = 0;
	VertexElement->offset = offset;
	VertexElement->format = VComp[fmt & 0xF].format;
	VertexElement->type = GX2_ATTRIB_INDEX_PER_VERTEX;
	VertexElement->aluDivisor = 0;
	VertexElement->mask = VComp[fmt & 0xF].mask;
	VertexElement->endianSwap = VComp[fmt & 0xF].endianSwap;
}

GX2FetchShader *DrawEngineGX2::SetupFetchShaderForDraw(GX2VertexShader *vshader, const DecVtxFormat &decFmt, u32 pspFmt) {
	// TODO: Instead of one for each vshader, we can reduce it to one for each type of shader
	// that reads TEXCOORD or not, etc. Not sure if worth it.
	FetchShaderKey key{ vshader, decFmt.id };
	GX2FetchShader *fetchShader = fetchShaderMap_.Get(key);
	if (fetchShader) {
		return fetchShader;
	}

	GX2AttribStream VertexElements[8];
	GX2AttribStream *VertexElement = &VertexElements[0];

	// POSITION
	// Always
	VertexAttribSetup(VertexElement, decFmt.posfmt, decFmt.posoff, (u32)GX2Gen::VSInput::POSITION);
	VertexElement++;

	// TC
	if (decFmt.uvfmt != 0) {
		VertexAttribSetup(VertexElement, decFmt.uvfmt, decFmt.uvoff, (u32)GX2Gen::VSInput::COORDS);
		VertexElement++;
	}

	// COLOR
	if (decFmt.c0fmt != 0) {
		VertexAttribSetup(VertexElement, decFmt.c0fmt, decFmt.c0off, (u32)GX2Gen::VSInput::COLOR0);
		VertexElement++;
	}

	// Never used ?
	if (decFmt.c1fmt != 0) {
		VertexAttribSetup(VertexElement, decFmt.c1fmt, decFmt.c1off, (u32)GX2Gen::VSInput::COLOR1);
		VertexElement++;
	}

	// NORMAL
	if (decFmt.nrmfmt != 0) {
		VertexAttribSetup(VertexElement, decFmt.nrmfmt, decFmt.nrmoff, (u32)GX2Gen::VSInput::NORMAL);
		VertexElement++;
	}

	// WEIGHT
	if (decFmt.w0fmt != 0) {
		VertexAttribSetup(VertexElement, decFmt.w0fmt, decFmt.w0off, (u32)GX2Gen::VSInput::WEIGHT0);
		VertexElement++;
	}

	if (decFmt.w1fmt != 0) {
		VertexAttribSetup(VertexElement, decFmt.w1fmt, decFmt.w1off, (u32)GX2Gen::VSInput::WEIGHT1);
		VertexElement++;
	}

	// Create fetchShader
	fetchShader = new GX2FetchShader;
	fetchShader->size = GX2CalcFetchShaderSize(VertexElement - VertexElements);
	fetchShader->program = (u8 *)MEM2_alloc(fetchShader->size, GX2_SHADER_ALIGNMENT);
	GX2InitFetchShader(fetchShader, fetchShader->program, VertexElement - VertexElements, VertexElements);
	GX2Invalidate(GX2_INVALIDATE_MODE_CPU_SHADER, fetchShader->program, fetchShader->size);

	// Add it to map
	fetchShaderMap_.Insert(key, fetchShader);
	return fetchShader;
}

void DrawEngineGX2::MarkUnreliable(VertexArrayInfoGX2 *vai) {
	vai->status = VertexArrayInfoGX2::VAI_UNRELIABLE;

	MEM2_free(vai->vbo);
	vai->vbo = nullptr;

	MEM2_free(vai->ebo);
	vai->ebo = nullptr;
}

extern int g_gx2FsDbg;
static int s_snapFrame = -1;
static int s_snapCount = 0;

// Diagnostic: after a traced draw, read the center and a corner pixel of the render target.
static void SnapPixelsAfterDraw(Draw::DrawContext *draw, FramebufferManagerGX2 *fbm, int drawNo) {
	VirtualFramebuffer *vfb = fbm->GetCurrentRenderVFB();
	if (!vfb || !vfb->fbo || vfb->renderWidth < 240 || vfb->renderHeight < 136)
		return;
	u32 c = 0, k = 0;
	draw->CopyFramebufferToMemorySync(vfb->fbo, Draw::FB_COLOR_BIT, 240, 136, 1, 1, Draw::DataFormat::R8G8B8A8_UNORM, &c, 1, "snapd");
	draw->CopyFramebufferToMemorySync(vfb->fbo, Draw::FB_COLOR_BIT, 4, 4, 1, 1, Draw::DataFormat::R8G8B8A8_UNORM, &k, 1, "snapd");
	NOTICE_LOG(G3D, "GX2SNAP   after draw %d: center=%08x corner=%08x rt=%08x %dx%d fmt=%d stride=%d fbo=%p", drawNo, c, k, vfb->fb_address, vfb->renderWidth, vfb->renderHeight, (int)vfb->format, (int)vfb->fb_stride, (void *)vfb->fbo);
}

void DrawEngineGX2::BeginFrame() {
	pushVerts_->Reset();
	pushInds_->Reset();
	pushUBO_->Reset();
	if (shaderManager_) {
		shaderManager_->DirtyLastShader();
	}

	if (false && gpuStats.numFlips > 0 && (gpuStats.numFlips % 300) == 180 && s_snapFrame != gpuStats.numFlips) {
		s_snapFrame = gpuStats.numFlips;
		s_snapCount = 0;
		{
			// Cycle pixel-shader diagnostic variants once the menu scene is up (frame >= 1080).
			int mode = 0; // diagnostic cycling disabled
			if (mode != g_gx2FsDbg) {
				g_gx2FsDbg = mode;
				shaderManager_->ClearShaders();
			}
		}
		NOTICE_LOG(G3D, "GX2SNAP frame %d FSDBG=%d", s_snapFrame, g_gx2FsDbg);
	}

	// Diagnostic: right after a traced frame, read a few pixels back from the PSP framebuffer.
	if (s_snapFrame >= 0 && gpuStats.numFlips == s_snapFrame + 1) {
		VirtualFramebuffer *vfb = framebufferManager_->GetDisplayVFB();
		if (vfb && vfb->fbo) {
			static const int pts[5][2] = { { 4, 4 }, { 240, 136 }, { 470, 4 }, { 4, 260 }, { 470, 260 } };
			u32 px[5] = {};
			for (int i = 0; i < 5; i++) {
				int x = pts[i][0] < vfb->renderWidth ? pts[i][0] : vfb->renderWidth - 1;
				int y = pts[i][1] < vfb->renderHeight ? pts[i][1] : vfb->renderHeight - 1;
				draw_->CopyFramebufferToMemorySync(vfb->fbo, Draw::FB_COLOR_BIT, x, y, 1, 1, Draw::DataFormat::R8G8B8A8_UNORM, &px[i], 1, "snap");
			}
			NOTICE_LOG(G3D, "GX2SNAP PIXELS fb=%08x %dx%d fmt=%d: (4,4)=%08x (240,136)=%08x (470,4)=%08x (4,260)=%08x (470,260)=%08x",
				vfb->fb_address, vfb->renderWidth, vfb->renderHeight, (int)vfb->format, px[0], px[1], px[2], px[3], px[4]);
		} else {
			NOTICE_LOG(G3D, "GX2SNAP PIXELS: no current vfb");
		}
		// Full-frame dump as a coarse grid so we can see the shape of what was rendered.
		if (vfb && vfb->fbo && vfb->renderWidth >= 480 && vfb->renderHeight >= 272) {
			static std::vector<u32> img;
			img.resize(480 * 272);
			draw_->CopyFramebufferToMemorySync(vfb->fbo, Draw::FB_COLOR_BIT, 0, 0, 480, 272, Draw::DataFormat::R8G8B8A8_UNORM, img.data(), 480, "snapgrid");
			for (int gy = 0; gy < 272; gy += 34) {
				char line[400];
				int n = 0;
				for (int gx = 0; gx < 480; gx += 40) {
					n += snprintf(line + n, sizeof(line) - n, "%08x ", img[(gy + 2) * 480 + gx + 2]);
				}
				NOTICE_LOG(G3D, "GX2SNAP GRID y=%3d: %s", gy + 2, line);
			}
		}
		s_snapFrame = -2;
	}

	if (--decimationCounter_ <= 0) {
		decimationCounter_ = VERTEXCACHE_DECIMATION_INTERVAL;
	} else {
		return;
	}

	const int threshold = gpuStats.numFlips - VAI_KILL_AGE;
	const int unreliableThreshold = gpuStats.numFlips - VAI_UNRELIABLE_KILL_AGE;
	int unreliableLeft = VAI_UNRELIABLE_KILL_MAX;
	vai_.Iterate([&](u32 hash, VertexArrayInfoGX2 *vai) {
		bool kill;
		if (vai->status == VertexArrayInfoGX2::VAI_UNRELIABLE) {
			// We limit killing unreliable so we don't rehash too often.
			kill = vai->lastFrame < unreliableThreshold && --unreliableLeft >= 0;
		} else {
			kill = vai->lastFrame < threshold;
		}
		if (kill) {
			delete vai;
			vai_.Remove(hash);
		}
	});
	vai_.Maintain();

	// Enable if you want to see vertex decoders in the log output. Need a better way.
#if 0
	char buffer[16384];
	for (std::map<u32, VertexDecoder*>::iterator dec = decoderMap_.begin(); dec != decoderMap_.end(); ++dec) {
		char *ptr = buffer;
		ptr += dec->second->ToString(ptr);
		//		*ptr++ = '\n';
		NOTICE_LOG(G3D, buffer);
	}
#endif

	lastRenderStepId_ = -1;
}

VertexArrayInfoGX2::~VertexArrayInfoGX2() {
	MEM2_free(vbo);
	MEM2_free(ebo);
}

static u32 SwapRB(u32 c) { return (c & 0xFF00FF00) | ((c >> 16) & 0xFF) | ((c << 16) & 0xFF0000); }

// The inline wrapper in the header checks for numDrawCalls == 0
void DrawEngineGX2::DoFlush() {
	PROFILE_THIS_SCOPE("Flush");
	gpuStats.numFlushes++;
	gpuStats.numTrackedVertexArrays = (int)vai_.size();

	// In GX2, we're synchronous and state carries over so all we reset here on a new step is the viewport/scissor.
	int curRenderStepId = draw_->GetCurrentStepId();
	if (lastRenderStepId_ != curRenderStepId) {
		// Dirty everything that has dynamic state that will need re-recording.
		gstate_c.Dirty(DIRTY_VIEWPORTSCISSOR_STATE);
		lastRenderStepId_ = curRenderStepId;
	}

	// This is not done on every drawcall, we collect vertex data
	// until critical state changes. That's when we draw (flush).

	GEPrimitiveType prim = prevPrim_;
	u64 flushT0 = OSGetSystemTime();
	ApplyDrawState(prim);

	// Always use software for flat shading to fix the provoking index.
	bool tess = gstate_c.bezier || gstate_c.spline;
	bool useHWTransform = CanUseHardwareTransform(prim);

	if (useHWTransform) {
		PROFILE_THIS_SCOPE("T-hard");
		void *vb_ = nullptr;
		void *ib_ = nullptr;

		int vertexCount = 0;
		int maxIndex = 0;
		bool useElements = true;

		// Cannot cache vertex data with morph enabled.
		bool useCache = g_Config.bVertexCache && !(lastVType_ & GE_VTYPE_MORPHCOUNT_MASK);
		// Also avoid caching when software skinning.
		if (g_Config.bSoftwareSkinning && (lastVType_ & GE_VTYPE_WEIGHT_MASK))
			useCache = false;

		if (useCache) {
			PROFILE_THIS_SCOPE("vcache");
			u32 id = dcid_ ^ gstate.getUVGenMode(); // This can have an effect on which UV decoder we need to use! And hence what the decoded data will look like. See #9263

			VertexArrayInfoGX2 *vai = vai_.Get(id);
			if (!vai) {
				vai = new VertexArrayInfoGX2();
				vai_.Insert(id, vai);
			}

			switch (vai->status) {
			case VertexArrayInfoGX2::VAI_NEW: {
				// Haven't seen this one before.
				uint64_t dataHash = ComputeHash();
				vai->hash = dataHash;
				vai->minihash = ComputeMiniHash();
				vai->status = VertexArrayInfoGX2::VAI_HASHING;
				vai->drawsUntilNextFullHash = 0;
				DecodeVerts(decoded); // writes to indexGen
				vai->numVerts = indexGen.VertexCount();
				vai->prim = indexGen.Prim();
				vai->maxIndex = indexGen.MaxIndex();
				vai->flags = gstate_c.vertexFullAlpha ? VAI11_FLAG_VERTEXFULLALPHA : 0;
				goto rotateVBO;
			}

				// Hashing - still gaining confidence about the buffer.
				// But if we get this far it's likely to be worth creating a vertex buffer.
			case VertexArrayInfoGX2::VAI_HASHING: {
				PROFILE_THIS_SCOPE("vcachehash");
				vai->numDraws++;
				if (vai->lastFrame != gpuStats.numFlips) {
					vai->numFrames++;
				}
				if (vai->drawsUntilNextFullHash == 0) {
					// Let's try to skip a full hash if mini would fail.
					const u32 newMiniHash = ComputeMiniHash();
					uint64_t newHash = vai->hash;
					if (newMiniHash == vai->minihash) {
						newHash = ComputeHash();
					}
					if (newMiniHash != vai->minihash || newHash != vai->hash) {
						MarkUnreliable(vai);
						DecodeVerts(decoded);
						goto rotateVBO;
					}
					if (vai->numVerts > 64) {
						// exponential backoff up to 24 frames
						vai->drawsUntilNextFullHash = std::min(24, vai->numFrames);
					} else {
						// Lower numbers seem much more likely to change.
						vai->drawsUntilNextFullHash = 0;
					}
				} else {
					vai->drawsUntilNextFullHash--;
					u32 newMiniHash = ComputeMiniHash();
					if (newMiniHash != vai->minihash) {
						MarkUnreliable(vai);
						DecodeVerts(decoded);
						goto rotateVBO;
					}
				}

				if (vai->vbo == 0) {
					DecodeVerts(decoded);
					vai->numVerts = indexGen.VertexCount();
					vai->prim = indexGen.Prim();
					vai->maxIndex = indexGen.MaxIndex();
					vai->flags = gstate_c.vertexFullAlpha ? VAI11_FLAG_VERTEXFULLALPHA : 0;
					useElements = !indexGen.SeenOnlyPurePrims() || prim == GE_PRIM_TRIANGLE_FAN;
					if (!useElements && indexGen.PureCount()) {
						vai->numVerts = indexGen.PureCount();
					}

					_dbg_assert_msg_(gstate_c.vertBounds.minV >= gstate_c.vertBounds.maxV, "Should not have checked UVs when caching.");

					// TODO: Combine these two into one buffer?
					u32 size = dec_->GetDecVtxFmt().stride * (indexGen.MaxIndex() + 1);
					vai->vbo = MEM2_alloc(size, GX2_VERTEX_BUFFER_ALIGNMENT);
					memcpy(vai->vbo, decoded, size);
					GX2Invalidate(GX2_INVALIDATE_MODE_CPU_ATTRIBUTE_BUFFER, vai->vbo, size);
					if (useElements) {
						u32 size = sizeof(short) * indexGen.VertexCount();
						vai->ebo = MEM2_alloc(size, GX2_INDEX_BUFFER_ALIGNMENT);
						memcpy(vai->ebo, decIndex, size);
						GX2Invalidate(GX2_INVALIDATE_MODE_CPU_ATTRIBUTE_BUFFER, vai->ebo, size);
					} else {
						vai->ebo = 0;
					}
				} else {
					gpuStats.numCachedDrawCalls++;
					useElements = vai->ebo ? true : false;
					gpuStats.numCachedVertsDrawn += vai->numVerts;
					gstate_c.vertexFullAlpha = vai->flags & VAI11_FLAG_VERTEXFULLALPHA;
				}
				vb_ = vai->vbo;
				ib_ = vai->ebo;
				vertexCount = vai->numVerts;
				maxIndex = vai->maxIndex;
				prim = static_cast<GEPrimitiveType>(vai->prim);
				break;
			}

				// Reliable - we don't even bother hashing anymore. Right now we don't go here until after a very long time.
			case VertexArrayInfoGX2::VAI_RELIABLE: {
				vai->numDraws++;
				if (vai->lastFrame != gpuStats.numFlips) {
					vai->numFrames++;
				}
				gpuStats.numCachedDrawCalls++;
				gpuStats.numCachedVertsDrawn += vai->numVerts;
				vb_ = vai->vbo;
				ib_ = vai->ebo;

				vertexCount = vai->numVerts;

				maxIndex = vai->maxIndex;
				prim = static_cast<GEPrimitiveType>(vai->prim);

				gstate_c.vertexFullAlpha = vai->flags & VAI11_FLAG_VERTEXFULLALPHA;
				break;
			}

			case VertexArrayInfoGX2::VAI_UNRELIABLE: {
				vai->numDraws++;
				if (vai->lastFrame != gpuStats.numFlips) {
					vai->numFrames++;
				}
				DecodeVerts(decoded);
				goto rotateVBO;
			}
			}

			vai->lastFrame = gpuStats.numFlips;
		} else {
			DecodeVerts(decoded);
		rotateVBO:
			gpuStats.numUncachedVertsDrawn += indexGen.VertexCount();
			useElements = !indexGen.SeenOnlyPurePrims() || prim == GE_PRIM_TRIANGLE_FAN;
			vertexCount = indexGen.VertexCount();
			maxIndex = indexGen.MaxIndex();
			if (!useElements && indexGen.PureCount()) {
				vertexCount = indexGen.PureCount();
			}
			prim = indexGen.Prim();
		}

		VERBOSE_LOG(G3D, "Flush prim %i! %i verts in one go", prim, vertexCount);
		bool hasColor = (lastVType_ & GE_VTYPE_COL_MASK) != GE_VTYPE_COL_NONE;
		if (gstate.isModeThrough()) {
			gstate_c.vertexFullAlpha = gstate_c.vertexFullAlpha && (hasColor || gstate.getMaterialAmbientA() == 255);
		} else {
			gstate_c.vertexFullAlpha = gstate_c.vertexFullAlpha && ((hasColor && (gstate.materialupdate & 1)) || gstate.getMaterialAmbientA() == 255) && (!gstate.isLightingEnabled() || gstate.getAmbientA() == 255);
		}

		ApplyDrawStateLate(false, dynState_.stencilRef);

		GX2VShader *vshader;
		GX2PShader *fshader;
		shaderManager_->GetShaders(prim, lastVType_, &vshader, &fshader, useHWTransform, useHWTessellation_);
		g_gx2Perf.draws++;
		if (fshader->IsFallback()) g_gx2Perf.psAllDraws++;
		if (vshader->IsFallback()) g_gx2Perf.vsFallbackDraws++;
		if (gpuStats.numFlips == s_snapFrame && s_snapCount < 400) {
			s_snapCount++;
			NOTICE_LOG(G3D, "GX2SNAP HW #%d prim=%d v=%d clr=%d bld=%d(%d,%d,%d) mask=%x sten=%d fog=%d(%08x) tex=%d(%d) vs=%08x:%08x fs=%08x:%08x",
				s_snapCount, prim, vertexCount, gstate.isModeClear(), gstate.isAlphaBlendEnabled(),
				gstate.getBlendFuncA(), gstate.getBlendFuncB(), gstate.getBlendEq(),
				(u32)keys_.blend.colorWriteMask, gstate.isStencilTestEnabled(), gstate.isFogEnabled(),
				gstate.fogcolor, gstate.isTextureMapEnabled(), gstate.getTextureFunction(),
				vshader->ID().d[1], vshader->ID().d[0], fshader->ID().d[1], fshader->ID().d[0]);
			{
				const u32 *w = (const u32 *)decoded;
				u32 stw = dec_->GetDecVtxFmt().stride / 4;
				NOTICE_LOG(G3D, "GX2SNAP   vt=%08x tfmt=%d tex=%dx%d taddr=%08x clutfmt=%d clutaddr=%08x atest=%d depth=%d/%d wr=%d stride=%u cached=%d v0=%08x %08x %08x %08x %08x %08x v1=%08x %08x %08x %08x %08x %08x",
					lastVType_, (int)gstate.getTextureFormat(), gstate.getTextureWidth(0), gstate.getTextureHeight(0), gstate.getTextureAddress(0),
					(int)gstate.getClutPaletteFormat(), gstate.getClutAddress(), gstate.isAlphaTestEnabled(), gstate.isDepthTestEnabled(), (int)gstate.getDepthTestFunction(), gstate.isDepthWriteEnabled(),
					(unsigned)stw * 4, vb_ ? 1 : 0,
					w[0], w[1], w[2], w[3], w[4], w[5], w[stw], w[stw + 1], w[stw + 2], w[stw + 3], w[stw + 4], w[stw + 5]);
				if (GX2Texture *gt = textureCache_ ? textureCache_->GetLastBoundTextureForDebug() : nullptr) {
					const u32 *img = (const u32 *)gt->surface.image;
					u32 tw = gt->surface.width, th = gt->surface.height, pitch = gt->surface.pitch;
					if (img && gt->surface.format == GX2_SURFACE_FORMAT_UNORM_R8_G8_B8_A8 && tw && th) {
						u32 white = 0, zero = 0, total = 0;
						for (u32 yy = 0; yy < th; yy += 2)
							for (u32 xx = 0; xx < tw; xx += 2) {
								u32 p = img[yy * pitch + xx];
								total++;
								if (p == 0xFFFFFFFF) white++;
								if (p == 0) zero++;
							}
						NOTICE_LOG(G3D, "GX2SNAP   TEX %ux%u pitch=%u fmt=%d img=%p samples=%u white=%u zero=%u row0=%08x %08x %08x %08x mid=%08x %08x %08x %08x",
							tw, th, pitch, (int)gt->surface.format, (void *)img, total, white, zero, img[0], img[1], img[2], img[3],
							img[(th / 2) * pitch + tw / 2], img[(th / 2) * pitch + tw / 2 + 1], img[(th / 2) * pitch + tw / 2 + 2], img[(th / 2) * pitch + tw / 2 + 3]);
					} else {
						NOTICE_LOG(G3D, "GX2SNAP   TEX %ux%u fmt=%d (not 8888 or no image)", tw, th, (int)gt->surface.format);
					}
				}
			}
		}
		GX2FetchShader *fetchShader = SetupFetchShaderForDraw(vshader, dec_->GetDecVtxFmt(), dec_->VertexType());
		if (boundShaderEpoch_ != shaderManager_->GetDirtyEpoch()) {
			boundShaderEpoch_ = shaderManager_->GetDirtyEpoch();
			lastBoundVShader_ = nullptr;
			lastBoundFShader_ = nullptr;
			lastBoundFetchShader_ = nullptr;
		}
		if (fshader != lastBoundFShader_) {
			GX2SetPixelShader(fshader);
			lastBoundFShader_ = fshader;
		}
		if (vshader != lastBoundVShader_) {
			GX2SetVertexShader(vshader);
			lastBoundVShader_ = vshader;
		}
		shaderManager_->UpdateUniforms(pushUBO_, framebufferManager_->UseBufferedRendering());

		if (fetchShader != lastBoundFetchShader_) {
			GX2SetFetchShader(fetchShader);
			lastBoundFetchShader_ = fetchShader;
		}
		u32 stride = dec_->GetDecVtxFmt().stride;
		//		GX2prim[prim];
		if (!vb_) {
			// Push!
			u32 vOffset;
			int vSize = (maxIndex + 1) * dec_->GetDecVtxFmt().stride;
			u8 *vptr = pushVerts_->BeginPush(&vOffset, vSize);
			memcpy(vptr, decoded, vSize);
			pushVerts_->EndPush();
			GX2SetAttribBuffer(0, vSize, stride, vptr);
			if (useElements) {
				u32 iOffset;
				int iSize = 2 * indexGen.VertexCount();
				u8 *iptr = pushInds_->BeginPush(&iOffset, iSize);
				memcpy(iptr, decIndex, iSize);
				pushInds_->EndPush();
				GX2DrawIndexedEx(GX2prim[prim], vertexCount, GX2_INDEX_TYPE_U16, iptr, 0, 1);
			} else {
				GX2DrawEx(GX2prim[prim], vertexCount, 0, 1);
			}
		} else {
			GX2SetAttribBuffer(0, (maxIndex + 1) * stride, stride, vb_);
			if (useElements) {
				GX2DrawIndexedEx(GX2prim[prim], vertexCount, GX2_INDEX_TYPE_U16, ib_, 0, 1);
			} else {
				GX2DrawEx(GX2prim[prim], vertexCount, 0, 1);
			}
		}
		if (gpuStats.numFlips == s_snapFrame && s_snapCount <= 400)
			SnapPixelsAfterDraw(draw_, framebufferManager_, s_snapCount);
	} else {
		PROFILE_THIS_SCOPE("T-soft");
		DecodeVerts(decoded);
		bool hasColor = (lastVType_ & GE_VTYPE_COL_MASK) != GE_VTYPE_COL_NONE;
		if (gstate.isModeThrough()) {
			gstate_c.vertexFullAlpha = gstate_c.vertexFullAlpha && (hasColor || gstate.getMaterialAmbientA() == 255);
		} else {
			gstate_c.vertexFullAlpha = gstate_c.vertexFullAlpha && ((hasColor && (gstate.materialupdate & 1)) || gstate.getMaterialAmbientA() == 255) && (!gstate.isLightingEnabled() || gstate.getAmbientA() == 255);
		}

		gpuStats.numUncachedVertsDrawn += indexGen.VertexCount();
		prim = indexGen.Prim();
		// Undo the strip optimization, not supported by the SW code yet.
		if (prim == GE_PRIM_TRIANGLE_STRIP)
			prim = GE_PRIM_TRIANGLES;
		VERBOSE_LOG(G3D, "Flush prim %i SW! %i verts in one go", prim, indexGen.VertexCount());

		u16 *inds = decIndex;
		SoftwareTransformResult result{};
		SoftwareTransformParams params{};
		params.decoded = decoded;
		params.transformed = transformed;
		params.transformedExpanded = transformedExpanded;
		params.fbman = framebufferManager_;
		params.texCache = textureCache_;
		params.allowClear = true;
		params.allowSeparateAlphaClear = false; // GX2 doesn't support separate alpha clears
		params.provokeFlatFirst = true;

		int maxIndex = indexGen.MaxIndex();
		SoftwareTransform swTransform(params);
		swTransform.Decode(prim, dec_->VertexType(), dec_->GetDecVtxFmt(), maxIndex, &result);
		if (result.action == SW_NOT_READY) {
			swTransform.DetectOffsetTexture(maxIndex);
			swTransform.BuildDrawingParams(prim, indexGen.VertexCount(), dec_->VertexType(), inds, maxIndex, &result);
		}

		if (result.setSafeSize)
			framebufferManager_->SetSafeSize(result.safeWidth, result.safeHeight);

		if (result.action == SW_DRAW_PRIMITIVES) {
			ApplyDrawStateLate(result.setStencil, result.stencilValue);

			GX2VShader *vshader;
			GX2PShader *fshader;
			shaderManager_->GetShaders(prim, lastVType_, &vshader, &fshader, false, false);
			g_gx2Perf.draws++;
			g_gx2Perf.swDraws++;
			if (fshader->IsFallback()) g_gx2Perf.psAllDraws++;
			if (gpuStats.numFlips == s_snapFrame && s_snapCount < 400) {
				s_snapCount++;
				NOTICE_LOG(G3D, "GX2SNAP SW #%d prim=%d v=%d clr=%d bld=%d(%d,%d,%d) mask=%x sten=%d fog=%d(%08x) tex=%d(%d) vs=%08x:%08x fs=%08x:%08x",
					s_snapCount, prim, result.drawNumTrans, gstate.isModeClear(), gstate.isAlphaBlendEnabled(),
					gstate.getBlendFuncA(), gstate.getBlendFuncB(), gstate.getBlendEq(),
					(u32)keys_.blend.colorWriteMask, gstate.isStencilTestEnabled(), gstate.isFogEnabled(),
					gstate.fogcolor, gstate.isTextureMapEnabled(), gstate.getTextureFunction(),
					vshader->ID().d[1], vshader->ID().d[0], fshader->ID().d[1], fshader->ID().d[0]);
				const u32 *w = (const u32 *)result.drawBuffer;
				u32 stw = sizeof(TransformedVertex) / 4;
				NOTICE_LOG(G3D, "GX2SNAP   SWV clrmask c=%d a=%d d=%d stride=%u indexed=%d tex=%dx%d v0=%08x %08x %08x %08x %08x %08x %08x %08x %08x %08x v1=%08x %08x %08x %08x %08x %08x %08x %08x %08x %08x",
					gstate.isClearModeColorMask(), gstate.isClearModeAlphaMask(), gstate.isClearModeDepthMask(), (unsigned)stw * 4, result.drawIndexed ? 1 : 0,
					gstate.getTextureWidth(0), gstate.getTextureHeight(0),
					w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9],
					w[stw], w[stw + 1], w[stw + 2], w[stw + 3], w[stw + 4], w[stw + 5], w[stw + 6], w[stw + 7], w[stw + 8], w[stw + 9]);
			}
			if (boundShaderEpoch_ != shaderManager_->GetDirtyEpoch()) {
				boundShaderEpoch_ = shaderManager_->GetDirtyEpoch();
				lastBoundVShader_ = nullptr;
				lastBoundFShader_ = nullptr;
				lastBoundFetchShader_ = nullptr;
			}
			if (fshader != lastBoundFShader_) {
				GX2SetPixelShader(fshader);
				lastBoundFShader_ = fshader;
			}
			if (vshader != lastBoundVShader_) {
				GX2SetVertexShader(vshader);
				lastBoundVShader_ = vshader;
			}
			shaderManager_->UpdateUniforms(pushUBO_, framebufferManager_->UseBufferedRendering());

			// We really do need a vertex layout for each vertex shader (or at least check its ID bits for what inputs it uses)!
			// Some vertex shaders ignore one of the inputs, and then the layout created from it will lack it, which will be a problem for others.
			FetchShaderKey key{ vshader, 0xFFFFFFFF }; // Let's use 0xFFFFFFFF to signify TransformedVertex
			GX2FetchShader *fetchShader = fetchShaderMap_.Get(key);
			if (!fetchShader) {
				fetchShader = new GX2FetchShader;
				fetchShader->size = GX2CalcFetchShaderSize(ARRAY_SIZE(TransformedVertexElements));
				fetchShader->program = (u8 *)MEM2_alloc(fetchShader->size, GX2_SHADER_ALIGNMENT);
				GX2InitFetchShader(fetchShader, fetchShader->program, ARRAY_SIZE(TransformedVertexElements), TransformedVertexElements);
				GX2Invalidate(GX2_INVALIDATE_MODE_CPU_SHADER, fetchShader->program, fetchShader->size);
				fetchShaderMap_.Insert(key, fetchShader);
			}
			if (fetchShader != lastBoundFetchShader_) {
				GX2SetFetchShader(fetchShader);
				lastBoundFetchShader_ = fetchShader;
			}

			u32 stride = sizeof(TransformedVertex);
			u32 vOffset = 0;
			int vSize = maxIndex * stride;
			u8 *vptr = pushVerts_->BeginPush(&vOffset, vSize);
			memcpy(vptr, result.drawBuffer, vSize);
			pushVerts_->EndPush();
			GX2SetAttribBuffer(0, vSize, stride, vptr);
			if (result.drawIndexed) {
				u32 iOffset;
				int iSize = sizeof(u16) * result.drawNumTrans;
				u8 *iptr = pushInds_->BeginPush(&iOffset, iSize);
				memcpy(iptr, inds, iSize);
				pushInds_->EndPush();
				GX2DrawIndexedEx(GX2prim[prim], result.drawNumTrans, GX2_INDEX_TYPE_U16, iptr, 0, 1);
			} else {
				GX2DrawEx(GX2prim[prim], result.drawNumTrans, 0, 1);
			}
			if (gpuStats.numFlips == s_snapFrame && s_snapCount <= 400)
				SnapPixelsAfterDraw(draw_, framebufferManager_, s_snapCount);
		} else if (result.action == SW_CLEAR) {
			u32 clearColor = result.color;
			float clearDepth = result.depth;

			u32 clearFlag = 0;

			if (gstate.isClearModeColorMask())
				clearFlag |= Draw::FBChannel::FB_COLOR_BIT;
			if (gstate.isClearModeAlphaMask())
				clearFlag |= Draw::FBChannel::FB_STENCIL_BIT;
			if (gstate.isClearModeDepthMask())
				clearFlag |= Draw::FBChannel::FB_DEPTH_BIT;

			if (clearFlag & Draw::FBChannel::FB_DEPTH_BIT) {
				framebufferManager_->SetDepthUpdated();
			}
			if (clearFlag & Draw::FBChannel::FB_COLOR_BIT) {
				framebufferManager_->SetColorUpdated(gstate_c.skipDrawReason);
			}

			u8 clearStencil = clearColor >> 24;
			if (gpuStats.numFlips == s_snapFrame && s_snapCount < 400) {
				s_snapCount++;
				NOTICE_LOG(G3D, "GX2SNAP CLEAR #%d flag=%x col=%08x depth=%f sten=%02x", s_snapCount, clearFlag, clearColor, clearDepth, clearStencil);
			}
			draw_->Clear(clearFlag, clearColor, clearDepth, clearStencil);

			if ((gstate_c.featureFlags & GPU_USE_CLEAR_RAM_HACK) && gstate.isClearModeColorMask() && (gstate.isClearModeAlphaMask() || gstate.FrameBufFormat() == GE_FORMAT_565)) {
				int scissorX1 = gstate.getScissorX1();
				int scissorY1 = gstate.getScissorY1();
				int scissorX2 = gstate.getScissorX2() + 1;
				int scissorY2 = gstate.getScissorY2() + 1;
				framebufferManager_->ApplyClearToMemory(scissorX1, scissorY1, scissorX2, scissorY2, clearColor);
			}
		}
	}

	gpuStats.numDrawCalls += numDrawCalls;
	gpuStats.numVertsSubmitted += vertexCountInDrawCalls_;
	g_gx2Perf.flushTicks += OSGetSystemTime() - flushT0;

	indexGen.Reset();
	decodedVerts_ = 0;
	numDrawCalls = 0;
	vertexCountInDrawCalls_ = 0;
	decodeCounter_ = 0;
	dcid_ = 0;
	prevPrim_ = GE_PRIM_INVALID;
	gstate_c.vertexFullAlpha = true;
	framebufferManager_->SetColorUpdated(gstate_c.skipDrawReason);

	// Now seems as good a time as any to reset the min/max coords, which we may examine later.
	gstate_c.vertBounds.minU = 512;
	gstate_c.vertBounds.minV = 512;
	gstate_c.vertBounds.maxU = 0;
	gstate_c.vertBounds.maxV = 0;

#if 0
	// We only support GPU debugging on Windows, and that's the only use case for this.
		GPUDebug::NotifyDraw();
#endif
}

void DrawEngineGX2::TessellationDataTransferGX2::SendDataToShader(const SimpleVertex *const *points, int size_u, int size_v, u32 vertType, const Spline::Weight2D &weights) {
#if 0
	// Position
	if (prevSize < size) {
		prevSize = size;
		MEM2_free(data_tex[0].surface.image);
		data_tex[0].surface.width = size;
		data_tex[0].surface.height = 1;
		data_tex[0].surface.depth = 1;
		data_tex[0].surface.dim = GX2_SURFACE_DIM_TEXTURE_1D;
		data_tex[0].surface.tileMode = GX2_TILE_MODE_LINEAR_ALIGNED;
		data_tex[0].surface.use = GX2_SURFACE_USE_TEXTURE;
		data_tex[0].viewNumSlices = 1;
		data_tex[0].surface.format = GX2_SURFACE_FORMAT_FLOAT_R32_G32_B32_A32;
		data_tex[0].compMap = GX2_COMP_SEL(_a, _r, _g, _b);
		GX2CalcSurfaceSizeAndAlignment(&data_tex[0].surface);
		GX2InitTextureRegs(&data_tex[0]);
		data_tex[0].surface.image = MEM2_alloc(data_tex[0].surface.imageSize, data_tex[0].surface.alignment);
		GX2Invalidate(GX2_INVALIDATE_MODE_CPU_TEXTURE, data_tex[0].surface.image, data_tex[0].surface.imageSize);

		if (!data_tex[0].surface.image) {
			INFO_LOG(G3D, "Failed to create GX2 texture for HW tessellation");
			return; // TODO: Turn off HW tessellation if texture creation error occured.
		}
		GX2SetVertexTexture(&data_tex[0], 0);
	}
	const u32 *src = (const u32 *)pos;
	u32 *dst = (u32 *)data_tex[0].surface.image;
	while (src < (u32 *)pos + size) {
		*dst++ = __builtin_bswap32(*src++);
	}

	// Texcoords
	if (hasTexCoords) {
		if (prevSizeTex < size) {
			prevSizeTex = size;
			MEM2_free(data_tex[1].surface.image);
			data_tex[1].surface.width = size;
			data_tex[1].surface.height = 1;
			data_tex[1].surface.depth = 1;
			data_tex[1].surface.dim = GX2_SURFACE_DIM_TEXTURE_1D;
			data_tex[1].surface.tileMode = GX2_TILE_MODE_LINEAR_ALIGNED;
			data_tex[1].surface.use = GX2_SURFACE_USE_TEXTURE;
			data_tex[1].viewNumSlices = 1;
			data_tex[1].surface.format = GX2_SURFACE_FORMAT_FLOAT_R32_G32_B32_A32;
			data_tex[1].compMap = GX2_COMP_SEL(_a, _r, _g, _b);
			GX2CalcSurfaceSizeAndAlignment(&data_tex[1].surface);
			GX2InitTextureRegs(&data_tex[1]);
			data_tex[1].surface.image = MEM2_alloc(data_tex[1].surface.imageSize, data_tex[1].surface.alignment);
			if (!data_tex[1].surface.image) {
				INFO_LOG(G3D, "Failed to create GX2 texture for HW tessellation");
				return; // TODO: Turn off HW tessellation if texture creation error occured.
			}
			GX2SetVertexTexture(&data_tex[1], 1);
		}
		src = (const u32 *)pos;
		dst = (u32 *)data_tex[1].surface.image;
		while (src < (u32 *)pos + size) {
			*dst++ = __builtin_bswap32(*src++);
		}
		GX2Invalidate(GX2_INVALIDATE_MODE_CPU_TEXTURE, data_tex[1].surface.image, data_tex[1].surface.imageSize);
	}

	// Color
	int sizeColor = hasColor ? size : 1;
	if (prevSizeCol < sizeColor) {
		prevSizeCol = sizeColor;
		MEM2_free(data_tex[2].surface.image);
		data_tex[2].surface.width = sizeColor;
		data_tex[2].surface.height = 1;
		data_tex[2].surface.depth = 1;
		data_tex[2].surface.dim = GX2_SURFACE_DIM_TEXTURE_1D;
		data_tex[2].surface.tileMode = GX2_TILE_MODE_LINEAR_ALIGNED;
		data_tex[2].surface.use = GX2_SURFACE_USE_TEXTURE;
		data_tex[2].viewNumSlices = 1;
		data_tex[2].surface.format = GX2_SURFACE_FORMAT_FLOAT_R32_G32_B32_A32;
		data_tex[2].compMap = GX2_COMP_SEL(_a, _r, _g, _b);
		GX2CalcSurfaceSizeAndAlignment(&data_tex[2].surface);
		GX2InitTextureRegs(&data_tex[2]);
		data_tex[2].surface.image = MEM2_alloc(data_tex[2].surface.imageSize, data_tex[2].surface.alignment);
		if (!data_tex[2].surface.image) {
			INFO_LOG(G3D, "Failed to create GX2 texture for HW tessellation");
			return; // TODO: Turn off HW tessellation if texture creation error occured.
		}
		GX2SetVertexTexture(&data_tex[2], 2);
	}
	src = (const u32 *)col;
	dst = (u32 *)data_tex[2].surface.image;
	while (src < (u32 *)pos + sizeColor) {
		*dst++ = __builtin_bswap32(*src++);
	}
	GX2Invalidate(GX2_INVALIDATE_MODE_CPU_TEXTURE, data_tex[2].surface.image, data_tex[2].surface.imageSize);
#endif
}
