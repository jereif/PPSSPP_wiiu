
#define GX2_COMP_SEL
#include "WiiU/GX2GraphicsContext.h"
#include "WiiU/WiiUHost.h"
#include "GPU/GX2/ShaderManagerGX2.h"
#include "Core/Host.h"

#include "Core/System.h"
#include "Common/GPU/thin3d.h"
#include "Common/GPU/thin3d_create.h"
#include "Common/System/NativeApp.h"
#include "Common/Input/InputState.h"
#include "Common/Profiler/Profiler.h"
#include "Common/TimeUtil.h"
#include "Common/Log.h"

#include <wiiu/gx2.h>
#include <wiiu/procui.h>
#include <wiiu/vpad.h>
#include <wiiu/os/memory.h>

static bool swap_is_pending(void *start_time) {
	uint32_t swap_count, flip_count;
	OSTime last_flip, last_vsync;

	GX2GetSwapStatus(&swap_count, &flip_count, &last_flip, &last_vsync);

	return last_vsync < *(OSTime *)start_time;
}

bool GX2GraphicsContext::Init() {
	static const RenderMode render_mode_map[] = {
		{ 0 },                                         /* GX2_TV_SCAN_MODE_NONE  */
		{ 854, 480, GX2_TV_RENDER_MODE_WIDE_480P },    /* GX2_TV_SCAN_MODE_576I  */
		{ 854, 480, GX2_TV_RENDER_MODE_WIDE_480P },    /* GX2_TV_SCAN_MODE_480I  */
		{ 854, 480, GX2_TV_RENDER_MODE_WIDE_480P },    /* GX2_TV_SCAN_MODE_480P  */
		{ 1280, 720, GX2_TV_RENDER_MODE_WIDE_720P },   /* GX2_TV_SCAN_MODE_720P  */
		{ 0 },                                         /* GX2_TV_SCAN_MODE_unk   */
		{ 1920, 1080, GX2_TV_RENDER_MODE_WIDE_1080P }, /* GX2_TV_SCAN_MODE_1080I */
		{ 1920, 1080, GX2_TV_RENDER_MODE_WIDE_1080P }  /* GX2_TV_SCAN_MODE_1080P */
	};
	render_mode_ = render_mode_map[GX2GetSystemTVScanMode()];
	render_mode_ = render_mode_map[GX2_TV_SCAN_MODE_480P];

	cmd_buffer_ = MEM2_alloc(0x400000, 0x40);
	u32 init_attributes[] = { GX2_INIT_CMD_BUF_BASE, (u32)cmd_buffer_, GX2_INIT_CMD_BUF_POOL_SIZE, 0x400000, GX2_INIT_ARGC, 0, GX2_INIT_ARGV, 0, GX2_INIT_END };
	GX2Init(init_attributes);
	u32 size = 0;
	u32 tmp = 0;
	GX2CalcTVSize(render_mode_.mode, GX2_SURFACE_FORMAT_UNORM_R8_G8_B8_A8, GX2_BUFFERING_MODE_DOUBLE, &size, &tmp);

	tv_scan_buffer_ = MEMBucket_alloc(size, GX2_SCAN_BUFFER_ALIGNMENT);
	if (tv_scan_buffer_) {
		memset(tv_scan_buffer_, 0, size);
		GX2Invalidate(GX2_INVALIDATE_MODE_CPU, tv_scan_buffer_, size);
	}
	GX2SetTVBuffer(tv_scan_buffer_, size, render_mode_.mode, GX2_SURFACE_FORMAT_UNORM_R8_G8_B8_A8, GX2_BUFFERING_MODE_DOUBLE);

	GX2CalcDRCSize(GX2_DRC_RENDER_MODE_SINGLE, GX2_SURFACE_FORMAT_UNORM_R8_G8_B8_A8, GX2_BUFFERING_MODE_DOUBLE, &size, &tmp);

	drc_scan_buffer_ = MEMBucket_alloc(size, GX2_SCAN_BUFFER_ALIGNMENT);
	if (drc_scan_buffer_) {
		memset(drc_scan_buffer_, 0, size);
		GX2Invalidate(GX2_INVALIDATE_MODE_CPU, drc_scan_buffer_, size);
	}
	GX2SetDRCBuffer(drc_scan_buffer_, size, GX2_DRC_RENDER_MODE_SINGLE, GX2_SURFACE_FORMAT_UNORM_R8_G8_B8_A8, GX2_BUFFERING_MODE_DOUBLE);

	color_buffer_.surface.dim = GX2_SURFACE_DIM_TEXTURE_2D;
	color_buffer_.surface.width = render_mode_.width;
	color_buffer_.surface.height = render_mode_.height;
	color_buffer_.surface.depth = 1;
	color_buffer_.surface.mipLevels = 1;
	color_buffer_.surface.format = GX2_SURFACE_FORMAT_UNORM_R8_G8_B8_A8;
	color_buffer_.surface.use = GX2_SURFACE_USE_TEXTURE_COLOR_BUFFER_TV;
	color_buffer_.viewNumSlices = 1;

	GX2CalcSurfaceSizeAndAlignment(&color_buffer_.surface);
	GX2InitColorBufferRegs(&color_buffer_);

	color_buffer_.surface.image = MEM1_alloc(color_buffer_.surface.imageSize, color_buffer_.surface.alignment);
	if (!color_buffer_.surface.image) {
		color_buffer_.surface.image = MEM2_alloc(color_buffer_.surface.imageSize, color_buffer_.surface.alignment);
	}
	if (color_buffer_.surface.image) {
		memset(color_buffer_.surface.image, 0, color_buffer_.surface.imageSize);
		GX2Invalidate(GX2_INVALIDATE_MODE_CPU, color_buffer_.surface.image, color_buffer_.surface.imageSize);
	}

	depth_buffer_.surface.dim = GX2_SURFACE_DIM_TEXTURE_2D;
	depth_buffer_.surface.width = render_mode_.width;
	depth_buffer_.surface.height = render_mode_.height;
	depth_buffer_.surface.depth = 1;
	depth_buffer_.surface.mipLevels = 1;
	depth_buffer_.surface.format = GX2_SURFACE_FORMAT_FLOAT_D24_S8;
	depth_buffer_.surface.use = GX2_SURFACE_USE_DEPTH_BUFFER;
	depth_buffer_.viewNumSlices = 1;

	GX2CalcSurfaceSizeAndAlignment(&depth_buffer_.surface);
	GX2InitDepthBufferRegs(&depth_buffer_);

	depth_buffer_.surface.image = MEM1_alloc(depth_buffer_.surface.imageSize, depth_buffer_.surface.alignment);
	if (!depth_buffer_.surface.image) {
		depth_buffer_.surface.image = MEM2_alloc(depth_buffer_.surface.imageSize, depth_buffer_.surface.alignment);
	}
	GX2Invalidate(GX2_INVALIDATE_MODE_CPU, depth_buffer_.surface.image, depth_buffer_.surface.imageSize);

	{
		FILE *f = fopen("sd:/ppsspp_boot.log", "a");
		if (f) {
			fprintf(f, "gx2: MEM1_avail=%u MEMBucket_avail=%u MEM2_avail=%u\n",
				MEM1_avail(), MEMBucket_avail(), MEM2_avail());
			fprintf(f, "gx2: tv=%p drc=%p color=%p depth=%p\n",
				tv_scan_buffer_, drc_scan_buffer_, color_buffer_.surface.image, depth_buffer_.surface.image);
			fprintf(f, "gx2: color %ux%u pitch=%u tile=%u size=%u align=%u\n",
				color_buffer_.surface.width, color_buffer_.surface.height,
				color_buffer_.surface.pitch, color_buffer_.surface.tileMode,
				color_buffer_.surface.imageSize, color_buffer_.surface.alignment);
			fclose(f);
		}
	}

	ctx_state_ = (GX2ContextState *)MEM2_alloc(sizeof(GX2ContextState), GX2_CONTEXT_STATE_ALIGNMENT);
	GX2SetupContextStateEx(ctx_state_, GX2_TRUE);

	GX2SetContextState(ctx_state_);
	GX2SetShaderMode(GX2_SHADER_MODE_UNIFORM_BLOCK);
	GX2SetRasterizerClipControlHalfZ(GX2_TRUE, GX2_TRUE, GX2_TRUE);
	GX2SetColorBuffer(&color_buffer_, GX2_RENDER_TARGET_0);
	GX2SetDepthBuffer(&depth_buffer_);
	GX2SetViewport(0.0f, 0.0f, color_buffer_.surface.width, color_buffer_.surface.height, 0.0f, 1.0f);
	GX2SetScissor(0, 0, color_buffer_.surface.width, color_buffer_.surface.height);
	GX2SetDepthOnlyControl(GX2_DISABLE, GX2_DISABLE, GX2_COMPARE_FUNC_ALWAYS);
	GX2SetColorControl(GX2_LOGIC_OP_COPY, 0xFF, GX2_DISABLE, GX2_ENABLE);
	GX2SetBlendControl(GX2_RENDER_TARGET_0, GX2_BLEND_MODE_SRC_ALPHA, GX2_BLEND_MODE_INV_SRC_ALPHA, GX2_BLEND_COMBINE_MODE_ADD, GX2_ENABLE, GX2_BLEND_MODE_SRC_ALPHA, GX2_BLEND_MODE_INV_SRC_ALPHA, GX2_BLEND_COMBINE_MODE_ADD);
	GX2SetCullOnlyControl(GX2_FRONT_FACE_CCW, GX2_DISABLE, GX2_DISABLE);

	GX2ClearColor(&color_buffer_, 0.0f, 0.0f, 0.0f, 1.0f);
	SwapBuffers();

	GX2SetTVEnable(GX2_ENABLE);
	GX2SetDRCEnable(GX2_ENABLE);

	draw_ = Draw::T3DCreateGX2Context(ctx_state_, &color_buffer_, &depth_buffer_);
	SetGPUBackend(GPUBackend::GX2);
	GX2SetSwapInterval(0);
	return draw_->CreatePresets();
}

void GX2GraphicsContext::Shutdown() {
	if (!draw_)
		return;
	in_shutdown_ = true;
	delete draw_;
	draw_ = nullptr;
	GX2ClearColor(&color_buffer_, 0.0f, 0.0f, 0.0f, 1.0f);
	SwapBuffers();
	GX2DrawDone();
	GX2Shutdown();

	GX2SetTVEnable(GX2_DISABLE);
	GX2SetDRCEnable(GX2_DISABLE);

	MEM2_free(ctx_state_);
	ctx_state_ = nullptr;
	MEM2_free(cmd_buffer_);
	cmd_buffer_ = nullptr;
	MEM1_free(color_buffer_.surface.image);
	color_buffer_ = {};
	MEM1_free(depth_buffer_.surface.image);
	depth_buffer_ = {};
	MEMBucket_free(tv_scan_buffer_);
	tv_scan_buffer_ = nullptr;
	MEMBucket_free(drc_scan_buffer_);
	drc_scan_buffer_ = nullptr;
}

void GX2GraphicsContext::SwapBuffers() {
	PROFILE_THIS_SCOPE("swap");
	static int swapCount = 0;
	swapCount++;
	if (host && swapCount > 1 && !in_shutdown_) {
		static_cast<WiiUHost *>(host)->PollInput();
	}
	GX2CopyColorBufferToScanBuffer(&color_buffer_, GX2_SCAN_TARGET_DRC);
	GX2CopyColorBufferToScanBuffer(&color_buffer_, GX2_SCAN_TARGET_TV);
	GX2SwapScanBuffers();
	GX2Flush();
	double t0 = time_now_d();
	GX2DrawDone();
	double t1 = time_now_d();
	GX2WaitForFlip();
	double t2 = time_now_d();
	{
		// Perf diagnostics: average frame time vs. GPU draw done vs. flip wait.
		static double lastSwap = 0.0, accFrame = 0.0, accDraw = 0.0, accFlip = 0.0;
		static int accCount = 0;
		if (lastSwap != 0.0) {
			accFrame += t2 - lastSwap;
			accDraw += t1 - t0;
			accFlip += t2 - t1;
			accCount++;
		}
		lastSwap = t2;
		if (accCount >= 300) {
			NOTICE_LOG(G3D, "perf: avg frame %.2f ms (%.1f fps), DrawDone %.2f ms, WaitFlip %.2f ms, draws %.1f (sw %.1f, psAll %.1f, vsFB %.1f)",
				accFrame * 1000.0 / accCount, accCount / accFrame,
				accDraw * 1000.0 / accCount, accFlip * 1000.0 / accCount,
				(double)g_gx2Perf.draws / accCount, (double)g_gx2Perf.swDraws / accCount,
				(double)g_gx2Perf.psAllDraws / accCount, (double)g_gx2Perf.vsFallbackDraws / accCount);
			accFrame = accDraw = accFlip = 0.0;
			accCount = 0;
			g_gx2Perf = {};
		}
	}
	GX2SetContextState(ctx_state_);
	GX2SetShaderMode(GX2_SHADER_MODE_UNIFORM_BLOCK);
	GX2SetRasterizerClipControlHalfZ(GX2_TRUE, GX2_TRUE, GX2_TRUE);
}
