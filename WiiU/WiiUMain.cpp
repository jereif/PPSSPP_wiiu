#include <string>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <fstream>

#include <wiiu/os/systeminfo.h>
#include <wiiu/os/thread.h>
#include <wiiu/os/debug.h>
#include <wiiu/gx2/event.h>
#include <wiiu/ax/core.h>

#include <wiiu/ios.h>

#include "Common/Profiler/Profiler.h"
#include "Common/System/System.h"
#include "Common/System/NativeApp.h"
#include "Common/System/Display.h"
#include "Core/Core.h"
#include "Common/Log.h"

#include "Common/GraphicsContext.h"
#include "WiiU/WiiUHost.h"

const char *PROGRAM_NAME = "PPSSPP";
const char *PROGRAM_VERSION = PPSSPP_GIT_VERSION;

// The app can live either in the classic sd:/ppsspp/ directory or in the Aroma
// homebrew app directory. Pick whichever one actually contains the assets.
static std::string g_app_dir = "sd:/ppsspp/";

static bool AppDirHasAssets(const char *dir) {
	FILE *f = fopen((std::string(dir) + "assets/langregion.ini").c_str(), "rb");
	if (!f)
		return false;
	fclose(f);
	return true;
}

static void PickAppDir() {
	static const char *candidates[] = {
		"sd:/wiiu/apps/PPSSPP/",
		"sd:/ppsspp/",
	};
	for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
		if (AppDirHasAssets(candidates[i])) {
			g_app_dir = candidates[i];
			break;
		}
	}
	chdir(g_app_dir.c_str());
}

extern bool g_gx2NoLighting;
static int g_QuitRequested;
void System_SendMessage(const char *command, const char *parameter) {
	if (!strcmp(command, "finish")) {
		g_QuitRequested = true;
		UpdateUIState(UISTATE_EXIT);
		Core_Stop();
	}
}

static void BootLog(const char *msg) {
	FILE *f = fopen("sd:/ppsspp_boot.log", "a");
	if (f) {
		fprintf(f, "%s\n", msg);
		fclose(f);
	}
}

#include <cstdarg>
extern "C" void WiiUTrace(const char *fmt, ...) {
	char buf[512];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	BootLog(buf);
}

int main(int argc, char **argv) {
	BootLog("main: start");
	PROFILE_INIT();
	PPCSetFpIEEEMode();

	PickAppDir();
	{
		FILE *nl_ = fopen((g_app_dir + "nolight.txt").c_str(), "rb");
		if (nl_) { g_gx2NoLighting = true; fclose(nl_); }
	}
	BootLog(("main: g_app_dir=" + g_app_dir).c_str());

	host = new WiiUHost();
	BootLog("main: WiiUHost created");

	std::string app_name;
	std::string app_name_nice;
	std::string version;
	bool landscape;
	NativeGetAppInfo(&app_name, &app_name_nice, &landscape, &version);

	std::string app_path = g_app_dir + "PPSSPP.rpx";
	// Optional: if <app dir>/autoboot.txt exists, its first line is a game path to boot directly (for testing).
	std::string autoboot;
	{
		std::ifstream in(g_app_dir + "autoboot.txt");
		if (in) {
			std::getline(in, autoboot);
			while (!autoboot.empty() && (autoboot.back() == '\r' || autoboot.back() == '\n' || autoboot.back() == ' '))
				autoboot.pop_back();
		}
	}
	const char *argv_[] = {
		app_path.c_str(),
		"--log=sd:/ppsspp.log",
		"-v",
		autoboot.empty() ? nullptr : autoboot.c_str(),
		nullptr
	};
	NativeInit(autoboot.empty() ? 3 : 4, argv_, g_app_dir.c_str(), g_app_dir.c_str(), nullptr);
	BootLog("main: NativeInit done");
#if 0
	UpdateScreenScale(854,480);
#else
	float dpi_scale = 1.0f;
	g_dpi = 96.0f;
	pixel_xres = 854;
	pixel_yres = 480;
	dp_xres = (float)pixel_xres * dpi_scale;
	dp_yres = (float)pixel_yres * dpi_scale;
	pixel_in_dps_x = (float)pixel_xres / dp_xres;
	pixel_in_dps_y = (float)pixel_yres / dp_yres;
	g_dpi_scale_x = dp_xres / (float)pixel_xres;
	g_dpi_scale_y = dp_yres / (float)pixel_yres;
	g_dpi_scale_real_x = g_dpi_scale_x;
	g_dpi_scale_real_y = g_dpi_scale_y;
#endif
	printf("Pixels: %i x %i\n", pixel_xres, pixel_yres);
	printf("Virtual pixels: %i x %i\n", dp_xres, dp_yres);

	g_Config.iPSPModel = PSP_MODEL_SLIM;
	g_Config.iGPUBackend = (int)GPUBackend::GX2;
	g_Config.bEnableSound = true;
	g_Config.bPauseExitsEmulator = false;
	g_Config.bPauseMenuExitsEmulator = false;
//	g_Config.iCpuCore = (int)CPUCore::JIT;
	g_Config.bVertexDecoderJit = false;
	g_Config.bSoftwareRendering = false;
//	g_Config.iFpsLimit = 0;
	g_Config.bHardwareTransform = true;
	g_Config.bSoftwareSkinning = false;
	g_Config.bVertexCache = true;
//	PSP_CoreParameter().gpuCore = GPUCORE_NULL;
	g_Config.bTextureBackoffCache = true;
	std::string error_string;
	GraphicsContext *ctx;
	bool gfxOk = host->InitGraphics(&error_string, &ctx);
	BootLog(gfxOk ? "main: InitGraphics ok" : "main: InitGraphics FAILED");
	NativeInitGraphics(ctx);
	BootLog("main: NativeInitGraphics done");
	NativeResized();

	host->InitSound();
	BootLog("main: InitSound done, entering loop");
	bool firstFrame = true;
	while (true) {
		if (g_QuitRequested)
			break;

		if (!Core_IsActive())
			UpdateUIState(UISTATE_MENU);
		Core_Run(ctx);
		if (firstFrame) {
			BootLog("main: first Core_Run completed");
			firstFrame = false;
		}
	}
	BootLog("main: exited loop");
	host->ShutdownSound();

	NativeShutdownGraphics();
	NativeShutdown();

	return 0;
}

std::string System_GetProperty(SystemProperty prop) {
	switch (prop) {
	case SYSPROP_NAME:
		return "Wii-U";
	case SYSPROP_LANGREGION:
		return "en_US";
	default:
		return "";
	}
}

int System_GetPropertyInt(SystemProperty prop) {
	switch (prop) {
	case SYSPROP_DISPLAY_REFRESH_RATE:
		return 60000; // internal refresh rate is always 59.940, even for PAL output.
	case SYSPROP_DISPLAY_XRES:
		return 854;
	case SYSPROP_DISPLAY_YRES:
		return 480;
	case SYSPROP_DEVICE_TYPE:
		return DEVICE_TYPE_TV;
	case SYSPROP_AUDIO_SAMPLE_RATE:
	case SYSPROP_AUDIO_OPTIMAL_SAMPLE_RATE:
		return 48000;
	case SYSPROP_AUDIO_FRAMES_PER_BUFFER:
	case SYSPROP_AUDIO_OPTIMAL_FRAMES_PER_BUFFER:
		return AX_FRAME_SIZE;
	case SYSPROP_SYSTEMVERSION:
	default:
		return -1;
	}
}
bool System_GetPropertyBool(SystemProperty prop) {
	switch (prop) {
	case SYSPROP_APP_GOLD:
#ifdef GOLD
		return true;
#else
		return false;
#endif
	default:
		return false;
	}
}

float System_GetPropertyFloat(SystemProperty prop) {
	switch (prop) {
	case SYSPROP_DISPLAY_REFRESH_RATE:
		return 60.0f;
	case SYSPROP_DISPLAY_SAFE_INSET_LEFT:
	case SYSPROP_DISPLAY_SAFE_INSET_RIGHT:
	case SYSPROP_DISPLAY_SAFE_INSET_TOP:
	case SYSPROP_DISPLAY_SAFE_INSET_BOTTOM:
		return 0.0f;
	default:
		return -1;
	}
}

void System_AskForPermission(SystemPermission permission) {}
PermissionStatus System_GetPermissionStatus(SystemPermission permission) { return PERMISSION_STATUS_GRANTED; }

void LaunchBrowser(const char *url) {}
void ShowKeyboard() {}
void Vibrate(int length_ms) {}
