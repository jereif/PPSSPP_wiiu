
#include <unistd.h>
#include <cmath>

#include <wiiu/os/cache.h>
#include <wiiu/os/foreground.h>
#include <wiiu/os/debug.h>
#include <wiiu/os/thread.h>
#include <wiiu/ax/core.h>
#include <wiiu/ax/multivoice.h>
#include <wiiu/vpad.h>
#include <wiiu/kpad.h>
#include <wiiu/procui.h>

#include "Common/Log.h"
#include "UI/OnScreenDisplay.h"
#include "Common/System/System.h"
#include "Common/System/NativeApp.h"
#include "Common/Input/InputState.h"

#include "WiiU/WiiUHost.h"
#include "WiiU/GX2GraphicsContext.h"

// The PSP analog stick is a square, while most physical sticks are circular.
// Stretch the vector so that the diagonals reach the corners like on the PSP.
static void StickRangeToSquare(float *x, float *y) {
	if (*x == 0.0f || *y == 0.0f) {
		return;
	}
	float ax = fabsf(*x);
	float ay = fabsf(*y);
	float phi = atanf(ay / ax);
	float scale = (ax > ay) ? 1.0f / cosf(phi) : 1.0f / sinf(phi);
	*x *= scale;
	*y *= scale;
}

static void VPADRangeToPSPRange(VPADVec2D *vec) {
	StickRangeToSquare(&vec->x, &vec->y);
}

// ---------------------------------------------------------------------------
// Wii U Pro Controller / Classic Controller support
//
// The GamePad is handled through VPAD, external controllers through KPAD. Both
// use different button bitmasks, so each controller type gets its own map.
// ---------------------------------------------------------------------------

struct PadButtonMapping {
	u32 bit;
	keycode_t key;
};

static const PadButtonMapping proButtonMap[] = {
	{ WPAD_PRO_BUTTON_A,       NKCODE_BUTTON_A },
	{ WPAD_PRO_BUTTON_B,       NKCODE_BUTTON_B },
	{ WPAD_PRO_BUTTON_X,       NKCODE_BUTTON_X },
	{ WPAD_PRO_BUTTON_Y,       NKCODE_BUTTON_Y },
	{ WPAD_PRO_BUTTON_UP,      NKCODE_DPAD_UP },
	{ WPAD_PRO_BUTTON_DOWN,    NKCODE_DPAD_DOWN },
	{ WPAD_PRO_BUTTON_LEFT,    NKCODE_DPAD_LEFT },
	{ WPAD_PRO_BUTTON_RIGHT,   NKCODE_DPAD_RIGHT },
	{ WPAD_PRO_TRIGGER_L,      NKCODE_BUTTON_L1 },
	{ WPAD_PRO_TRIGGER_R,      NKCODE_BUTTON_R1 },
	{ WPAD_PRO_TRIGGER_ZL,     NKCODE_BUTTON_L2 },
	{ WPAD_PRO_TRIGGER_ZR,     NKCODE_BUTTON_R2 },
	{ WPAD_PRO_BUTTON_PLUS,    NKCODE_BUTTON_START },
	{ WPAD_PRO_BUTTON_MINUS,   NKCODE_BUTTON_SELECT },
	{ WPAD_PRO_BUTTON_HOME,    NKCODE_HOME },
	{ WPAD_PRO_BUTTON_STICK_L, NKCODE_BUTTON_THUMBL },
	{ WPAD_PRO_BUTTON_STICK_R, NKCODE_BUTTON_THUMBR },
};

// Classic Controller triggers are analog, so we turn them into two fake bits
// that live above the real WPAD_CLASSIC_BUTTON_* values.
#define KPAD_FAKE_L2_BIT (1u << 30)
#define KPAD_FAKE_R2_BIT (1u << 29)

static const PadButtonMapping classicButtonMap[] = {
	{ WPAD_CLASSIC_BUTTON_A,     NKCODE_BUTTON_A },
	{ WPAD_CLASSIC_BUTTON_B,     NKCODE_BUTTON_B },
	{ WPAD_CLASSIC_BUTTON_X,     NKCODE_BUTTON_X },
	{ WPAD_CLASSIC_BUTTON_Y,     NKCODE_BUTTON_Y },
	{ WPAD_CLASSIC_BUTTON_UP,    NKCODE_DPAD_UP },
	{ WPAD_CLASSIC_BUTTON_DOWN,  NKCODE_DPAD_DOWN },
	{ WPAD_CLASSIC_BUTTON_LEFT,  NKCODE_DPAD_LEFT },
	{ WPAD_CLASSIC_BUTTON_RIGHT, NKCODE_DPAD_RIGHT },
	{ WPAD_CLASSIC_BUTTON_L,     NKCODE_BUTTON_L1 },
	{ WPAD_CLASSIC_BUTTON_R,     NKCODE_BUTTON_R1 },
	{ KPAD_FAKE_L2_BIT,          NKCODE_BUTTON_L2 },
	{ KPAD_FAKE_R2_BIT,          NKCODE_BUTTON_R2 },
	{ WPAD_CLASSIC_BUTTON_PLUS,  NKCODE_BUTTON_START },
	{ WPAD_CLASSIC_BUTTON_MINUS, NKCODE_BUTTON_SELECT },
	{ WPAD_CLASSIC_BUTTON_HOME,  NKCODE_HOME },
	{ WPAD_CLASSIC_BUTTON_ZL,    NKCODE_BUTTON_THUMBL },
	{ WPAD_CLASSIC_BUTTON_ZR,    NKCODE_BUTTON_THUMBR },
};

struct KPADPadState {
	u32 buttons;
	bool isClassic;
	float lx, ly, rx, ry;
};

static KPADPadState kpadState[4];

static void KPADSendKey(s32 chan, keycode_t key, bool down) {
	NativeKey({ DEVICE_ID_PAD_0 + chan, key, down ? KEY_DOWN : KEY_UP });
}

// Releases every button that we previously reported as pressed, so that no key
// stays stuck when a controller is disconnected or suspended.
static void KPADReleaseAll(s32 chan, KPADPadState &state) {
	if (!state.buttons)
		return;

	const PadButtonMapping *map = state.isClassic ? classicButtonMap : proButtonMap;
	size_t mapCount = state.isClassic ? countof(classicButtonMap) : countof(proButtonMap);
	for (size_t i = 0; i < mapCount; i++) {
		if (state.buttons & map[i].bit)
			KPADSendKey(chan, map[i].key, false);
	}
	state.buttons = 0;
}

static void KPADHandleChannel(s32 chan) {
	KPADData data;
	KPADPadState &state = kpadState[chan];

	s32 count = KPADRead(chan, &data, 1);
	bool isPro = count > 0 && data.device_type == WPAD_EXT_PRO_CONTROLLER;
	bool isClassic = count > 0 && (data.device_type == WPAD_EXT_CLASSIC || data.device_type == WPAD_EXT_MPLUS_CLASSIC);

	if (!isPro && !isClassic) {
		// No controller (or an unsupported one) on this channel.
		KPADReleaseAll(chan, state);
		return;
	}

	u32 buttons;
	float lx, ly, rx, ry;
	const PadButtonMapping *map;
	size_t mapCount;

	if (isPro) {
		buttons = data.pro.hold;
		lx = data.pro.lstick_x;
		ly = data.pro.lstick_y;
		rx = data.pro.rstick_x;
		ry = data.pro.rstick_y;
		map = proButtonMap;
		mapCount = countof(proButtonMap);
	} else {
		buttons = data.classic.btns_h;
		lx = data.classic.lstick_x;
		ly = data.classic.lstick_y;
		rx = data.classic.rstick_x;
		ry = data.classic.rstick_y;
		if (data.classic.ltrigger > 0.4f)
			buttons |= KPAD_FAKE_L2_BIT;
		if (data.classic.rtrigger > 0.4f)
			buttons |= KPAD_FAKE_R2_BIT;
		map = classicButtonMap;
		mapCount = countof(classicButtonMap);
	}

	// The controller type changed while keys were held: release the old ones.
	if (state.isClassic != isClassic)
		KPADReleaseAll(chan, state);

	u32 pressed = buttons & ~state.buttons;
	u32 released = state.buttons & ~buttons;
	state.buttons = buttons;
	state.isClassic = isClassic;

	for (size_t i = 0; i < mapCount; i++) {
		if (pressed & map[i].bit)
			KPADSendKey(chan, map[i].key, true);
		if (released & map[i].bit)
			KPADSendKey(chan, map[i].key, false);
	}

	if (state.lx != lx || state.ly != ly) {
		state.lx = lx;
		state.ly = ly;
		float x = lx, y = ly;
		StickRangeToSquare(&x, &y);
		NativeAxis({ DEVICE_ID_PAD_0 + chan, JOYSTICK_OUYA_AXIS_LS_X, x });
		NativeAxis({ DEVICE_ID_PAD_0 + chan, JOYSTICK_OUYA_AXIS_LS_Y, y });
	}
	if (state.rx != rx || state.ry != ry) {
		state.rx = rx;
		state.ry = ry;
		float x = rx, y = ry;
		StickRangeToSquare(&x, &y);
		NativeAxis({ DEVICE_ID_PAD_0 + chan, JOYSTICK_OUYA_AXIS_RS_X, x });
		NativeAxis({ DEVICE_ID_PAD_0 + chan, JOYSTICK_OUYA_AXIS_RS_Y, y });
	}

	// ZL+ZR both pressed on an external controller also quits.
	u32 zlBit = isClassic ? KPAD_FAKE_L2_BIT : WPAD_PRO_TRIGGER_ZL;
	u32 zrBit = isClassic ? KPAD_FAKE_R2_BIT : WPAD_PRO_TRIGGER_ZR;
	if ((buttons & zlBit) && (buttons & zrBit))
		System_SendMessage("finish", "");
}

static void KPADPollAll() {
	for (s32 chan = 0; chan < 4; chan++)
		KPADHandleChannel(chan);
}

static void PollVPADAndKPAD(s32 chan) {
	static keycode_t keymap[] = {
		NKCODE_UNKNOWN,       // VPAD_BUTTON_SYNC_BIT             = 0,
		NKCODE_HOME,          // VPAD_BUTTON_HOME_BIT             = 1,
		NKCODE_BUTTON_SELECT, // VPAD_BUTTON_MINUS_BIT            = 2,
		NKCODE_BUTTON_START,  // VPAD_BUTTON_PLUS_BIT             = 3,
		NKCODE_BUTTON_R1,     // VPAD_BUTTON_R_BIT                = 4,
		NKCODE_BUTTON_L1,     // VPAD_BUTTON_L_BIT                = 5,
		NKCODE_BUTTON_R2,     // VPAD_BUTTON_ZR_BIT               = 6,
		NKCODE_BUTTON_L2,     // VPAD_BUTTON_ZL_BIT               = 7,
		NKCODE_DPAD_DOWN,     // VPAD_BUTTON_DOWN_BIT             = 8,
		NKCODE_DPAD_UP,       // VPAD_BUTTON_UP_BIT               = 9,
		NKCODE_DPAD_RIGHT,    // VPAD_BUTTON_RIGHT_BIT            = 10,
		NKCODE_DPAD_LEFT,     // VPAD_BUTTON_LEFT_BIT             = 11,
		NKCODE_BUTTON_Y,      // VPAD_BUTTON_Y_BIT                = 12,
		NKCODE_BUTTON_X,      // VPAD_BUTTON_X_BIT                = 13,
		NKCODE_BUTTON_B,      // VPAD_BUTTON_B_BIT                = 14,
		NKCODE_BUTTON_A,      // VPAD_BUTTON_A_BIT                = 15,
		NKCODE_UNKNOWN,       // VPAD_BUTTON_TV_BIT               = 16,
		NKCODE_BUTTON_THUMBR, // VPAD_BUTTON_STICK_R_BIT          = 17,
		NKCODE_BUTTON_THUMBL, // VPAD_BUTTON_STICK_L_BIT          = 18,
		NKCODE_UNKNOWN,       // VPAD_BUTTON_TOUCH_BIT            = 19,
		NKCODE_UNKNOWN,       // VPAD_BUTTON_UNUSED1_BIT          = 20,
		NKCODE_UNKNOWN,       // VPAD_BUTTON_UNUSED2_BIT          = 21,
		NKCODE_UNKNOWN,       // VPAD_BUTTON_UNUSED3_BIT          = 22,
		NKCODE_UNKNOWN,       // VPAD_STICK_R_EMULATION_DOWN_BIT  = 23,
		NKCODE_UNKNOWN,       // VPAD_STICK_R_EMULATION_UP_BIT    = 24,
		NKCODE_UNKNOWN,       // VPAD_STICK_R_EMULATION_RIGHT_BIT = 25,
		NKCODE_UNKNOWN,       // VPAD_STICK_R_EMULATION_LEFT_BIT  = 26,
		NKCODE_UNKNOWN,       // VPAD_STICK_L_EMULATION_DOWN_BIT  = 27,
		NKCODE_UNKNOWN,       // VPAD_STICK_L_EMULATION_UP_BIT    = 28,
		NKCODE_UNKNOWN,       // VPAD_STICK_L_EMULATION_RIGHT_BIT = 29,
		NKCODE_UNKNOWN,       // VPAD_STICK_L_EMULATION_LEFT_BIT  = 30,
	};

	VPADStatus vpad;
	VPADReadError readError;
	VPADRead(chan, &vpad, 1, &readError);

	if (!readError) {
		static int touchflags;
		if (vpad.tpFiltered1.validity != VPAD_VALID) {
			vpad.tpFiltered1.touched = false;
		}
		if (touchflags == TOUCH_DOWN || touchflags == TOUCH_MOVE) {
			touchflags = vpad.tpFiltered1.touched ? TOUCH_MOVE : TOUCH_UP;
		} else {
			touchflags = vpad.tpFiltered1.touched ? TOUCH_DOWN : 0;
		}
		if (touchflags) {
			VPADTouchData calibrated;
			VPADGetTPCalibratedPointEx(chan, VPAD_TOUCH_RESOLUTION_854x480, &calibrated, &vpad.tpFiltered1);
			NativeTouch({ (float)calibrated.x, (float)calibrated.y, 0, touchflags });
		}
		for (int i = 0; i < countof(keymap); i++) {
			if (keymap[i] == NKCODE_UNKNOWN)
				continue;

			if ((vpad.trigger | vpad.release) & (1 << i)) {
				NativeKey({ DEVICE_ID_PAD_0 + chan, keymap[i], (vpad.trigger & (1 << i)) ? KEY_DOWN : KEY_UP });
			}
		}

		static VPADVec2D prevLeftStick, prevRightStick;
		if (prevLeftStick.x != vpad.leftStick.x || prevLeftStick.y != vpad.leftStick.y) {
			prevLeftStick = vpad.leftStick;
			VPADRangeToPSPRange(&vpad.leftStick);
			NativeAxis({ DEVICE_ID_PAD_0 + chan, JOYSTICK_OUYA_AXIS_LS_X, vpad.leftStick.x });
			NativeAxis({ DEVICE_ID_PAD_0 + chan, JOYSTICK_OUYA_AXIS_LS_Y, vpad.leftStick.y });
		}
		if (prevRightStick.x != vpad.rightStick.x || prevRightStick.y != vpad.rightStick.y) {
			prevRightStick = vpad.rightStick;
			VPADRangeToPSPRange(&vpad.rightStick);
			NativeAxis({ DEVICE_ID_PAD_0 + chan, JOYSTICK_OUYA_AXIS_RS_X, vpad.rightStick.x });
			NativeAxis({ DEVICE_ID_PAD_0 + chan, JOYSTICK_OUYA_AXIS_RS_Y, vpad.rightStick.y });
		}
#if 1
		if (vpad.trigger & VPAD_BUTTON_STICK_L) {
			extern bool g_TakeScreenshot;
			g_TakeScreenshot = true;
		}
#endif
	}

	// KPAD has no sampling callback available in this runtime, so poll the
	// external controllers from here. This runs at the GamePad sampling rate.
	KPADPollAll();
}

static void SaveCallback(void) { OSSavesDone_ReadyToRelease(); }

WiiUHost::WiiUHost() {
	ProcUIInit(&SaveCallback);
	VPADInit();
	WPADEnableURCC(true);
	WPADEnableWiiRemote(true);
	KPADInit();
}

WiiUHost::~WiiUHost() {
}

void WiiUHost::PollInput() {
	PollVPADAndKPAD(0);
}

bool WiiUHost::InitGraphics(std::string *error_message, GraphicsContext **ctx) {
	if (ctx_.Init()) {
		*ctx = &ctx_;
		return true;
	} else {
		*ctx = nullptr;
		return false;
	}
}

void WiiUHost::ShutdownGraphics() { ctx_.Shutdown(); }

#define AX_FRAMES 2
static_assert(!(AX_FRAMES & (AX_FRAMES - 1)), "AX_FRAMES must be a power of two");

static AXMVoice *mvoice;
static s16 __attribute__((aligned(64))) axBuffers[2][AX_FRAMES][AX_FRAME_SIZE];

static void AXCallback() {
	static s16 mixBuffer[AX_FRAME_SIZE * 2];
	static int pos;
#if 1
	AXVoiceOffsets offsets;
	AXGetVoiceOffsets(mvoice->v[0], &offsets);
	if ((offsets.currentOffset / AX_FRAME_SIZE) == pos) {
		pos = ((offsets.currentOffset / AX_FRAME_SIZE) + (AX_FRAMES >> 1)) & (AX_FRAMES - 1);
	}
#endif

	int count = NativeMix(mixBuffer, AX_FRAME_SIZE);
	int extra = AX_FRAME_SIZE - count;

	const s16 *src = mixBuffer;
	s16 *dst_l = axBuffers[0][pos];
	s16 *dst_r = axBuffers[1][pos];

	while (count--) {
		*dst_l++ = *src++;
		*dst_r++ = *src++;
	}
	while (extra--) {
		*dst_l++ = 0;
		*dst_r++ = 0;
	}

	DCStoreRangeNoSync(axBuffers[0][pos], AX_FRAME_SIZE * sizeof(s16));
	DCStoreRangeNoSync(axBuffers[1][pos], AX_FRAME_SIZE * sizeof(s16));

	pos++;
	pos &= AX_FRAMES - 1;
}

void WiiUHost::InitSound() {
	if (InitSoundRefCount_++)
		return;

	AXInitParams initParams = { AX_INIT_RENDERER_48KHZ };
	AXInitWithParams(&initParams);

	AXMVoiceParams mVoiceParams = {};
	mVoiceParams.count = 2;
	AXAcquireMultiVoice(31, NULL, 0, &mVoiceParams, &mvoice);

	if (mvoice && mvoice->channels == 2) {
		AXVoiceOffsets offsets[2];
		offsets[0].currentOffset = AX_FRAME_SIZE;
		offsets[0].loopOffset = 0;
		offsets[0].endOffset = (AX_FRAMES * AX_FRAME_SIZE) - 1;
		offsets[0].loopingEnabled = AX_VOICE_LOOP_ENABLED;
		offsets[0].dataType = AX_VOICE_FORMAT_LPCM16;
		offsets[0].data = axBuffers[0];

		offsets[1] = offsets[0];
		offsets[1].data = axBuffers[1];
		AXSetMultiVoiceOffsets(mvoice, offsets);

		AXSetMultiVoiceSrcType(mvoice, AX_VOICE_SRC_TYPE_NONE);
		AXSetMultiVoiceSrcRatio(mvoice, 1.0f);
		AXVoiceVeData ve = { 0x8000, 0 };
		AXSetMultiVoiceVe(mvoice, &ve);

		AXSetMultiVoiceDeviceMix(mvoice, AX_DEVICE_TYPE_DRC, 0, 0, 0x8000, 0);
		AXSetMultiVoiceDeviceMix(mvoice, AX_DEVICE_TYPE_TV, 0, 0, 0x8000, 0);
	}

	AXRegisterFrameCallback(AXCallback);
	AXSetMultiVoiceState(mvoice, AX_VOICE_STATE_PLAYING);
}

void WiiUHost::ShutdownSound() {
	if (--InitSoundRefCount_)
		return;

	AXSetMultiVoiceState(mvoice, AX_VOICE_STATE_STOPPED);
	AXRegisterFrameCallback(NULL);

	AXFreeMultiVoice(mvoice);
	AXQuit();
}

void WiiUHost::NotifyUserMessage(const std::string &message, float duration, u32 color, const char *id) { osm.Show(message, duration, color, -1, true, id); }

void WiiUHost::SendUIMessage(const std::string &message, const std::string &value) { NativeMessageReceived(message.c_str(), value.c_str()); }
