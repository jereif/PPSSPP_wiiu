# PPSSPP for Wii U (Experimental)

[Español (README_ES.md)](README_ES.md)

An optimized experimental port of **PPSSPP** (PlayStation Portable emulator) for **Nintendo Wii U** (Aroma), built around a native **GX2** graphics backend and a **PowerPC 750CL (Espresso)** MIPS/VFPU JIT recompiler.

> **Note:** This is an experimental community build intended for testing and tinkering. Performance and compatibility vary per game, and minor graphical issues may still occur in some titles.

---

## Highlights & Wii U Optimizations

- **3-Core Architecture (Espresso PPC 750CL):**
  - **Core 2:** MIPS CPU + VFPU JIT execution (`PpcJit`).
  - **Core 1:** GE command processing, vertex caching/decoding, and GX2 rendering.
  - **Core 0:** Lock-free parallel vertex decoding worker for large draw calls.
- **PowerPC 750CL JIT Improvements:**
  - Native **Paired Singles** acceleration for VFPU operations and software skinning.
  - Pipelined FPU/VFPU loads & stores, fused multi-`lw`/`sw` sequences, shift/branch peepholes, fast `dispatcherPCInR3` returns, and `Comp_ReplacementFunc` HLE fast-paths.
  - Adaptive tiered JIT compilation to maximize block density inside Wii U's RWX code cache.
- **Native GX2 Backend Optimizations:**
  - Hardware R700 vertex & pixel shader generation with cached GX2 pipeline states.
  - `dcbz`-accelerated ring-buffer UBO/vertex/index uploads and sync-free `PushBufferGX2::EndPush`.
  - Specialized zero-branch vertex decoders (`DecodeHWSkinnedT`, `DecodeSkinnedU8T`, `DecodeNonSkinnedT`) and deferred VBO/EBO packing.
- **Updated Compatibility Database:**
  - Merged upstream `compat.ini` game fixes (including framebuffer and framerate hacks).

---

## Installation (Aroma)

1. Download the latest `PPSSPP.rpx` from **Releases**.
2. Copy `PPSSPP.rpx` and the `assets/` folder to your SD card (for example `sd:/wiiu/apps/ppsspp/`).
3. Place your `.iso` / `.cso` PSP game backups anywhere on your SD card (e.g., `sd:/psp/isos/`).
4. Launch **PPSSPP** from the Wii U Aroma Home Menu.

No PSP BIOS file is required — PPSSPP uses High-Level Emulation (HLE).

---

## Recommended Settings on Wii U

- **Backend:** GX2 (default)
- **Rendering Resolution:** `1x PSP` (best speed)
- **Buffered Rendering:** On (required for most 3D games)
- **Hardware Transform:** On
- **Software Skinning:** Off (uses GX2 hardware skinning)
- **Vertex Cache:** On
- **Frameskipping:** Off (or `1` / Auto for heavier 3D titles)

---

## Tested Games & Compatibility

Full community compatibility sheet: [PPSSPP Wii U Compatibility List (Google Sheets)](https://docs.google.com/spreadsheets/d/1xdpvZjI92Se57OirUxY-D7OrPpR9esePtuLJ4PZaUxA/edit?usp=sharing)

| Game | Status | Observed Performance | Notes / Known Issues |
|---|---|---|---|
| **Tekken 6** | Playable | ~45–60 FPS | Character/bone rendering fixed in GX2. |
| **Silent Hill: Shattered Memories** | Playable | 45–60 FPS (25+ FPS in heaviest outdoor scenes) | Smooth in interiors and most exteriors; heavy outdoor scenes are CPU/vertex-bound. |
| **God of War: Ghost of Sparta** | Playable | 38–51 FPS (28–38 FPS in 950+ draw combat) | Some 2D sprite glitches. |
| **God of War: Chains of Olympus** | Playable | 30–45 FPS | Some 2D sprite glitches. |
| **GTA: Liberty City Stories** | Playable | 25–40 FPS | Minor character head/mesh glitch. |
| **GTA: Chinatown Wars** | Playable | 50–60 FPS | Runs smoothly. |

> Found a game that works well or has a specific bug? Feel free to open an Issue with the Game ID and your findings.

---

## Known Limitations

- **JIT Code Cache Size:** Wii U's RWX code space is ~1.78 MB; games with very large code footprints may occasionally flush and recompile the JIT cache.
- **Hardware Ceiling:** Espresso is a 3-core 1.24 GHz in-order PowerPC 750CL CPU, so games pushing 1,000+ draw calls per frame or heavy CPU workloads will not always hold a locked 60 FPS.
- **Minor GX2 Shader Gaps:** A few rare PSP pixel-shader combinations (e.g. certain color-test / logic-op paths) fall back to the generic shader.

---

## Building from Source

Requirements: **devkitPPC** (devkitPro), **CMake**, **Ninja**, and `rpltool`.

```bash
cmake -B build-wiiu -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchains/wiiu.cmake
ninja -C build-wiiu PPSSPP
```

---

## Credits & License

- **PPSSPP** was created by **Henrik Rydgård** and its many contributors ([ppsspp.org](https://www.ppsspp.org/) | [github.com/hrydgard/ppsspp](https://github.com/hrydgard/ppsspp)).
- **Wii U Base Port:** Based on **aliaspider**'s Wii U port ([github.com/aliaspider/ppsspp/tree/wiiu](https://github.com/aliaspider/ppsspp/tree/wiiu)), which brought initial Wii U / GX2 / PowerPC JIT compatibility.
- Built with the **devkitPro / libwiiu / GX2** homebrew ecosystem.
- Parts of the emulator include definitions/headers from **PSPSDK** (BSD-compatible license).

This project is licensed under the **GNU General Public License v2.0 (or later)**. See [LICENSE.TXT](LICENSE.TXT) for the full license text.
