# PPSSPP para Wii U (Experimental)

[English (README.md)](README.md)

Port experimental optimizado de **PPSSPP** (emulador de PlayStation Portable) para **Nintendo Wii U** (Aroma), con backend gráfico nativo **GX2** y recompilador JIT MIPS/VFPU para **PowerPC 750CL (Espresso)**.

> **Notas:**
> - Esta es una compilación experimental para la comunidad orientada a pruebas. El rendimiento y la compatibilidad varían según el juego y todavía pueden presentarse fallos gráficos menores en algunos títulos.
> - **Desarrollo asistido por IA:** Se utilizaron herramientas de IA para analizar cuellos de botella e implementar las optimizaciones del JIT y GX2 en Wii U.
> - **Estado del proyecto:** Este repositorio se entrega tal cual y probablemente ya no reciba más actualizaciones.

---

## Novedades y optimizaciones en Wii U

- **Arquitectura de 3 núcleos (Espresso PPC 750CL):**
  - **Core 2:** Ejecución de CPU MIPS + VFPU JIT (`PpcJit`).
  - **Core 1:** Procesamiento de comandos GE, caché/decodificación de vértices y renderizado GX2.
  - **Core 0:** Hilo paralelo lock-free de decodificación de vértices para draw calls grandes.
- **Mejoras en el JIT PowerPC 750CL:**
  - Aceleración nativa con **Paired Singles** para operaciones VFPU y skinning por software.
  - Cargas/escrituras FPU y VFPU en pipeline, fusión de múltiples `lw`/`sw` consecutivos, peepholes de shifts/saltos, retornos rápidos `dispatcherPCInR3` y rutas rápidas HLE mediante `Comp_ReplacementFunc`.
  - Compilación JIT adaptativa por niveles para aprovechar al máximo la caché RWX de Wii U.
- **Optimizaciones en el backend nativo GX2:**
  - Generación de vertex y pixel shaders R700 por hardware con caché de estados de pipeline GX2.
  - Subidas de UBO/vértices/índices en ring-buffer aceleradas con `dcbz` y `PushBufferGX2::EndPush` sin barreras `sync` por draw.
  - Decodificadores de vértices especializados sin ramas (`DecodeHWSkinnedT`, `DecodeSkinnedU8T`, `DecodeNonSkinnedT`) y empaquetado diferido de VBO/EBO.
- **Base de compatibilidad actualizada:**
  - Integración de reglas `compat.ini` de PPSSPP actual (incluyendo correcciones de framebuffer y hacks de framerate).

---

## Instalación (Aroma)

1. Descarga la última versión de `PPSSPP.rpx` desde **Releases**.
2. Copia `PPSSPP.rpx` y la carpeta `assets/` a tu tarjeta SD (por ejemplo en `sd:/wiiu/apps/ppsspp/`).
3. Coloca tus copias de juegos `.iso` / `.cso` de PSP en cualquier carpeta de la tarjeta SD (por ejemplo, `sd:/psp/isos/`).
4. Inicia **PPSSPP** desde el Menú de Wii U (Aroma).

No hace falta ningún archivo de BIOS de PSP: PPSSPP funciona mediante emulación de alto nivel (HLE).

---

## Ajustes recomendados en Wii U

- **Backend:** GX2 (por defecto)
- **Resolución de renderizado:** `1x PSP` (máximo rendimiento)
- **Renderizado con búfer (Buffered Rendering):** Activado (necesario en la mayoría de juegos 3D)
- **Transformación por hardware:** Activado
- **Skinning por software:** Desactivado (usa skinning por hardware en GX2)
- **Caché de vértices:** Activado
- **Salto de cuadros (Frameskipping):** Desactivado (o `1` / Auto en títulos 3D muy pesados)

---

## Juegos probados y compatibilidad

Consulta la hoja de compatibilidad de la comunidad y reporta ahí mismo los juegos que vayas probando (Game ID, FPS y notas gráficas/de audio):
**[Lista de compatibilidad PPSSPP Wii U (Google Sheets)](https://docs.google.com/spreadsheets/d/1xdpvZjI92Se57OirUxY-D7OrPpR9esePtuLJ4PZaUxA/edit?usp=sharing)**

| Juego | Estado | Rendimiento observado | Notas / Problemas conocidos |
|---|---|---|---|
| **Tekken 6** | Jugable | ~45–60 FPS | Renderizado de personajes/huesos corregido en GX2. |
| **Silent Hill: Shattered Memories** | Jugable | 45–60 FPS (25+ FPS en exteriores muy cargados) | Fluido en interiores y la mayoría de exteriores; las zonas exteriores más densas dependen de CPU/vértices. |
| **God of War: Ghost of Sparta** | Jugable | 38–51 FPS (28–38 FPS en combates de 950+ draws) | Algunos glitches en sprites 2D. |
| **God of War: Chains of Olympus** | Jugable | 30–45 FPS | Algunos glitches en sprites 2D. |
| **GTA: Liberty City Stories** | Jugable | 25–40 FPS | Glitch menor en la cabeza/malla del personaje. |
| **GTA: Chinatown Wars** | Jugable | 50–60 FPS | Funciona fluido. |

---

## Limitaciones conocidas

- **Tamaño de la caché JIT:** El espacio RWX de Wii U es de ~1.78 MB; juegos con muchísimo código ejecutable pueden vaciar y recompilar la caché JIT ocasionalmente.
- **Techo de hardware:** Espresso es una CPU PowerPC 750CL in-order de 3 núcleos a 1.24 GHz, por lo que escenas con más de 1,000 draw calls por frame o carga pesada de CPU no siempre mantendrán 60 FPS constantes.
- **Combinaciones raras de shaders GX2:** Algunas combinaciones poco frecuentes de pixel shaders de PSP (por ejemplo ciertos modos de color-test o logic-op) usan el shader genérico de respaldo.

---

## Compilar desde el código fuente

Requisitos: **devkitPPC** (devkitPro), **CMake**, **Ninja** y `rpltool`.

```bash
cmake -B build-wiiu -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchains/wiiu.cmake
ninja -C build-wiiu PPSSPP
```

---

## Créditos y licencia

- **PPSSPP** fue creado por **Henrik Rydgård** y sus colaboradores ([ppsspp.org](https://www.ppsspp.org/) | [github.com/hrydgard/ppsspp](https://github.com/hrydgard/ppsspp)).
- **Port base de Wii U:** Basado en el port de Wii U de **aliaspider** ([github.com/aliaspider/ppsspp/tree/wiiu](https://github.com/aliaspider/ppsspp/tree/wiiu)), que trajo la compatibilidad inicial de Wii U, GX2 y el JIT PowerPC.
- **Asistencia en optimizaciones y depuración en Wii U:** **Antigravity** (Google DeepMind).
- Compilado con el ecosistema homebrew de **devkitPro / libwiiu / GX2**.
- Partes del emulador incluyen definiciones y cabeceras de **PSPSDK** (licencia compatible con BSD).

Este proyecto se distribuye bajo la licencia **GNU General Public License v2.0 (o posterior)**. Consulta [LICENSE.TXT](LICENSE.TXT) para ver el texto completo.
