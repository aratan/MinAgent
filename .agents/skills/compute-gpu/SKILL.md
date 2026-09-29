---
name: compute-gpu
description: Genera voz, vídeo y música en la GPU local con una política de VRAM para una tarjeta de 8 GB. Úsalo cuando pidan hablar, transcribir un audio, generar un vídeo o una música, o cuando una generación falle por falta de memoria en VRAM. Install and drive the Kokoro/whisper.cpp/LTX-Video/MusicGen backends.
allowed-tools:
  - speak_text
  - transcribe_audio
  - generate_video
  - generate_music
  - compute_status
---

# Cómputo en GPU

Cuatro herramientas locales: hablar, transcribir, generar vídeo y generar música. Todas
comparten una política de VRAM, que es el motivo de que estén separadas del resto.

## La restricción física

La RTX 4060 Laptop tiene **8188 MiB de VRAM**. Dos modelos pesados a la vez no son una
configuración lenta: es imposible. La política que implementa `compute` es:

- **La voz se queda resident.** Kokoro (TTS) y whisper.cpp (STT) ocupan menos de 2.5 GB
  juntos y se cargan una vez. Responden al instante y **siguen funcionando mientras se
  genera un vídeo**: no esperes a que termine un render para hablar.
- **Los trabajos pesados van de uno en uno.** Hay un lock: un vídeo y una canción nunca
  compiten por la tarjeta. El segundo espera.
- **Cada trabajo pesado corre en un subproceso** con la RAM del sistema (30 GB) como
  almacén de capas, en vez de mantener los pesos en VRAM.
- **La VRAM libre se mide, no se supone.** Otro programa puede tener la tarjeta casi
  llena. En esta máquina el servidor de Ollama llegó a tener **6390 de 8188 MiB**, que es
  justo la diferencia entre un trabajo que corre y uno que revienta.

## Antes de un trabajo pesado, si falló por memoria

Un trabajo que no cabe **se rechaza antes de empezar**, con los números dentro del error.
No se intenta y se revienta a mitad. Cuando pase:

1. Llama a `compute_status`. Nombra el proceso que tiene la memoria.
2. Libera esa memoria, **o** reduce el trabajo:
   - **Vídeo**: menos fotogramas. `frames: 17` o `25` en vez de `49`.
     **Los fotogramas son los que mandan en la VRAM, no los pasos.** Bajar los pasos
     ahorra tiempo, no memoria.
   - **Vídeo**: `offload: "sequential"` es el que menos VRAM pide, y el más lento.
   - **Música**: `--modelo small` es el único que cabe junto a la voz. `medium` y `large`
     necesitan 6.5 GB y 9.5 GB, y esta tarjeta no los tiene.
3. **No repitas el mismo trabajo sin cambios.** Falla igual y cuesta minutos.

## Los fotogramas del vídeo

LTX-Video espera fotogramas de la forma **8k+1**: 9, 17, 25, 49, 97. Cualquier otro número
no es "un vídeo más corto", es un vídeo roto, así que el script lo ajusta y avisa. Los
valores que funcionan:

| frames | Cuándo |
|--------|--------|
| 17 | prueba rápida, para ver si el prompt funciona |
| 25 | corto y razonable |
| 49 | por defecto, ~2 s por clip a 97 pasos |
| 97 | calidad alta, mucho más tiempo y más VRAM |

## Instalación

Ninguna de estas dependencias es obligatoria para el resto del agente. Sin ellas, las
llamadas fallan con un mensaje que dice exactamente qué instalar.

**Voz (Kokoro):**

```bash
uv pip install kokoro soundfile numpy
```

Los pesos (~300 MB) se descargan de HuggingFace la primera vez, al sintetizar. Es
deliberado: la descarga es algo que alguien lanzó a sabiendas, no una sorpresa al hablar.

**STT (whisper.cpp):**

```bash
# Arch
sudo pacman -S whisper.cpp
```

El binario se llama `whisper-cli`. Si no está en el `PATH`, compílalo de
<https://github.com/g/ggerganov/whisper.cpp>.

Modelo por defecto: `base`. `tiny` se equivoca tanto que no sirve; `small` es mejor pero
sube el gasto de la reserva de voz. Para los pesos:

```bash
mkdir -p ~/.cache/whisper && cd ~/.cache/whisper
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
export WHISPER_MODEL_DIR=~/.cache/whisper
```

**Vídeo y música (diffusers):**

```bash
uv pip install torch diffusers transformers accelerate imageio imageio-ffmpeg scipy
```

Los pesos se descargan de HuggingFace la primera vez: LTX-Video son ~17 GB y MusicGen small
~2 GB. Es una descarga larga la primera vez; después van desde la caché.

## Salida

Todo lo generado cae en `salida/`, como las descargas. Las herramientas devuelven **una
ruta, no el medio**: un mp4 no se puede meter en el contexto, y la ruta sí se puede volver
a pasar a otra herramienta.

## Ejemplos

- "dime en voz alta que ya está listo" → `speak_text`
- "transcribe salida/audio.mp3" → `transcribe_audio`
- "haz un vídeo de un gato con gafas de sol en la playa" → `generate_video`
- "pon música lo-fi tranquila de fondo" → `generate_music`
- "no hay memoria, ¿qué está ocupando la tarjeta?" → `compute_status`

## También como servidor MCP

La misma política está expuesta en `.agents/mcp/compute/index.py`, para que otro cliente
MCP pueda generar con el mismo presupuesto de VRAM. Si lo añades a `.minagent/mcp.json`:

```json
{
  "mcpServers": {
    "compute": {
      "command": "python3",
      "args": [".agents/mcp/compute/index.py"],
      "cwd": "/ruta/al/proyecto"
    }
  }
}
```

No reinstala ni reimplementa nada: delega en el mismo `minagent.compute`, de modo que un
vídeo lanzado por MCP y otro lanzado por el agente no pueden planificarse como si la
tarjeta tuviera sitio para los dos.
