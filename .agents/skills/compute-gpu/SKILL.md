---
name: compute-gpu
description: Genera voz, vídeo y música en la GPU local con una política de VRAM para una tarjeta de 8 GB. Úsalo cuando pidan hablar, transcribir un audio, generar un vídeo o una música, o cuando una generación falle por falta de memoria en VRAM. Install and drive the Kokoro/whisper.cpp/LTX-Video/MusicGen backends.
allowed-tools:
  - speak_text
  - transcribe_audio
  - generate_video
  - generate_music
  - queue_job
  - compute_result
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

## Varios trabajos: encola, no esperes

Un render son minutos. Si el usuario pide tres vídeos, llamar a `generate_video` tres veces
seguidas quema una hora de turno sin decir nada. En su lugar:

1. `queue_job` para cada uno. Devuelve un id al instante (`job-1`, `job-2`...).
2. `compute_status` para ver las posiciones.
3. `compute_result` con un id para recoger cada resultado cuando termine.

Un trabajo pesado entra por la misma cola llegue como llegue, así que uno encolado y uno pedido
directamente siguen compartiendo la tarjeta de uno en uno. La cola está limitada
(`COMPUTE_QUEUE_LIMIT`): una lista sin techo de trabajos de minutos no es una función.

Un trabajo que de momento no cabe **se encola, no se rechaza**: lo que suele ocupar la tarjeta es
otro trabajo igual, y rechazar el segundo vídeo de un encargo de dos sería un fallo disfrazado de
política. La decisión de VRAM se toma al llegar a la cabeza de la cola.

## Antes de un trabajo pesado, si falló por memoria

Un trabajo que sigue sin caber **se rechaza antes de empezar**, con los números dentro del error.
No se intenta y se revienta a mitad. Cuando pase:

1. Llama a `compute_status`. Nombra el proceso que tiene la memoria.
2. Libera esa memoria, **o** reduce el trabajo:
   - **Vídeo**: menos fotogramas. `frames: 17` o `25` en vez de `49`.
     **Los fotogramas son los que mandan en la VRAM, no los pasos.** Bajar los pasos
     ahorra tiempo, no memoria.
   - **Vídeo**: `offload: "sequential"` es el que menos VRAM pide, y el más lento.
   - **Música**: el modelo `music` es el único que cabe junto a la voz. `base` y `full`
     traen vocoders y necesitan 4.6 GB y 6.2 GB, y esta tarjeta no los tiene.
3. **No repitas el mismo trabajo sin cambios.** Falla igual y cuesta minutos.

### El servidor de Ollama

Con `COMPUTE_UNLOAD_OLLAMA=on` (por defecto **off**), un trabajo que no cabe ejecuta
`ollama stop` antes, lo que caduca el *keep-alive* de un modelo resident. Está apagado a
propósito: ese modelo se recarga en su siguiente petición, y decidir eso le corresponde a quien
está delante de la pantalla, no al agente. Con la opción apagada, el rechazo nombra el proceso y
no toca nada.

## Los fotogramas del vídeo

LTX-Video espera fotogramas de la forma **8k+1**: 9, 17, 25, 49, 97. Cualquier otro número
no es "un vídeo más corto", es un vídeo roto, así que el script lo ajusta y avisa. Los
valores que funcionan:

| frames | Cuándo |
|--------|--------|
| 17 | prueba rápida, para ver si el prompt funciona (~1 min medido) |
| 25 | corto y razonable |
| 49 | por defecto, bastante más tiempo |
| 97 | calidad alta, mucho más tiempo y más VRAM |

## El modo de offload, y por qué el defecto es `sequential`

`sequential` baja un submódulo entero a RAM cada vez. `group` deja unas capas en la tarjeta y
baja el resto por bloques, y es más rápido, pero **no cabe aquí**: medido, `group` revienta
mientras `sequential` peaked en 696 MiB. La diferencia no es la velocidad, es que uno sale y el
otro no.

El modo por defecto tiene que ser el que sale en esta máquina, no el que sería más rápido en una
tarjeta vacía.

## Música

El backend es **AudioLDM2**, no MusicGen: MusicGen se retiró de diffusers y
`from diffusers import MusicgenPipeline` ya falla. Usa el modelo `music`, el más ligero; `base` y
`full` traen vocoders Miscellaneous y necesitan más VRAM de la que queda con la voz resident.

La duración máxima por trabajo son **30 s**, medido: la duración es la entrada del UNet, no un
presupuesto de decodificación, así que un clip de 120 s no es cuatro veces trabajo, es una latent
que no cabe. Para más largo, encadena clips.

## Instalación

**El paso que no es opcional: un entorno aparte.** El agente corre en Python 3.14, y la pila de
generación no llega ahí. Kokoro necesita `misaki`, que exige Python <3.13, y el tokenizer de
LTX-Video necesita `transformers` 4.x. Un entorno 3.12 dedicado es la combinación que funciona:

```bash
uv venv --python 3.12 .venv-compute
uv pip install --python .venv-compute/bin/python torch \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-compute/bin/python \
    "transformers==4.49.0" sentencepiece protobuf diffusers accelerate \
    imageio imageio-ffmpeg scipy soundfile "misaki[en]>=0.7.16" \
    num2words loguru kokoro tiktoken huggingface_hub
```

Y en `.env`:

```
COMPUTE_PYTHON=/ruta/al/proyecto/.venv-compute/bin/python
```

Sin `COMPUTE_PYTHON` los backends heredan el intérprete del agente, que es el que no tiene
torch. La palabra exacta de `transformers` importa: la 4.49 funciona, la 4.55 ya no (quitó un
método que AudioLDM2 llama) y la 5.x no sabe leer el SentencePiece de LTX-Video.

**Voz en inglés.** `misaki[en]` trae spaCy pero no sus datos, y el modelo se instala aparte:

```bash
uv pip install --python .venv-compute/bin/python \
  "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
```

Usa esa URL, no `python -m spacy download`: aquí ese comando imprimió "installation successful"
sin instalar nada. En español no hace falta.

Los pesos de Kokoro (~330 MB) se descargan de HuggingFace la primera vez, al sintetizar.

**STT (whisper.cpp).** No está en los repos de Arch de esta máquina, y sin `sudo` no se
instala, así que se compila:

```bash
git clone --depth 1 https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF
cmake --build build -j12 --target whisper-cli
ln -sf $PWD/build/bin/whisper-cli ~/.local/bin/
```

`GGML_CUDA=OFF` porque en este equipo no hay `nvcc`, solo las bibliotecas de runtime: una
compilación con CUDA falla en el `cmake`. El binario resultante va por CPU, y el backend lo
detecta preguntándole, así que no le pasa un flag de GPU que no entiende.

El modelo se busca en `~/.cache/whisper` sin configurar nada:

```bash
mkdir -p ~/.cache/whisper && cd ~/.cache/whisper
curl -L -o ggml-base.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
```

**Vídeo y música (diffusers).** Los pesos se descargan de HuggingFace la primera vez: LTX-Video
son ~17 GB y AudioLDM2 ~8 GB. Es una descarga larga; después van desde la caché.

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
