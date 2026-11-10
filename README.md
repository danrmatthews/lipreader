# LipReader

Real-time meeting assistant that watches the screen or webcam, uses lip movement to identify which on-screen speaker is active, and transcribes microphone / system audio with Whisper.

> This is **audio transcription enhanced by visual speaker tracking**, not true silent lip-reading. Lip movement is used for speaker ID, Whisper still needs audio — even a whisper. Only use in meetings where everyone consents to recording/transcription.

## Dependencies

### System dependencies (Ubuntu)

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip portaudio19-dev libportaudio2
```

- `portaudio19-dev` / `libportaudio2` — required by `sounddevice`. Without it you get `OSError: PortAudio library not found`
- `pavucontrol` — optional, helps select the *Monitor of ...* source for system audio in screen mode

### Python dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | `cv2` — camera/screen capture and display window |
| `mediapipe` | `FaceLandmarker` — face and lip landmarks |
| `mss` | Screen capture (`--source screen`) |
| `numpy` | Image / audio arrays |
| `sounddevice` | Microphone and system-audio input |
| `soundfile` | Audio file I/O |
| `faster-whisper` | Whisper speech-to-text (CPU, `int8`) |
| `PyYAML` | YAML config support |

Optional `requirements.txt`:

```
opencv-python
mediapipe
mss
numpy
sounddevice
soundfile
faster-whisper
PyYAML
```

### Models (downloaded automatically on first run)

- `face_landmarker.task` — MediaPipe FaceLandmarker model, downloaded into the project folder if missing
- Whisper model (`tiny.en` by default in tests, `small.en` recommended for whispers) — downloaded by `faster-whisper` on first run

## Setup

> Keep the virtual environment **outside** the project folder — it contains gigabytes of ML packages and should not be synced. The project itself can live anywhere, e.g. `~/Projects/lipreader`.

### 1. Create and activate the virtual environment

```bash
python3 -m venv ~/.venvs/lipreader-venv
source ~/.venvs/lipreader-venv/bin/activate
```

Your prompt should show `(lipreader-venv)`, **not** `(firmware)`. Using the wrong venv causes `ModuleNotFoundError: No module named 'cv2'`.

### 2. Install Python dependencies

```bash
pip install -U pip
pip install -U opencv-python mediapipe mss numpy sounddevice soundfile faster-whisper PyYAML
```

On Python 3.14, make sure you use the latest `mediapipe`. Do **not** downgrade to `mediapipe==0.10.30` to restore the old `mp.solutions` API — that release has no Python 3.14 wheel.

### 3. Verify the installation

```bash
python3 -c "import cv2, sounddevice, mediapipe; print(cv2.__version__); print(mediapipe.__version__); print(sounddevice.query_devices())"
python3 -c "from mediapipe.tasks.python import vision; print('tasks ok')"
```

Both commands should succeed without `PortAudio library not found` or `module 'mediapipe' has no attribute 'solutions'`.

## Usage

### List audio devices

```bash
python3 main.py --list-devices
```

On the ThinkPad T590 used for development: `25` is the built-in mic (`HiFi__Mic__source`) and `20` is the Speaker monitor (system audio). Device numbers can change — re-check with `--list-devices`.

### Webcam test

```bash
python3 main.py --source camera --model tiny.en --mic-device 25
```

### Screen / meeting mode

Pin the active speaker large on screen, then:

```bash
python3 main.py --source screen --model small.en --mic-device 20
```

For system audio, the device must be a `*.monitor` source. In `pavucontrol` > Recording, select `Monitor of ...` for your output device.

### Visual-only test (no audio)

```bash
python3 main.py --source camera --no-audio
```

### Common options

| Option | Meaning |
|---|---|
| `--source screen\|camera\|video` | Input source |
| `--video <file>` | Video file when `--source video` |
| `--model tiny.en\|base.en\|small.en` | Whisper model. `small.en` is best for whispers, `tiny.en` is fastest |
| `--mic-device <index>` | Audio device from `--list-devices` |
| `--list-devices` | List audio devices and exit |
| `--no-audio` | Skip transcription, visual tracking only |
| `--output <folder>` | Output folder. Default: folder containing `main.py` |
| `--save-lip-crops` | Save lip crops for a future true lip-reading model |

Close the window with the `X` button, `q`, or `Esc`. It saves output on exit.

## Output files

By default, written live to the **executable directory** (the folder containing `main.py`):

- `transcript.txt` — timestamped plain text
- `transcript.json` — structured segments with speaker labels
- `transcript.srt` — subtitles, written on exit

Each new run overwrites them. Rename the old files first to keep them, or pass `--output transcripts` to restore timestamped subfolders.

## Project files

- `main.py` — main program
- `README.md` — this file
- `face_landmarker.task` — downloaded MediaPipe model
- `transcript.txt` / `transcript.json` / `transcript.srt` — generated output

<details>
<summary>Troubleshooting</summary>

**`OSError: PortAudio library not found`**
```bash
sudo apt update && sudo apt install -y portaudio19-dev libportaudio2
```

**`AttributeError: module 'mediapipe' has no attribute 'solutions'`**
Old API was removed in `mediapipe` 0.10.31. Run `pip install -U mediapipe` and make sure `main.py` uses the FaceLandmarker Tasks API patch.

**`ModuleNotFoundError: No module named 'cv2'`**
You activated the wrong venv. Use `source ~/.venvs/lipreader-venv/bin/activate` and check the prompt shows `(lipreader-venv)`.

**Window won't close with `X`**
Press `q` or `Esc`. If the window is frozen, that is a performance issue — report it with the resolution you are running at.

**Whisper / `ctranslate2` / `av` errors on Python 3.14**
```bash
pip install -U "faster-whisper>=1.3"
```

</details>
