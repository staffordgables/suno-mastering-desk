# Installation

The Suno Mastering Desk is a single Python file, `suno_master.py`. What you need to install depends on which part you want to use.

| You want to | You need |
|---|---|
| Use the desk in your browser | Python 3.9 or newer. Nothing else. |
| Master files from the command line (batch mode) | Python 3.9 or newer, plus `numpy`, `scipy` and `soundfile`. |
| Make batch mode fast | Also `numba` (optional). |
| Build a standalone application | See [BUILDING.md](BUILDING.md). |

A modern web browser is needed for the desk: a current version of Chrome, Edge, Firefox or Safari. It was developed and tested with Python 3.13 and Chromium.

## 1. Install Python

Check whether Python is already installed:

```bash
python3 --version      # macOS and Linux
py --version           # Windows
```

Any version from 3.9 upward works. If Python is missing or older:

- **Windows:** download the installer from [python.org/downloads](https://www.python.org/downloads/). On the first screen, tick **Add python.exe to PATH**. Alternatively, run `winget install Python.Python.3.13`.
- **macOS:** download the installer from [python.org/downloads](https://www.python.org/downloads/), or with Homebrew run `brew install python`.
- **Linux:** use your distribution's package manager, for example `sudo apt install python3 python3-venv python3-pip` on Debian and Ubuntu, or `sudo dnf install python3` on Fedora.

In the commands below, use `python3` on macOS and Linux and `py` on Windows wherever `python` appears, unless `python` already points to Python 3 on your system.

## 2. Get the code

Clone the repository:

```bash
git clone https://github.com/staffordgables/suno-mastering-desk.git
cd suno-mastering-desk
```

Or download the repository as a ZIP from GitHub (**Code**, then **Download ZIP**) and unzip it. Only `suno_master.py` is needed to run the tool.

## 3. Run the desk

No further installation is needed:

```bash
python suno_master.py
```

Your browser opens at an address such as `http://127.0.0.1:52814/`. If it does not open by itself, copy the address printed in the terminal into your browser. Press **Ctrl+C** in the terminal to stop the desk.

## 4. Install batch mode requirements (optional)

A virtual environment keeps these packages separate from the rest of your system. Create and activate one inside the project folder:

```bash
# macOS and Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
py -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again. In the Windows Command Prompt, use `.venv\Scripts\activate.bat` instead.

Then install the requirements:

```bash
pip install -r requirements.txt
pip install numba          # optional, recommended
```

Each time you open a new terminal, activate the environment again before running the script.

### What each package does

| Package | Minimum version | Purpose |
|---|---|---|
| `numpy` | 1.22 | Array processing for the whole chain. |
| `scipy` | 1.8 | EQ filters, oversampling for true-peak measurement, limiter lookahead. |
| `soundfile` | 0.12 | Reading WAV, FLAC and MP3; writing WAV and FLAC. Its wheels include the libsndfile library. |
| `numba` | any current | Optional. Compiles the compressor and limiter loops. Without it, batch mode prints a notice and runs in plain Python, which is much slower. The first run with numba takes a few extra seconds while it compiles; later runs reuse the compiled code. |

### Check the installation

```bash
python suno_master.py --list-presets
python -c "import numpy, scipy, soundfile; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'soundfile', soundfile.__version__, 'libsndfile', soundfile.__libsndfile_version__)"
```

Then master a file:

```bash
python suno_master.py "path/to/song.wav"
```

## Troubleshooting

**"Batch mode needs numpy, scipy and soundfile"**
The packages are not installed in the Python you are running. Activate the virtual environment, or run `pip install -r requirements.txt` with the same Python that runs the script (`python -m pip install -r requirements.txt` guarantees this).

**MP3 files fail to open in batch mode**
MP3 support needs libsndfile 1.1 or newer. The check command above prints the version. On Windows and macOS, `pip install --upgrade soundfile` installs wheels that bundle a suitable library. On some Linux systems `soundfile` uses the system's libsndfile instead; upgrade it through your package manager, or convert the file to WAV first, for example `ffmpeg -i song.mp3 song.wav`. The desk is not affected: it decodes MP3 in the browser.

**The browser does not open**
Copy the address printed in the terminal into your browser. To start without trying to open a browser, add `--no-browser`.

**"Address already in use"**
Only happens with `--port`. Choose another port, or leave `--port` out to let the system pick a free one.

**No sound in the desk**
Browsers allow audio only after you interact with the page, so press **Play** rather than expecting playback to start by itself. Check that the browser tab is not muted.

**The desk shows plain system fonts**
The desk loads its fonts from Google Fonts. Offline, or behind a firewall that blocks Google Fonts, it falls back to system fonts. Everything else works the same.

**numba fails to install**
numba supports a limited range of Python versions and can lag behind the newest Python release. Batch mode works without it; skip it, or use a Python version that numba supports.
