# Building a standalone application

`suno_master.py` already contains everything it needs, including the desk's interface, in one file. This guide turns it into a single executable, a "monolithic" application, that runs on computers without Python installed. Double-clicking it opens the desk; giving it file names on the command line runs batch mode.

The recommended tool is [PyInstaller](https://pyinstaller.org/). It packs the Python interpreter, the script and its libraries into one file. [Nuitka](https://nuitka.net/) is described at the end as an alternative.

## What was tested

The steps below were tested on Linux (x86-64) with Python 3.13 and PyInstaller 6.22, in clean virtual environments:

| Build | Contents | Size | Result |
|---|---|---|---|
| Desk only | Python standard library | about 12 MB | Desk opens and saves masters. Batch mode reports that its packages are missing. |
| Full | numpy, scipy, soundfile, numba | about 120 MB | Desk and batch mode both work. A 40-second song masters in about 4 seconds, including start-up. |

The same commands are expected to work on Windows and macOS, since PyInstaller supports both, but they have not been run there. Report problems with your platform and versions.

## Before you start

1. **Build on the operating system you are targeting.** PyInstaller does not cross-compile. A Windows executable must be built on Windows, a macOS app on macOS, and a Linux binary on Linux. To build all three without owning three machines, use the GitHub Actions workflow at the end of this guide.
2. **Build in a fresh virtual environment.** PyInstaller bundles whatever it finds installed. Building from a system Python with years of accumulated packages makes the executable larger and can break it: in testing, an outdated `setuptools` in the system Python produced an executable that crashed on start-up, while the same build from a clean environment worked.
3. **Choose desk-only or full.** The script detects at run time which packages were bundled. If numpy, scipy and soundfile are absent from the build environment, you get the small desk-only build. If they are present, batch mode is included too.

## Desk-only build

```bash
# macOS and Linux
python3 -m venv build-env
source build-env/bin/activate
pip install pyinstaller
pyinstaller --onefile --name suno-mastering-desk suno_master.py
```

```powershell
# Windows (PowerShell)
py -m venv build-env
build-env\Scripts\Activate.ps1
pip install pyinstaller
pyinstaller --onefile --name suno-mastering-desk suno_master.py
```

The executable appears in the `dist` folder: `dist/suno-mastering-desk` on macOS and Linux, `dist\suno-mastering-desk.exe` on Windows.

## Full build, with batch mode

```bash
# macOS and Linux
python3 -m venv build-env
source build-env/bin/activate
pip install -r requirements.txt numba pyinstaller
pyinstaller --onefile --name suno-mastering-desk --collect-data soundfile suno_master.py
```

```powershell
# Windows (PowerShell)
py -m venv build-env
build-env\Scripts\Activate.ps1
pip install -r requirements.txt numba pyinstaller
pyinstaller --onefile --name suno-mastering-desk --collect-data soundfile suno_master.py
```

`--collect-data soundfile` makes sure the libsndfile library that ships inside the `soundfile` package is bundled. Leave `numba` out of the `pip install` line for a smaller executable with slower batch rendering.

## Check the result

```bash
dist/suno-mastering-desk --list-presets          # starts, prints presets
dist/suno-mastering-desk                          # opens the desk in the browser
dist/suno-mastering-desk "path/to/song.wav"       # full build: masters a file
```

On Windows, use `dist\suno-mastering-desk.exe` in place of `dist/suno-mastering-desk`.

## Options worth knowing

| Option | Effect |
|---|---|
| `--onefile` | One executable. It unpacks itself to a temporary folder at each launch, which adds a second or two of start-up time for the full build. |
| `--onedir` | A folder holding the executable and its libraries. Starts faster; distribute it as a ZIP. |
| `--name NAME` | File name of the executable. |
| `--icon FILE` | Application icon: `.ico` on Windows, `.icns` on macOS. |
| `--clean` | Clear PyInstaller's cache before building. Use it if a build behaves oddly after you change packages. |

Keep the console window. Do not use `--noconsole` or `--windowed`: the terminal window is where the desk shows its address and saved files, and closing it, or pressing Ctrl+C in it, is how the desk stops. Without it, the server would keep running in the background with no visible way to stop it.

## Platform notes

**Windows.** SmartScreen may warn about an unrecognized app on first launch; choose **More info**, then **Run anyway**. Some antivirus products flag `--onefile` PyInstaller executables as suspicious because of the self-unpacking step. Building with `--onedir` or signing the executable with a code-signing certificate reduces this.

**macOS.** An unsigned executable downloaded from the internet is blocked by Gatekeeper. Users can right-click it and choose **Open**, or clear the quarantine flag with `xattr -dr com.apple.quarantine suno-mastering-desk`. For wide distribution, sign and notarize it with an Apple Developer account. An executable built on Apple Silicon runs only on Apple Silicon Macs. For Intel Macs, build on an Intel Mac, or on GitHub's `macos-13` machines (see the workflow below).

**Linux.** An executable runs on systems whose C library (glibc) is the same age as, or newer than, the one on the build machine. Build on the oldest distribution you want to support, for example Ubuntu 22.04.

## How the script supports bundling

These are already handled in `suno_master.py`; they are listed so you know what to keep if you modify it.

- The interface is embedded in the script as one string (`PAGE_HTML`), so there are no data files to bundle.
- Batch-mode packages are imported inside a `try` block. A build without them still runs the desk.
- numba normally caches compiled code next to the source file. A bundled executable has no writable source location, so the script turns caching off when it detects it is frozen (`sys.frozen` for PyInstaller, `__compiled__` for Nuitka). Compilation then happens once per launch, adding a few seconds before the first file in a batch.

## Alternative: Nuitka

Nuitka compiles the script to C before bundling it, which can give faster start-up. It needs a C compiler (on Windows, Nuitka offers to download one).

```bash
pip install nuitka
python -m nuitka --onefile --output-filename=suno-mastering-desk suno_master.py
```

For the full build, install the requirements first and add `--include-package-data=soundfile`. Nuitka builds take considerably longer than PyInstaller builds. This route has not been tested with this project.

## Building for all platforms with GitHub Actions

GitHub can build the executables for you on its own Windows, macOS and Linux machines. Save the following as `.github/workflows/build.yml`. It runs when you push a tag starting with `v` (for example `git tag v1.0.0 && git push --tags`), or manually from the **Actions** tab. When it finishes, each platform's executable is listed under the run's **Artifacts**.

```yaml
name: Build executables

on:
  push:
    tags: ['v*']
  workflow_dispatch:

jobs:
  build:
    strategy:
      fail-fast: false
      matrix:
        os: [windows-latest, macos-latest, ubuntu-22.04]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install build requirements
        run: python -m pip install -r requirements.txt numba pyinstaller

      - name: Build
        run: pyinstaller --onefile --name suno-mastering-desk --collect-data soundfile suno_master.py

      - name: Smoke test
        run: ./dist/suno-mastering-desk --list-presets

      - uses: actions/upload-artifact@v4
        with:
          name: suno-mastering-desk-${{ matrix.os }}
          path: dist/
```

`macos-latest` builds for Apple Silicon. Add `macos-13` to the `os` list for an Intel build. This workflow follows the tested local commands, but it has not itself been run; check the first run's log.

To attach the executables to a GitHub release, create a release for the tag and upload the files from each run's Artifacts, or extend the workflow with a release action such as `softprops/action-gh-release`.
