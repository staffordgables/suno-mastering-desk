# Suno Mastering Desk

A mastering tool for songs exported from Suno. It cleans up the problems AI-generated audio tends to have, shapes tone and stereo width, adds gentle compression, and brings each song to a chosen loudness for streaming, CD or club play.

It works in two ways from a single Python file:

- **The desk** is a visual mixing-desk interface that opens in your web browser. You hear every change live while the song plays, compare the master against the original at matched volume, and save a finished WAV. It runs entirely on your own computer and needs nothing beyond Python itself.
- **Batch mode** masters one file or a whole folder from the command line, using exactly the same processing. Settings you dial in on the desk can be saved and applied to every track on an album.

This project is not affiliated with or endorsed by Suno, Inc.

## Contents

- [Quick start](#quick-start)
- [What the desk does](#what-the-desk-does)
- [A typical session](#a-typical-session)
- [Batch mode](#batch-mode)
- [Settings files](#settings-files)
- [Presets](#presets)
- [Choosing a loudness target](#choosing-a-loudness-target)
- [The standalone web page](#the-standalone-web-page)
- [How it works](#how-it-works)
- [Limitations](#limitations)
- [Installation](INSTALL.md) (separate guide)
- [Building a standalone application](BUILDING.md) (separate guide)

## Quick start

With Python 3.9 or newer installed (see [INSTALL.md](INSTALL.md)):

```bash
python suno_master.py
```

Your browser opens the desk with a short demo loop already loaded. Click **Load track** or drag a WAV or MP3 onto the page to work on your own song.

To open the desk with a song already loaded:

```bash
python suno_master.py --gui "My Song.wav"
```

To master files without opening the desk (requires `pip install -r requirements.txt`):

```bash
python suno_master.py "My Song.wav"
python suno_master.py *.wav -o masters
```

Stop the desk by pressing **Ctrl+C** in the terminal window where you started it.

## What the desk does

Every panel and every control has built-in help. Hover over a panel title, or the **?** beside it, for an explanation of the panel; hover over a control's underlined name for a one-line note. On a touch screen, tap the **?**.

The processing runs in this order:

| Stage | Panel | What it does |
|---|---|---|
| 1 | **Cleanup** | Input trim, low cut for rumble, a mud band for boxiness, a sweepable harshness band for the metallic ring common in generated vocals and cymbals, and a top-end fizz cut for brittle haze above 16 kHz. |
| 2 | **Tone** | Low shelf at 100 Hz, presence at 2.8 kHz and an air shelf at 12 kHz. |
| 3 | **Stereo** | Mid/side width, with mono bass below a chosen frequency so low end stays solid on phones and club systems. |
| 4 | **Glue and color** | A stereo-linked bus compressor with soft knee, followed by soft saturation. |
| 5 | **Loudness** | A 5 ms lookahead brickwall limiter with adjustable ceiling and release, driven to hit your loudness target. |

Alongside the chain:

- **Listen and compare** shows the waveform of the original (grey) and the master (orange outline). Click the waveform to jump. **Original / Master** switches what you hear (or press **B**); **Space** plays and pauses. **Level-matched A/B** plays the original at the master's loudness so you judge the sound, not the volume.
- **Spectrum** shows the frequency content of the original and of what you hear, live while playing.
- **Output meters** show momentary and short-term loudness, peak level, and how much the compressor and limiter are reducing the level.
- **Loudness report** measures the whole song before and after: integrated loudness, short-term maximum, true peak, sample peak, loudness range and peak-to-loudness ratio. It adds plain-language notes when something needs attention.
- **Export** saves the master as a 24-bit WAV, or 16-bit with dither for CD, at the song's own sample rate.

### What happens live and what does not

Everything you hear during playback is processed in real time, so slider moves are audible within milliseconds. Integrated loudness, however, can only be measured over a whole song. **Fit to target** therefore renders the entire track (much faster than real time), measures it, and adjusts the limiter drive, repeating until the master lands within 0.1 LU of the target. If you change settings afterwards, the report marks its figures as out of date until you fit or measure again. **Save master** always renders the full song with the current settings; it does not record the playback.

## A typical session

1. Load the song. The desk measures the original and fits the default preset to −14 LUFS automatically.
2. Choose a preset close to the style of the song.
3. Press **Play**. Work through **Cleanup** first: sweep **Harshness frequency** slowly while listening for the point where the metallic ring drops away, then set **Harshness** to taste.
4. Make small **Tone** moves, usually 1 to 2 dB.
5. Watch **Glue GR** while playing; 1 to 3 dB of reduction is typical.
6. Pick a loudness target and click **Fit to target**.
7. Switch between **Original** and **Master** with level matching on. If the master sounds worse at equal volume, back off.
8. Check the loudness report notes, then **Save master**. Click **Save settings** if you want to reuse the settings.

When run through Python, the desk saves files to the folder of the song you launched it with, or to `~/Music/Suno Masters` otherwise. Use `-o` to choose a different folder. Existing files are never overwritten; a numbered copy is written instead.

## Batch mode

Batch mode needs `numpy`, `scipy` and `soundfile` (see [INSTALL.md](INSTALL.md)). For each file it measures the original, fits the limiter drive to the target, prints a loudness report and writes the master.

```bash
python suno_master.py song.wav                          # master one file next to the original
python suno_master.py *.wav -o masters                  # a whole folder, into ./masters
python suno_master.py *.wav --settings album.json       # use settings saved from the desk
python suno_master.py song.wav --preset warm --target -16
python suno_master.py song.wav --set harshGain=-3.5 --set width=1.2
python suno_master.py song.wav --no-fit --drive 6       # fixed drive, no fitting
python suno_master.py *.wav --json report.json          # save every report and setting
```

Sample output:

```
song.wav
    0:40.0  44.1 kHz
    fitting to -14.0 LUFS
    pass 1: drive +5.52 dB -> -15.49 LUFS
    pass 2: drive +7.00 dB -> -14.00 LUFS
    drive +7.00 dB, glue GR max 2.2 dB, limiter GR max 0.0 dB
                              Original    Master
    Integrated (LUFS)            -19.5     -14.0
    Short-term max (LUFS)        -19.4     -13.9
    True peak (dBTP)              -7.1      -0.8
    Sample peak (dBFS)            -7.1      -1.0
    Loudness range (LU)            0.2       0.2
    Peak to loudness (dB)         12.5      13.2
    wrote song_master.wav
```

### Options

| Option | Meaning |
|---|---|
| `inputs` | Audio files (WAV, FLAC, MP3). With none, the desk opens. |
| `--gui` | Open the desk, loading the first input file. |
| `--port N` | Port for the desk. Default: any free port. |
| `--no-browser` | Start the desk without opening a browser; the address is printed. |
| `-o`, `--outdir DIR` | Output folder. Batch default: next to each input. |
| `--settings FILE` | Settings file saved from the desk. |
| `-p`, `--preset NAME` | `cleanup`, `loud`, `acoustic`, `warm` or `flat`. Default `cleanup`. |
| `-t`, `--target LUFS` | Integrated loudness target, for example `-14`. |
| `-c`, `--ceiling DBFS` | Limiter ceiling, for example `-1`. |
| `--set KEY=VALUE` | Override any single setting. Repeatable. Keys are listed below. |
| `--no-fit` | Skip fitting and use `--drive` as given. |
| `--drive DB` | Limiter drive: a starting point for fitting, or fixed with `--no-fit`. |
| `--bits 16\|24` | Output bit depth. 16-bit adds TPDF dither. Default 24. |
| `--format wav\|flac` | Output format. Default `wav`. |
| `--suffix TEXT` | Added to output file names. Default `_master`. |
| `--json FILE` | Write all settings, measurements and notes to a JSON file. |
| `--list-presets` | Show the presets and how they differ from the default. |
| `--export-html FILE` | Write the desk as a standalone web page and exit. |

Options are applied in this order: preset, then settings file, then `--target` and `--ceiling`, then each `--set`. Later ones win, except that a preset named in a settings file is used only when `--preset` is not given.

### Setting keys

These names are used by `--set`, by settings files, and match the controls on the desk.

| Key | Desk control | Unit | Default |
|---|---|---|---|
| `trim` | Input trim | dB | 0 |
| `hpf` | Low cut (10 means off) | Hz | 30 |
| `mudGain` / `mudFreq` | Mud / Mud frequency | dB / Hz | −1.5 / 280 |
| `harshGain` / `harshFreq` | Harshness / Harshness frequency | dB / Hz | −2 / 3800 |
| `fizz` | Top-end fizz cut (22000 means off) | Hz | 18000 |
| `low` | Low shelf, 100 Hz | dB | 1 |
| `pres` | Presence, 2.8 kHz | dB | 0 |
| `air` | Air shelf, 12 kHz | dB | 1.5 |
| `width` | Width (1.0 = 100%) | ratio | 1.1 |
| `monoBass` | Mono bass (0 means off) | Hz | 120 |
| `compThr` / `compRatio` | Glue threshold / ratio (1 means off) | dB / ratio | −20 / 2 |
| `compAtt` / `compRel` | Attack / Release | ms | 30 / 200 |
| `sat` | Saturation | 0 to 1 | 0.15 |
| `drive` | Limiter drive | dB | set by fitting |
| `ceiling` | Ceiling | dBFS | −1 |
| `limRel` | Limiter release | ms | 80 |
| `target` | Loudness target | LUFS | −14 |

## Settings files

**Save settings** on the desk writes a small JSON file holding the preset name and every control value. **Load settings** reads one back. The same file works in batch mode:

```bash
python suno_master.py album/*.wav --settings "track1_settings.json" -o album/masters
```

This makes it practical to tune one representative song by ear on the desk, then master the rest of an album consistently. Each file is still fitted to the target individually, so quieter and louder songs all land at the same loudness.

## Presets

| Preset | Character | Target |
|---|---|---|
| **Suno cleanup** (`cleanup`) | Balanced default: gentle harshness and fizz control, slight width, light glue. | −14 LUFS |
| **Loud and bright** (`loud`) | More air and saturation, firmer compression, faster limiter. | −9 LUFS |
| **Acoustic and vocal** (`acoustic`) | Lighter touch, more harshness control around 4.2 kHz, no added width. | −16 LUFS |
| **Warm and dark** (`warm`) | Stronger top-end roll-off, more low end, more saturation. | −14 LUFS |
| **Limiter only** (`flat`) | No EQ, width or compression; loudness only. | −14 LUFS |

## Choosing a loudness target

| Target | Typical use |
|---|---|
| −16 LUFS | Apple Music, podcasts, quieter and more dynamic material. |
| −14 LUFS | Spotify, YouTube, Tidal and Amazon Music normalize to about this level. A safe default for streaming. |
| −11 LUFS | Loud masters for genres that expect density. |
| −9 LUFS | Club and DJ use, very dense styles. |

Streaming services turn louder songs down to their reference level, so a master much louder than −14 LUFS will not play louder there; it will mostly lose punch. Most distributors ask for a true peak no higher than −1 dBTP. The limiter holds sample peaks exactly at the ceiling, but peaks that form between samples can land a few tenths above it. The report flags this; lowering the ceiling to around −1.3 dBFS usually resolves it.

## The standalone web page

```bash
python suno_master.py --export-html desk.html
```

This writes the desk as one self-contained HTML file. Opened directly in a browser, or hosted on any web server (GitHub Pages works), it offers everything except saving to a chosen folder: **Save master** downloads the WAV through the browser instead. The page loads its fonts from Google Fonts when online and falls back to system fonts otherwise.

## How it works

- **Live preview** uses the Web Audio API. The EQ and stereo stages are standard biquad filters; the compressor, saturation and limiter run in an AudioWorklet.
- **Batch mode** implements the same chain with numpy and scipy. Masters rendered from the desk and from batch mode with the same settings were compared in testing and differed by about 82 dB below the signal level, which is effectively identical.
- **Loudness** follows ITU-R BS.1770-4 and EBU R128: K-weighting, 400 ms blocks with absolute (−70 LUFS) and relative (−10 LU) gating for integrated loudness, and 3 s windows for short-term loudness and loudness range.
- **True peak** is measured with 4× oversampling.
- **The local server** listens only on 127.0.0.1. Saving requires a random token issued when the server starts, requests addressed to any other host name are refused, and only `.wav` and `.json` files can be written, only into the output folder.

## Limitations

- The desk works on audio files only. It cannot process a live stream, such as Suno playing in another browser tab.
- Rendering and fitting take a few seconds per song in the desk; batch mode without numba is noticeably slower than with it.
- Batch mode reads MP3 only with libsndfile 1.1 or newer, which current `soundfile` packages include. See [INSTALL.md](INSTALL.md) if MP3 files fail to open.
- Mastering cannot fix problems baked into a mix. If a Suno export is already heavily limited (the report notes a true peak near 0 dBTP), extra limiting adds little.
