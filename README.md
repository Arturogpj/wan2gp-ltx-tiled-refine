# LTX Tiled Refine — WanGP plugin

Makes the **LTX-2.5 Refine Details** (and **Restore**) IC-LoRAs work in [WanGP](https://github.com/deepbeepmeep/Wan2GP) at 1080p, 2K and above, without the blurry result.

## Why

Lightricks trained these IC-LoRAs on **1024x576 windows**. Their own pipelines (ComfyUI `LTXVTiledFusionSampler`, LTX-2 `TiledDiffusionModel`) run them above that size with **per-step tiled fusion**: one full-size latent, and at every denoising step the transformer runs on overlapping 1024x576 windows whose predictions are blended back together.

WanGP runs the whole frame in one call. At 1080p the LoRA works far outside its training window, and the output ends up soft or just resized.

This plugin adds the tiled fusion to WanGP. No WanGP file is edited; it wraps a few functions when the plugin loads.

## Install

WanGP → **Plugins** tab → install from URL:

```
https://github.com/Arturogpj/wan2gp-ltx-tiled-refine
```

Enable it and restart WanGP. The console shows:

```
[LTX Tiled Refine] Active: ic-lora-refine-details on 1024x576, ic-lora-restore on 960x544 fused windows (tiles: quality; low-VRAM control video: on).
```

## Use

| Setting | Value |
|---|---|
| Model | **LTX-2 2.5 Distilled 22B** |
| LoRA | `ltx-2.5-22b-ic-lora-refine-details-1.0` (or `...ic-lora-restore...`), strength 1 |
| Control Video | **LTX2 Raw Format / Control Video for IC-LoRA**, with your source clip |
| Denoising Strength | 1 |
| Phases | **2** (fastest, recommended) or 1 (full 8 steps at full size, a little more natural, much slower) |
| Steps | 8 |
| Resolution | your target size, e.g. 1920x1088 or 2560x1408 |
| Frame rate | leave the model default (24 fps). The LoRA was trained on 24 fps; forcing 30 fps only repeats frames |
| Prompt | a short description of the look, e.g. `sharp photographic detail, crisp natural texture, fine surface detail, clean edges, natural film grain, high resolution footage` |

The plugin only activates when a LoRA whose file name contains `ic-lora-refine-details` or `ic-lora-restore` is selected **and** an IC-LoRA control video is used. Everything else runs as normal WanGP.

### Ready-made presets

On first start the plugin adds two presets to WanGP's LTX-2 presets list (they show up after the next restart; existing files with the same name are never overwritten):

| Preset | What it does | LoRA download |
|---|---|---|
| **LTX Refine Details 1080p 2 Phases (Tiled Refine)** | Refine Details, 1920x1088, 2 Phases | automatic (WanGP's open mirror) |
| **LTX Restore - archive VHS (Tiled Refine)** | Restore, ~1.57 MP keeping the clip's shape (4:3 tape -> ~1408x1088), 1 Phase, 97-frame windows | automatic **after** you accept the licence on the [Restore page](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Restore) and log in to Hugging Face on this PC (`hf auth login`). Otherwise download the file into `loras/ltx2` yourself |

Pick the model **LTX-2 2.5 Distilled 22B**, load a preset, add your clip as the control video, and for Restore replace the `[describe the scene ...]` part of the prompt.

### Restore (old tapes, VHS, film scans)

The **Restore** IC-LoRA ([Lightricks/LTX-2.5-22b-IC-LoRA-Restore](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Restore)) cleans up archive footage: compression damage, tape and sepia casts, flicker, dirt and scratches. Use the Restore preset above, or download `ltx-2.5-22b-ic-lora-restore-1.0.safetensors` into WanGP's `loras/ltx2` folder (the Hugging Face page is gated: accept the licence first). The plugin runs it on **960x544** windows, the size it was trained on (Refine Details uses 1024x576).

Settings from Lightricks' model card:

| Setting | Value |
|---|---|
| LoRA strength | **1.0** (it works like a switch; lower values stop the restoration) |
| Phases | 1, Steps 8 |
| Resolution | **1440 wide**: 1440x816 for 16:9 (WanGP makes it 1408x768, 4 windows), 1440x1088 for 4:3 (1408x1088, 6 windows) |
| Sliding window | **97 frames** (it was trained on 49 and 97 frame clips) |
| Source | deinterlace VHS / telecined footage first, but **don't denoise or sharpen** it |
| Prompt | describe the period, place, light, materials and clothing: it colourises from what you write. Keep it about the whole frame. Put things it must not invent (modern objects, logos, lettering) in the negative prompt |

For more resolution, run **Restore first, then Refine Details** on the restored clip (e.g. at 2x). Lightricks notes that the other order gives emptier results.

### Plugin settings (under **Phases** in the WanGP UI)

**LTX Refine Details tiles** — how much the windows overlap (applies to both LoRAs; the table is for Refine Details). More overlap = more windows = slower.

| Option | 1920x1088 | 2560x1408 | Notes |
|---|---|---|---|
| Quality (default) | 9 windows | 16 windows | at least 50% overlap, Lightricks' recommendation |
| **Balanced** | 6 windows | 9 windows | at least 128 px overlap. No seams seen in our tests; recommended for speed |
| Fast | 4 windows | 9 windows | thinnest overlap (64 px at 1080p), highest seam risk |

If you ever see a faint line where a fine repeating pattern (fabric weave, stripes, grids) shifts, render that clip with **Quality**.

**LTX Refine phase 1 (2 Phases only)** — with 2 Phases, phase 1 is a half-size draft. **Fast** (default) runs it as one window when it is close to the trained size (e.g. 1280x704 for a 2560x1408 output) instead of 2–4 overlapping windows. Same look in our tests, about 2 minutes saved at 2K. **Tiled** is the old behaviour.

Both apply to the next render that starts, no restart needed.

### Speed

RTX 4080 Super (16 GB), 64 GB RAM, 720p source, 5 seconds (121 frames), measured:

| Output | Phases | Tiles | Time |
|---|---|---|---|
| 1920x1088 | 2 | Balanced | 4 min 14 s |
| 1920x1088 | 1 | Balanced | 7 min 06 s |
| **2560x1408** | **2** | **Balanced**, phase 1 Fast | **~7 min** |
| 2560x1408 | 1 | Balanced | 10 min 34 s |

Longer clips cost a bit more per second of video. After each render the console shows where the time went:

```
[LTX Tiled Refine] timing: total 418s = clip encoding 56s | phase 1 denoising 93s | phase 1->2 upscale 1s | phase 2 denoising 209s | video decoding 42s | ...
```

### Memory

WanGP normally loads the whole control clip onto the GPU as float32 before encoding it, which runs out of memory on 16 GB cards with long clips at 1080p and above (e.g. ~13 GB peak for 289 frames). The plugin builds the clip in system RAM in 16-bit, 16 frames at a time, and sends it to the VAE one tile at a time. The result is bit-identical. This applies to every LTX control video, not just the detail LoRAs.

Up to 481 frames (20 s) fit in one sliding window. The transformer always works on 1024x576 windows, so VRAM use stays about the same at any output size.

### Resolutions

LTX needs sizes divisible by 64, and WanGP rounds other values (2560x1440 becomes 2560x1408). To add sizes to the resolution list, put them in `resolutions.json` in the WanGP folder and restart:

```json
[
  ["2048x1152 (16:9) - 2 Phases: phase 1 = 1 window", "2048x1152"],
  ["2560x1440 (16:9) - 2K", "2560x1440"]
]
```

## How it works

Ported from Lightricks' `TiledDiffusionModel` / `VideoModalityTilingHelper`:

- Windows are the LoRA's trained size: 1024x576 for Refine Details, 960x544 for Restore (swapped for portrait). The first and last windows are pinned to the frame edges, the rest spread evenly.
- Each window keeps its own generated tokens plus the IC-LoRA guide tokens that overlap it. Positions are shifted to start at zero, and each window gets its own runtime cache.
- Window outputs are blended with trapezoidal weights. Guide tokens and audio are averaged.

It wraps, in WanGP's `models.ltx2`:

- `LTX2.generate`: detects the LoRA + control video.
- `ltx_pipelines.distilled.denoise_audio_video`: records the canvas size and phase.
- `ltx_pipelines.distilled.simple_denoising_func`: swaps in the tiling transformer wrapper.
- `ltx_pipelines.utils.helpers` control-video loading and VAE encoding: the low-VRAM path.

## Limitations

- **Distilled models only.** The Dev model uses a different WanGP pipeline that this plugin does not patch yet.
- Tested on WanGP v13.14, one GPU (RTX 4080 Super), with the Refine Details LoRA. Restore uses the same code with its own window size, but has had less real-world testing.
- A WanGP update that changes these internal functions can stop the plugin from working until it is updated.

## Changelog

- **1.3.0** — ready-made Refine Details and Restore presets, installed into WanGP on first start.
- **1.2.0** — Restore runs on its trained 960x544 windows (was 1024x576); Restore section in the README.
- **1.1.0** — 2 Phases supported (one-window phase 1 draft), tile presets (Quality / Balanced / Fast) in the UI, low-VRAM control video for long clips, timing line, measured speeds.
- **1.0.0** — per-step tiled fusion, 1 Phase only.

## Credits

Tiled fusion design and the IC-LoRAs by [Lightricks](https://github.com/Lightricks/LTX-2). WanGP by [DeepBeepMeep](https://github.com/deepbeepmeep/Wan2GP).

## License

Apache-2.0
