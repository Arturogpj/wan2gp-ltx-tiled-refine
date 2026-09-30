# LTX Tiled Refine — WanGP plugin

Makes the **LTX-2.5 Refine Details** (and **Restore**) IC-LoRAs work in [WanGP](https://github.com/deepbeepmeep/Wan2GP) at 1080p and above, without the blurry result.

## Why

Lightricks trained these IC-LoRAs on **1024x576 windows**. Their own pipelines (ComfyUI `LTXVTiledFusionSampler`, LTX-2 `TiledDiffusionModel`) run them above that size with **per-step tiled fusion**: one full-size latent, and at every denoising step the transformer runs on overlapping 1024x576 windows whose predictions are blended back together.

WanGP runs the whole frame in one call. At 1080p the LoRA works far outside its training window, and the output ends up soft or just resized.

This plugin adds the tiled fusion to WanGP. No WanGP file is edited; it wraps three functions when the plugin loads.

## Install

WanGP → **Plugins** tab → install from URL:

```
https://github.com/Arturogpj/wan2gp-ltx-tiled-refine
```

Enable it and restart WanGP. The console shows:

```
[LTX Tiled Refine] Active: ic-lora-refine-details, ic-lora-restore IC-LoRAs run on 1024x576 fused windows.
```

## Use

| Setting | Value |
|---|---|
| Model | **LTX-2 2.5 Distilled 22B** |
| LoRA | `ltx-2.5-22b-ic-lora-refine-details-1.0` (or `...ic-lora-restore...`), strength 1 |
| Control Video | **LTX2 Raw Format / Control Video for IC-LoRA**, with your source clip |
| Denoising Strength | 1 |
| Phases | 1 (the plugin forces 1: two phases would refine a half-size canvas) |
| Steps | 8 |
| Resolution | your target size, e.g. 1920x1088 |
| Prompt | a short description of the look, e.g. `sharp photographic detail, crisp natural texture, fine surface detail, clean edges, natural film grain, high resolution footage` |

The plugin only activates when a LoRA whose file name contains `ic-lora-refine-details` or `ic-lora-restore` is selected **and** an IC-LoRA control video is used. Everything else runs as normal WanGP.

When it runs, the console shows the windows, e.g.:

```
[LTX Tiled Refine] 1920x1088, 121 frames: 9 windows of 1024x576 fused every step
```

### Speed

Time grows with the number of windows. On an RTX 4080 Super (16 GB), 5 seconds (121 frames):

| Output | Windows | Time |
|---|---|---|
| 1536x864 | 4 | fastest |
| 1920x1088 | 9 | ~10 min |
| 2560x1440 | 16 | ~18 min (estimate) |

VRAM use stays about the same at any size, since every transformer call is one 1024x576 window. Only the VAE encode/decode grows; lower the VAE tiling setting if it runs out of memory.

Tip: to get sizes like 1536x864 or 2560x1440 in the resolution list, add them to `resolutions.json` in the WanGP folder and restart:

```json
[
  ["1536x864 (16:9) - 4 windows", "1536x864"],
  ["2560x1440 (16:9) - 16 windows", "2560x1440"]
]
```

## How it works

Ported from Lightricks' `TiledDiffusionModel` / `VideoModalityTilingHelper`:

- Windows are 1024x576 (576x1024 for portrait), with at least 50% overlap. The first and last windows are pinned to the frame edges.
- Each window keeps its own generated tokens plus the IC-LoRA guide tokens that overlap it. Positions are shifted to start at zero, and each window gets its own runtime cache.
- Window outputs are blended with trapezoidal weights. Guide tokens and audio are averaged.

It wraps, in WanGP's `models.ltx2`:

- `LTX2.generate`: detects the LoRA + control video and forces one full-resolution phase.
- `ltx_pipelines.distilled.denoise_audio_video`: records the canvas size.
- `ltx_pipelines.distilled.simple_denoising_func`: swaps in the tiling transformer wrapper.

## Limitations

- **Distilled models only.** The Dev model uses a different WanGP pipeline that this plugin does not patch yet.
- Tested on WanGP v13.14, one GPU (RTX 4080 Super), with the Refine Details LoRA. Restore uses the same path but is less tested.
- A WanGP update that changes these internal functions can stop the plugin from working until it is updated.

## Credits

Tiled fusion design and the IC-LoRAs by [Lightricks](https://github.com/Lightricks/LTX-2). WanGP by [DeepBeepMeep](https://github.com/deepbeepmeep/Wan2GP).

## License

Apache-2.0
