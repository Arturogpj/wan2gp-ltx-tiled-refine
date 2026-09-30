"""Per-step tiled fusion for the LTX-2.5 detail IC-LoRAs (Refine Details, Restore) in WanGP.

Lightricks trained these IC-LoRAs on 1024x576 windows and runs them above that size with
per-step tiled fusion (ComfyUI ``LTXVTiledFusionSampler``; LTX-2 ``TiledDiffusionModel`` in
``ltx_pipelines/utils/tiled_diffusion.py``): one full-canvas latent, and at every denoising
step the transformer runs on overlapping fixed-size windows whose predictions are blended back.
WanGP runs the whole frame in one call, which puts the LoRA outside its training window.

This module ports ``TiledDiffusionModel`` / ``VideoModalityTilingHelper`` onto WanGP's LTX-2
code (no WanGP file is edited). At plugin load it wraps, in ``models.ltx2``:

- ``LTX2.generate``: when a tiled detail IC-LoRA is selected with an IC-LoRA control video,
  force a single full-resolution stage (two stages would refine a half-size canvas).
- ``ltx_pipelines.distilled.denoise_audio_video``: remember the canvas size of the stage.
- ``ltx_pipelines.distilled.simple_denoising_func``: hand it a transformer wrapper that tiles
  every call. Each window keeps its generated tokens plus the IC-LoRA guide tokens that
  overlap it (same positions, downscale factor 1), gets positions shifted to start at zero and
  its own runtime cache, sees the full audio stream, and is blended with trapezoidal weights
  (guide tokens and audio are averaged), as in Lightricks' code.
"""

import dataclasses
import math
import os
import threading

import torch

TILED_LORA_PATTERNS = ("ic-lora-refine-details", "ic-lora-restore")
TILE_LONG, TILE_SHORT = 1024, 576  # the LoRAs' trained window, in pixels
SPATIAL_SCALE, TEMPORAL_SCALE = 32, 8  # LTX-2 video VAE
LOG = "[LTX Tiled Refine]"

_lock = threading.RLock()
_state = threading.local()
_patched = False


def _tiled_lora(loras_selected):
    for lora in loras_selected or ():
        name = os.path.basename(str(lora).split("|", 1)[0]).lower()
        if any(pattern in name for pattern in TILED_LORA_PATTERNS):
            return name
    return None


def _axis_tiles(length, size):
    """Fixed-size windows with at least 50% overlap; first and last pinned to the canvas edges."""
    if length <= size:
        return [(0, length)]
    step = size - size // 2
    count = math.ceil((length - size) / step) + 1
    return [(round(index * (length - size) / (count - 1)), size) for index in range(count)]


def _axis_weights(tiles):
    """Trapezoidal ramps across each overlap, flat elsewhere and at the canvas edges."""
    weights = []
    for index, (start, size) in enumerate(tiles):
        weight = torch.ones(size)
        position = torch.arange(size, dtype=torch.float32)
        if index > 0:
            overlap = tiles[index - 1][0] + tiles[index - 1][1] - start
            if overlap > 0:
                weight = torch.minimum(weight, (position + 0.5) / overlap)
        if index < len(tiles) - 1:
            overlap = start + size - tiles[index + 1][0]
            if overlap > 0:
                weight = torch.minimum(weight, (size - position - 0.5) / overlap)
        weights.append(weight)
    return weights


class TiledRefineTransformer:
    """Stands in for the X0 transformer inside ``simple_denoising_func``; everything but the
    call is forwarded to the wrapped transformer (LoRA step, preprocessors, ...)."""

    def __init__(self, module, frames, height, width, tile_height, tile_width):
        self._module = module
        self._frames, self._height, self._width = frames, height, width
        rows, cols = _axis_tiles(height, tile_height), _axis_tiles(width, tile_width)
        row_weights, col_weights = _axis_weights(rows), _axis_weights(cols)
        self._tiles = [(row, col, row_weight[:, None] * col_weight[None, :])
                       for row, row_weight in zip(rows, row_weights) for col, col_weight in zip(cols, col_weights)]
        self._caches = {}

    def __getattr__(self, name):
        return getattr(self._module, name)

    @property
    def tile_count(self):
        return len(self._tiles)

    def _generated_indices(self, tile_index, device):
        (row, height), (col, width), _ = self._tiles[tile_index]
        frames = torch.arange(self._frames, device=device)
        rows = torch.arange(row, row + height, device=device)
        cols = torch.arange(col, col + width, device=device)
        return (frames[:, None, None] * self._height * self._width + rows[None, :, None] * self._width + cols[None, None, :]).reshape(-1)

    def _runtime_cache(self, tile_index, entry_index):
        from models.ltx2.ltx_core.types import LatentStateRuntimeCache

        return self._caches.setdefault((tile_index, entry_index), LatentStateRuntimeCache())

    def _tile_modality(self, modality, tile_index, entry_index, generated):
        token_count = modality.latent.shape[1]
        generated_count = self._frames * self._height * self._width
        positions = modality.positions
        keep, conditioning = generated, None
        if token_count > generated_count:
            tile_positions = positions[:, :, generated]
            starts, ends = tile_positions[..., 0].amin(dim=2), tile_positions[..., 1].amax(dim=2)  # (B, 3)
            cond_positions = positions[:, :, generated_count:]
            overlaps = ((cond_positions[..., 0] < ends[..., None]) & (cond_positions[..., 1] > starts[..., None])).all(dim=1)
            reference = cond_positions[:, 0, :, 0] < 0
            conditioning = generated_count + (overlaps | reference).any(dim=0).nonzero(as_tuple=False).squeeze(1)
            keep = torch.cat([generated, conditioning])
        tile_positions = positions[:, :, keep]
        offset = tile_positions[:, :, : generated.numel(), 0].amin(dim=2, keepdim=True).unsqueeze(-1)
        changes = {
            "latent": modality.latent[:, keep],
            "positions": tile_positions - offset,
            "runtime_cache": self._runtime_cache(tile_index, entry_index),
        }
        if modality.frame_indices is not None:
            changes["frame_indices"] = modality.frame_indices[:, keep]
        elif torch.is_tensor(modality.timesteps) and modality.timesteps.ndim >= 2 and modality.timesteps.shape[1] == token_count:
            changes["timesteps"] = modality.timesteps[:, keep]
        if modality.attention_mask is not None:
            changes["attention_mask"] = modality.attention_mask[:, keep][:, :, keep]
        for name in ("keyframes_mask", "cross_attention_mask", "context_mask"):
            value = getattr(modality, name, None)
            if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == token_count:
                changes[name] = value[:, keep]
        return dataclasses.replace(modality, **changes), conditioning

    def __call__(self, video=None, audio=None, perturbations=None, **kwargs):
        entries = list(video) if isinstance(video, (list, tuple)) else [video]
        generated_count = self._frames * self._height * self._width
        if video is None or any(entry is None or entry.latent.shape[1] < generated_count for entry in entries):
            return self._module(video=video, audio=audio, perturbations=perturbations, **kwargs)
        device = entries[0].latent.device
        video_sums = [None] * len(entries)
        video_weights = [torch.zeros(entry.latent.shape[1], device=device, dtype=torch.float32) for entry in entries]
        audio_sums = [None] * len(entries)
        for tile_index in range(len(self._tiles)):
            generated = self._generated_indices(tile_index, device)
            tiled, conditionings = [], []
            for entry_index, entry in enumerate(entries):
                tile_entry, conditioning = self._tile_modality(entry, tile_index, entry_index, generated)
                tiled.append(tile_entry)
                conditionings.append(conditioning)
            out_video, out_audio = self._module(video=tiled if isinstance(video, (list, tuple)) else tiled[0],
                                                audio=audio, perturbations=perturbations, **kwargs)
            if out_video is None and out_audio is None:
                return None, None
            out_videos = list(out_video) if isinstance(video, (list, tuple)) else [out_video]
            out_audios = list(out_audio) if isinstance(video, (list, tuple)) else [out_audio]
            weight = self._tiles[tile_index][2].to(device)[None].expand(self._frames, -1, -1).reshape(-1)
            for entry_index, tile_video in enumerate(out_videos):
                if tile_video is None:
                    continue
                if video_sums[entry_index] is None:
                    video_sums[entry_index] = torch.zeros((tile_video.shape[0], entries[entry_index].latent.shape[1], tile_video.shape[2]),
                                                          device=device, dtype=torch.float32)
                count = generated.numel()
                video_sums[entry_index][:, generated] += tile_video[:, :count].float() * weight[None, :, None]
                video_weights[entry_index][generated] += weight
                conditioning = conditionings[entry_index]
                if conditioning is not None and conditioning.numel():
                    video_sums[entry_index][:, conditioning] += tile_video[:, count:].float()
                    video_weights[entry_index][conditioning] += 1.0
                tile_audio = out_audios[entry_index] if entry_index < len(out_audios) else None
                if tile_audio is not None:
                    audio_sums[entry_index] = tile_audio.float() if audio_sums[entry_index] is None else audio_sums[entry_index] + tile_audio.float()
            del out_video, out_audio, out_videos, out_audios, tiled
        dtype = entries[0].latent.dtype
        results_video, results_audio = [], []
        for entry_index in range(len(entries)):
            total = video_sums[entry_index]
            if total is not None:
                total = (total / video_weights[entry_index].clamp_min(1e-6)[None, :, None]).to(dtype)
            results_video.append(total)
            audio_total = audio_sums[entry_index]
            results_audio.append(None if audio_total is None else (audio_total / len(self._tiles)).to(dtype))
        if isinstance(video, (list, tuple)):
            return results_video, results_audio
        return results_video[0], results_audio[0]


def apply_patches():
    global _patched
    with _lock:
        if _patched:
            return
        from models.ltx2 import ltx2 as ltx2_module
        from models.ltx2.ltx_pipelines import distilled

        original_generate = ltx2_module.LTX2.generate
        original_denoise = distilled.denoise_audio_video
        original_denoising_func = distilled.simple_denoising_func

        def generate(self, *args, **kwargs):
            lora = _tiled_lora(kwargs.get("loras_selected"))
            video_prompt_type = kwargs.get("video_prompt_type") or ""
            if args or lora is None or not ("V" in video_prompt_type and "G" in video_prompt_type) or kwargs.get("input_frames") is None:
                return original_generate(self, *args, **kwargs)
            if int(kwargs.get("guide_phases", 1) or 1) != 1:
                print(f"{LOG} {lora} runs as one full-resolution stage; switching Phases to 1")
                kwargs["guide_phases"] = 1
            _state.active = True
            try:
                return original_generate(self, *args, **kwargs)
            finally:
                _state.active = False
                _state.shape = None

        def denoise_audio_video(*args, **kwargs):
            if not getattr(_state, "active", False):
                return original_denoise(*args, **kwargs)
            _state.shape = kwargs.get("output_shape")
            try:
                return original_denoise(*args, **kwargs)
            finally:
                _state.shape = None

        def simple_denoising_func(*args, **kwargs):
            shape = getattr(_state, "shape", None)
            transformer = kwargs.get("transformer")
            if getattr(_state, "active", False) and shape is not None and transformer is not None:
                frames = (int(shape.frames) - 1) // TEMPORAL_SCALE + 1
                height, width = int(shape.height) // SPATIAL_SCALE, int(shape.width) // SPATIAL_SCALE
                portrait = shape.height > shape.width
                tile_height = (TILE_LONG if portrait else TILE_SHORT) // SPATIAL_SCALE
                tile_width = (TILE_SHORT if portrait else TILE_LONG) // SPATIAL_SCALE
                wrapped = TiledRefineTransformer(transformer, frames, height, width, tile_height, tile_width)
                if wrapped.tile_count > 1:
                    print(f"{LOG} {shape.width}x{shape.height}, {shape.frames} frames: {wrapped.tile_count} windows of "
                          f"{tile_width * SPATIAL_SCALE}x{tile_height * SPATIAL_SCALE} fused every step")
                    kwargs["transformer"] = wrapped
            return original_denoising_func(*args, **kwargs)

        ltx2_module.LTX2.generate = generate
        distilled.denoise_audio_video = denoise_audio_video
        distilled.simple_denoising_func = simple_denoising_func
        _patched = True
        print(f"{LOG} Active: {', '.join(TILED_LORA_PATTERNS)} IC-LoRAs run on {TILE_LONG}x{TILE_SHORT} fused windows.")
