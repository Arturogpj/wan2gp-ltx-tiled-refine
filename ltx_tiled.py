"""Per-step tiled fusion for the LTX-2.5 detail IC-LoRAs (Refine Details, Restore) in WanGP.

Lightricks trained these IC-LoRAs on fixed windows (Refine Details 1024x576, Restore 960x544) and runs them above that size with
per-step tiled fusion (ComfyUI ``LTXVTiledFusionSampler``; LTX-2 ``TiledDiffusionModel`` in
``ltx_pipelines/utils/tiled_diffusion.py``): one full-canvas latent, and at every denoising
step the transformer runs on overlapping fixed-size windows whose predictions are blended back.
WanGP runs the whole frame in one call, which puts the LoRA outside its training window.

This module ports ``TiledDiffusionModel`` / ``VideoModalityTilingHelper`` onto WanGP's LTX-2
code (no WanGP file is edited). At plugin load it wraps, in ``models.ltx2``:

- ``LTX2.generate``: activates when a tiled detail IC-LoRA is selected with an IC-LoRA control video.
- ``ltx_pipelines.distilled.denoise_audio_video``: remember the canvas size and phase of the stage.
- ``ltx_pipelines.distilled.simple_denoising_func``: hand it a transformer wrapper that tiles
  every call. Each window keeps its generated tokens plus the IC-LoRA guide tokens that
  overlap it (same positions, downscale factor 1), gets positions shifted to start at zero and
  its own runtime cache, sees the full audio stream, and is blended with trapezoidal weights
  (guide tokens and audio are averaged), as in Lightricks' code.
- ``ltx_pipelines.utils.helpers`` control-video loading/encoding: low-VRAM path (below).

With 2 Phases, phase 1 is a half-size draft. When that draft is at most 1.6x the trained window
(2560x1408 -> 1280x704), it runs as one window instead of 2-4 heavily overlapping ones.

Settings are chosen in the WanGP UI (dropdowns under Phases) and saved to ``config.json``:
- ``tiles``: ``quality`` (windows overlap by half, Lightricks' default; 9 windows at 1920x1088),
  ``balanced`` (>= 128 px overlap; 6 at 1080p) or ``fast`` (>= 32 px; 4 at 1080p, thinnest seams).
- ``phase1_single_window``: the one-window phase 1 draft above (default on).
- ``low_vram_control_video``: build the IC-LoRA / control clip in system RAM in 16-bit and send it to the
  VAE one tile at a time, instead of holding it on the GPU in float32 (applies to every LTX control video).
A timing line is printed after each detail render.
"""

import dataclasses
import json
import math
import os
import sys
import threading

import torch

# Each IC-LoRA runs on the window it was trained on (model cards): Refine Details 1024x576, Restore 960x544.
TILE_WINDOWS = {"ic-lora-refine-details": (1024, 576), "ic-lora-restore": (960, 544)}
TILED_LORA_PATTERNS = tuple(TILE_WINDOWS)
FORCE_ONE_PHASE = False  # True: always run one full-resolution phase
TILE_LONG, TILE_SHORT = TILE_WINDOWS["ic-lora-refine-details"]  # default window, in pixels
SPATIAL_SCALE, TEMPORAL_SCALE = 32, 8  # LTX-2 video VAE
LOG = "[LTX Tiled Refine]"
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
# Minimum overlap between neighbouring windows, in latent cells (32 px). 1920x1088: quality 9, balanced 6, fast 4 windows.
TILE_MODES = {"quality": None, "balanced": 4, "fast": 1}  # None = at least half a window (Lightricks' default)
CONTROL_CHUNK_FRAMES = 16
# Phase 1 of 2 Phases is the low-res draft (half the final size). When it is at most this many trained windows
# in area (2K: 1280x704 = 1.53), one pass over the whole draft replaces 2-4 heavily overlapping windows.
PHASE1_SINGLE_WINDOW_MAX_AREA = 1.6

_lock = threading.RLock()
_state = threading.local()
_patched = False


def _load_config():
    config = {"tiles": "quality", "low_vram_control_video": True, "phase1_single_window": True}
    try:
        with open(CONFIG_PATH, encoding="utf-8") as handle:
            config.update(json.load(handle))
    except FileNotFoundError:
        pass
    except Exception as error:
        print(f"{LOG} could not read config.json ({error}); using defaults")
    if config["tiles"] not in TILE_MODES:
        print(f"{LOG} unknown tiles mode {config['tiles']!r}; using 'quality' (choices: {', '.join(TILE_MODES)})")
        config["tiles"] = "quality"
    return config


def _tiled_lora(loras_selected):
    for lora in loras_selected or ():
        name = os.path.basename(str(lora).split("|", 1)[0]).lower()
        if any(pattern in name for pattern in TILED_LORA_PATTERNS):
            return name
    return None


def _axis_tiles(length, size, min_overlap=None):
    """Fixed-size windows overlapping by at least ``min_overlap`` cells (default: half a window);
    first and last pinned to the canvas edges, the rest spread evenly."""
    if length <= size:
        return [(0, length)]
    min_overlap = size // 2 if min_overlap is None else max(1, min(int(min_overlap), size // 2))
    step = size - min_overlap
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

    def __init__(self, module, frames, height, width, tile_height, tile_width, min_overlap=None):
        self._module = module
        self._frames, self._height, self._width = frames, height, width
        rows, cols = _axis_tiles(height, tile_height, min_overlap), _axis_tiles(width, tile_width, min_overlap)
        self.min_overlap_px = min([start + size - nxt for axis in (rows, cols) for (start, size), (nxt, _) in zip(axis, axis[1:])] or [0]) * SPATIAL_SCALE
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


def _timed(label, function):
    """Stopwatch for one pipeline job while a detail-LoRA render runs; totals are printed at the end."""
    import time

    def wrapper(*args, **kwargs):
        timings = getattr(_state, "timings", None)
        if timings is None:
            return function(*args, **kwargs)
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings.append((label, time.perf_counter() - start))
    return wrapper


def _print_timings(total):
    timings = getattr(_state, "timings", None) or []
    merged = {}
    phase = 0
    for label, seconds in timings:
        if label == "denoise":
            phase += 1
            label = f"phase {phase} denoising"
        merged[label] = merged.get(label, 0.0) + seconds
    other = total - sum(merged.values())
    parts = [f"{label} {seconds:.0f}s" for label, seconds in merged.items()] + [f"other {other:.0f}s"]
    print(f"{LOG} timing: total {total:.0f}s = " + " | ".join(parts))


def _load_control_video_cpu(video_path, height, width, frame_cap, dtype, device):
    """Same pixels as media_io.load_video_conditioning, but built 16 frames at a time and kept in system RAM
    in the target dtype. WanGP's version holds the whole clip on the GPU in float32 (plus a temporary copy),
    which is ~6.75 GiB for 289 frames at 1920x1088 and ran out of memory before the second phase."""
    from models.ltx2.ltx_pipelines.utils import media_io

    def prepared(chunk):
        chunk = media_io.resize_and_center_crop(chunk.to(device=device, dtype=torch.float32), height, width)
        return media_io.normalize_latent(chunk, device, dtype).to("cpu")

    parts = []
    if isinstance(video_path, str):
        parts = [prepared(frame) for frame in media_io.decode_video_from_file(path=video_path, frame_cap=frame_cap, device="cpu")]
    else:
        video = video_path
        if not torch.is_tensor(video) or video.ndim != 4:
            video = media_io._coerce_video_input(video)
        video = media_io._normalize_video_tensor(video)
        if frame_cap is not None and video.shape[0] > frame_cap:
            video = video[:frame_cap]
        # _scale_to_255 decides from the whole clip's range, so take it once and apply the same rule per chunk
        low, high = (float(video.min()), float(video.max())) if torch.is_floating_point(video) else (0.0, 255.0)
        for start in range(0, video.shape[0], CONTROL_CHUNK_FRAMES):
            chunk = video[start:start + CONTROL_CHUNK_FRAMES].to(torch.float32)
            if torch.is_floating_point(video) and high <= 1.0 and low >= -1.0:
                chunk = (chunk + 1.0) * 127.5
            elif torch.is_floating_point(video) and high <= 1.0 and low >= 0.0:
                chunk = chunk * 255.0
            parts.append(prepared(chunk))
    return torch.cat(parts, dim=2) if parts else None


def _caller_locals(function_name):
    frame = sys._getframe(2)
    while frame is not None and frame.f_code.co_name != function_name:
        frame = frame.f_back
    return frame.f_locals if frame is not None else None


def _hold_last_frame(tensor, frames, padding):
    if tensor is None or not torch.is_tensor(tensor) or tensor.ndim < 2 or tensor.shape[1] != frames:
        return tensor
    return torch.cat([tensor, tensor[:, -1:].expand(-1, padding, *tensor.shape[2:])], dim=1)


def _last_window_padding(video_guides, video_masks, pre_video_guide):
    """WanGP shrinks a sliding window to the length of its control clip (and then stops generating), but LTX
    control clips after the first window leave out the overlap frames (all but one). So the window came out
    overlap - 1 frames short and every later window was skipped; on the last window up to 7 more frames were
    dropped to reach a valid frame count (~20 frames, 0.8 s, lost at 24 fps on a two-window clip).
    Hold the last control frame so WanGP keeps the right window length; frames past the end of the control
    video are trimmed after generation. Returns padded (guides, masks), or None to leave WanGP unchanged."""
    caller = _caller_locals("generate_media")
    if caller is None or pre_video_guide is not None or "control_frames_offset" in caller:  # unknown WanGP or already fixed there
        return None
    if not (caller.get("control_video_trim") and caller.get("sliding_window") and caller.get("dont_cat_preguide")):
        return None
    if "ltx2" not in str(caller.get("base_model_type", "")):
        return None
    keep_frames, window_frames = caller.get("keep_frames_parsed"), caller.get("current_video_length")
    latent_size, frame_offset = int(caller.get("latent_size", 8) or 8), int(caller.get("frames_offset", 1) or 1)
    guide = video_guides[0] if video_guides else None
    if keep_frames is None or window_frames is None or guide is None or not torch.is_tensor(guide):
        return None
    available = int(guide.shape[1])
    uncovered = int(window_frames) - len(keep_frames)
    if not 0 < available <= len(keep_frames) or uncovered <= 0 or uncovered % latent_size:
        return None
    needed = available + uncovered
    valid = needed + (frame_offset - needed) % latent_size
    if valid > window_frames:
        return None
    padding = valid - available
    _state.trim_tail = valid - needed
    _state.trim_fps = caller.get("fps")
    if available < len(keep_frames):
        print(f"{LOG} last window: control video ends after {available} frames; holding its last frame for {padding} more "
              f"so the window keeps all of them ({_state.trim_tail} held frames trimmed from the output)")
    return ([_hold_last_frame(t, available, padding) for t in video_guides],
            [_hold_last_frame(t, available, padding) for t in video_masks])


def _trim_tail(result):
    tail, fps = getattr(_state, "trim_tail", 0), getattr(_state, "trim_fps", None)
    _state.trim_tail = 0
    if not tail or not isinstance(result, dict) or not torch.is_tensor(result.get("x")) or result["x"].shape[1] <= tail:
        return result
    result["x"] = result["x"][:, :-tail]
    audio, rate = result.get("audio"), result.get("audio_sampling_rate")
    if audio is not None and rate and fps:
        cut = int(round(tail * rate / fps))
        if 0 < cut < audio.shape[0]:
            result["audio"] = audio[:-cut]
    return result


class _InputToDevice:
    """Encoder proxy: VAE tiles sliced from a system-RAM clip are moved to the GPU one at a time."""

    def __init__(self, encoder, device):
        self._encoder, self._device = encoder, device

    def __getattr__(self, name):
        return getattr(self._encoder, name)

    def __call__(self, tile, *args, **kwargs):
        return self._encoder(tile.to(self._device, non_blocking=True), *args, **kwargs)


def apply_patches():
    global _patched
    with _lock:
        if _patched:
            return
        from models.ltx2 import ltx2 as ltx2_module
        from models.ltx2.ltx_pipelines import distilled, ti2vid_two_stages
        from models.ltx2.ltx_pipelines.utils import helpers

        original_generate = ltx2_module.LTX2.generate
        original_denoise = distilled.denoise_audio_video
        original_denoising_func = distilled.simple_denoising_func
        original_control_video = helpers.video_conditionings_by_control_video
        original_load_video = helpers.load_video_conditioning
        original_encode = helpers.vae_encode_video
        from shared.utils import utils as wangp_utils

        original_prepare = wangp_utils.prepare_video_guide_and_mask

        def video_conditionings_by_control_video(*args, **kwargs):
            # Every LTX control/IC-LoRA guide clip passes here; only inside it is the clip built in system RAM.
            if not _load_config().get("low_vram_control_video", True):
                return original_control_video(*args, **kwargs)
            _state.control_device = kwargs.get("device")
            try:
                return original_control_video(*args, **kwargs)
            finally:
                _state.control_device = None

        def load_video_conditioning(*args, **kwargs):
            device = getattr(_state, "control_device", None)
            if device is None or args or torch.device(device).type != "cuda":
                return original_load_video(*args, **kwargs)
            return _load_control_video_cpu(kwargs["video_path"], kwargs["height"], kwargs["width"], kwargs.get("frame_cap"),
                                           kwargs["dtype"], device)

        def vae_encode_video(video, video_encoder, tiling_config=None, *args, **kwargs):
            device = getattr(_state, "control_device", None)
            # newer WanGP passes device= / dtype= and moves each tile to the GPU itself
            if device is None or video.device.type != "cpu" or args or kwargs.get("device") is not None:
                return original_encode(video, video_encoder, tiling_config, *args, **kwargs)
            if tiling_config is None or (tiling_config.spatial_config is None and tiling_config.temporal_config is None):
                return original_encode(video.to(device), video_encoder, tiling_config, **kwargs)
            return original_encode(video, _InputToDevice(video_encoder, device), tiling_config, **kwargs)

        def prepare_video_guide_and_mask(video_guides, video_masks, pre_video_guide, *args, **kwargs):
            _state.trim_tail = 0
            try:
                padded = _last_window_padding(video_guides, video_masks, pre_video_guide)
            except Exception as error:  # never break a generation over this fix
                print(f"{LOG} last window fix skipped: {error}")
                padded = None
            if padded is not None:
                video_guides, video_masks = padded
            return original_prepare(video_guides, video_masks, pre_video_guide, *args, **kwargs)

        def generate(self, *args, **kwargs):
            return _trim_tail(tiled_generate(self, *args, **kwargs))

        def tiled_generate(self, *args, **kwargs):
            lora = _tiled_lora(kwargs.get("loras_selected"))
            video_prompt_type = kwargs.get("video_prompt_type") or ""
            if args or lora is None or not ("V" in video_prompt_type and "G" in video_prompt_type) or kwargs.get("input_frames") is None:
                return original_generate(self, *args, **kwargs)
            if "~" in video_prompt_type:  # WanGP's own spatial tiling of phase 2 is on: don't tile the same canvas twice
                print(f"{LOG} WanGP's own phase 2 tiling is on; LTX Tiled Refine stands aside for this render")
                return original_generate(self, *args, **kwargs)
            if int(kwargs.get("guide_phases", 1) or 1) != 1:
                if FORCE_ONE_PHASE:
                    print(f"{LOG} {lora} runs as one full-resolution stage; switching Phases to 1")
                    kwargs["guide_phases"] = 1
                else:
                    print(f"{LOG} {lora} with {kwargs.get('guide_phases')} Phases: stages above 1024x576 are tiled")
            import time

            _state.active = True
            _state.window = next(window for pattern, window in TILE_WINDOWS.items() if pattern in lora)
            _state.config = _load_config()
            _state.timings = []
            _state.phase = 0
            _state.guide_phases = int(kwargs.get("guide_phases", 1) or 1)
            start = time.perf_counter()
            try:
                return original_generate(self, *args, **kwargs)
            finally:
                _print_timings(time.perf_counter() - start)
                _state.active = False
                _state.shape = None
                _state.timings = None
                _state.window = None

        def denoise_audio_video(*args, **kwargs):
            if not getattr(_state, "active", False):
                return original_denoise(*args, **kwargs)
            _state.shape = kwargs.get("output_shape")
            _state.phase = getattr(_state, "phase", 0) + 1
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
                tile_long, tile_short = getattr(_state, "window", None) or (TILE_LONG, TILE_SHORT)
                tile_height = (tile_long if portrait else tile_short) // SPATIAL_SCALE
                tile_width = (tile_short if portrait else tile_long) // SPATIAL_SCALE
                config = getattr(_state, "config", None) or _load_config()
                mode = config["tiles"]
                wrapped = TiledRefineTransformer(transformer, frames, height, width, tile_height, tile_width, TILE_MODES[mode])
                area = height * width / (tile_height * tile_width)
                draft = getattr(_state, "guide_phases", 1) >= 2 and getattr(_state, "phase", 0) == 1
                if wrapped.tile_count > 1 and draft and config.get("phase1_single_window", True) and area <= PHASE1_SINGLE_WINDOW_MAX_AREA:
                    print(f"{LOG} {shape.width}x{shape.height}, {shape.frames} frames: phase 1 draft in one window "
                          f"({area:.2f}x the trained window) instead of {wrapped.tile_count} tiles")
                elif wrapped.tile_count > 1:
                    print(f"{LOG} {shape.width}x{shape.height}, {shape.frames} frames: {wrapped.tile_count} windows of "
                          f"{tile_width * SPATIAL_SCALE}x{tile_height * SPATIAL_SCALE} fused every step "
                          f"(tiles: {mode}, smallest overlap {wrapped.min_overlap_px}px)")
                    kwargs["transformer"] = wrapped
            return original_denoising_func(*args, **kwargs)

        ltx2_module.LTX2.generate = generate
        distilled.denoise_audio_video = denoise_audio_video
        distilled.simple_denoising_func = simple_denoising_func
        timed_control_video = _timed("clip encoding", video_conditionings_by_control_video)
        for module in (helpers, distilled, ti2vid_two_stages):
            module.video_conditionings_by_control_video = timed_control_video
        distilled.denoise_audio_video = _timed("denoise", denoise_audio_video)
        distilled.upsample_video = _timed("phase 1->2 upscale", distilled.upsample_video)
        distilled.vae_decode_video_to_tensor = _timed("video decoding", distilled.vae_decode_video_to_tensor)
        distilled.vae_decode_audio = _timed("audio decoding", distilled.vae_decode_audio)
        helpers.load_video_conditioning = load_video_conditioning
        helpers.vae_encode_video = vae_encode_video
        wangp_utils.prepare_video_guide_and_mask = prepare_video_guide_and_mask  # wgp imports it at every window
        _patched = True
        config = _load_config()
        print(f"{LOG} Active: " + ", ".join(f"{pattern} on {long}x{short}" for pattern, (long, short) in TILE_WINDOWS.items()) + " fused windows "
              f"(tiles: {config['tiles']}; low-VRAM control video: {'on' if config['low_vram_control_video'] else 'off'}).")
