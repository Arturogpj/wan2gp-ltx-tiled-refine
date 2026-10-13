"""Decode long clips with the NAD diffusion decoder in pieces, split at camera cuts.

The NAD decoder runs its first three stages on the whole latent at once, so its memory grows with
clip length: 121 frames at 2560x1408 fit a 16 GB card, 481 frames do not. Here the latent is cut
into pieces that fit, preferably where the shot changes (the picture jumps there anyway, so no
seam can show). A shot longer than one piece is split evenly and the pieces are cross-faded.

LTX latents are causal: latent 0 is pixel frame 0, latent i >= 1 holds pixel frames 8(i-1)+1..8i.
A piece starting at latent s > 0 is decoded with a few earlier latents in front for context; the
decoder turns the first of those into one throw-away frame.
"""

import math

import torch

LOG = "[LTX Tiled Refine]"
REFERENCE_PIXELS = 2560 * 1408       # measured: 16 latents (121 frames) decode at this size on 16 GB
REFERENCE_LATENTS = 16
CONTEXT_LATENTS = 2                  # causal warm-up in front of every piece after the first
BLEND_LATENTS = 2                    # cross-fade on each side of a split inside a shot
CUT_RATIO = 3.0                      # latent change this many times the clip's median = a camera cut


def _pixel_start(latent_index):
    return 0 if latent_index == 0 else 8 * (latent_index - 1) + 1


def max_piece_latents(height, width):
    budget = REFERENCE_LATENTS * REFERENCE_PIXELS / max(1, height * width)
    return int(max(6, min(64, math.floor(budget))))


def find_pixel_cuts(frames):
    """First frame of each new shot in a (C, T, H, W) clip: a frame-to-frame jump far above the clip's usual change."""
    if frames is None or not torch.is_tensor(frames) or frames.ndim != 4 or frames.shape[1] < 3:
        return []
    with torch.no_grad():
        small = frames[:, :, ::16, ::16].float()
        changes = (small[:, 1:] - small[:, :-1]).abs().mean(dim=(0, 2, 3))
    median = float(changes.median().clamp_min(1e-6))
    cuts = []
    for index, value in enumerate(changes.tolist()):
        local = changes[max(0, index - 2):index + 3]
        if value > 6.0 * median and value >= float(local.max()):
            cuts.append(index + 1)
    return cuts


def find_cuts(latent):
    """Latent indices i where a new shot starts (big jump between latent i-1 and i)."""
    frames = latent.shape[2]
    if frames < 4:
        return []
    with torch.no_grad():
        changes = torch.stack([(latent[:, :, i].float() - latent[:, :, i - 1].float()).abs().mean() for i in range(2, frames)]).cpu()
    median = float(changes.median().clamp_min(1e-6))
    cuts = []
    for offset, value in enumerate(changes.tolist()):
        index = offset + 2
        neighbours = changes[max(0, offset - 1):offset + 2]
        if value > CUT_RATIO * median and value >= float(neighbours.max()):
            cuts.append(index)
    return cuts


def plan_pieces(frames, cuts, max_latents):
    """[(start, end, hard_left)] covering latents 0..frames; hard_left = starts at a cut (no cross-fade)."""
    usable = max_latents - CONTEXT_LATENTS - 2 * BLEND_LATENTS
    usable = max(2, usable)
    bounds = [0] + [cut for cut in cuts if 0 < cut < frames] + [frames]
    pieces = []
    for start, end in zip(bounds, bounds[1:]):
        count = max(1, math.ceil((end - start) / usable))
        edges = [start + round(k * (end - start) / count) for k in range(count + 1)]
        for k, (a, b) in enumerate(zip(edges, edges[1:])):
            pieces.append((a, b, k == 0 and a > 0))
    return pieces


def decode_in_pieces(original_decode, latent_list, video_decoder, tiling_config, expected_frames, expected_height, expected_width, pixel_cuts=None, **kwargs):
    latent = latent_list[0]
    latent_frames, height, width = latent.shape[2], int(expected_height), int(expected_width)
    max_latents = max_piece_latents(height, width)
    total_frames = int(expected_frames) if expected_frames is not None else _pixel_start(latent_frames)
    if pixel_cuts:  # shot changes found in the source clip: the latent holding each new shot's first frame
        cuts = sorted({(frame - 1) // 8 + 1 for frame in pixel_cuts if 0 < frame < total_frames and (frame - 1) // 8 + 1 < latent_frames})
    else:
        cuts = find_cuts(latent)
    pieces = plan_pieces(latent_frames, cuts, max_latents)
    print(f"{LOG} NAD decoder: {latent_frames} latents at {width}x{height} decoded in {len(pieces)} pieces "
          f"(max {max_latents} latents each); camera cuts at frames {[_pixel_start(c) for c in cuts] or 'none'}")

    output = None
    blends = {}  # frame -> [weighted sum (float32), total weight]; only frames inside a cross-fade
    written = torch.zeros(total_frames, dtype=torch.bool)
    for index, (start, end, hard_left) in enumerate(pieces):
        hard_right = index + 1 < len(pieces) and pieces[index + 1][2]
        blend_left = 0 if start == 0 or hard_left else BLEND_LATENTS
        blend_right = 0 if end >= latent_frames or hard_right else BLEND_LATENTS
        own_start, own_end = start - blend_left, min(latent_frames, end + blend_right)
        decode_start = max(0, own_start - CONTEXT_LATENTS)
        piece = latent[:, :, decode_start:own_end].clone()
        piece_frames = total_frames if decode_start == 0 and own_end == latent_frames else _pixel_start(own_end - decode_start)
        decoded = original_decode([piece], video_decoder, tiling_config, expected_frames=piece_frames,
                                  expected_height=height, expected_width=width, **kwargs)
        piece = None
        if decoded is None:
            return None
        if output is None:
            output = torch.empty((total_frames,) + tuple(decoded.shape[1:]), dtype=decoded.dtype)
        # decoded frame k is clip frame k + offset (k >= 1 when the piece starts after latent 0)
        offset = 0 if decode_start == 0 else 8 * decode_start
        first = _pixel_start(own_start)
        last = total_frames if own_end >= latent_frames else min(total_frames, _pixel_start(own_end))
        left_ramp = (_pixel_start(own_start), _pixel_start(start + blend_left)) if blend_left else None
        right_ramp = (_pixel_start(end - blend_right), last) if blend_right else None
        for frame in range(first, last):
            k = frame - offset
            if k < (0 if decode_start == 0 else 1) or k >= decoded.shape[0]:
                continue
            weight = 1.0
            if left_ramp and frame < left_ramp[1]:
                weight = (frame - left_ramp[0] + 1) / (left_ramp[1] - left_ramp[0] + 1)
            if right_ramp and frame >= right_ramp[0]:
                weight = min(weight, (right_ramp[1] - frame) / (right_ramp[1] - right_ramp[0] + 1))
            if weight >= 1.0 and frame not in blends:
                output[frame].copy_(decoded[k])
                written[frame] = True
                continue
            entry = blends.setdefault(frame, [torch.zeros(decoded.shape[1:], dtype=torch.float32), 0.0])
            entry[0].add_(decoded[k].float(), alpha=weight)
            entry[1] += weight
            if entry[1] >= 0.999:  # both sides of the cross-fade are in
                output[frame].copy_(entry[0].div_(entry[1]).round_().clamp_(0, 255) if output.dtype == torch.uint8 else entry[0].div_(entry[1]))
                written[frame] = True
                del blends[frame]
        decoded = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for frame, (accumulated, weight) in blends.items():  # clip edges: only one side exists
        output[frame].copy_(accumulated.div_(max(weight, 1e-6)).round_().clamp_(0, 255) if output.dtype == torch.uint8 else accumulated.div_(max(weight, 1e-6)))
        written[frame] = True
    missing = int((~written).sum())
    if missing:
        print(f"{LOG} NAD pieces left {missing} frame(s) empty; please report this")
    return output


def wrap_decode(original_decode, cut_provider=None):
    def vae_decode_video_to_tensor(latent, video_decoder, tiling_config=None, expected_frames=None, expected_height=None,
                                   expected_width=None, **kwargs):
        latent_list = latent if isinstance(latent, list) else [latent]
        tensor = latent_list[0]
        nad = getattr(video_decoder, "is_diffusion_decoder", False)
        if (not nad or kwargs.get("keyframes") is not None or expected_height is None or expected_width is None
                or tensor.shape[2] <= max_piece_latents(int(expected_height), int(expected_width))):
            return original_decode(latent, video_decoder, tiling_config, expected_frames=expected_frames,
                                   expected_height=expected_height, expected_width=expected_width, **kwargs)
        try:
            pixel_cuts = cut_provider() if cut_provider is not None else None
            result = decode_in_pieces(original_decode, latent_list, video_decoder, tiling_config, expected_frames,
                                      expected_height, expected_width, pixel_cuts=pixel_cuts, **kwargs)
        finally:
            latent_list.clear() if isinstance(latent, list) else None
        return result

    return vae_decode_video_to_tensor
