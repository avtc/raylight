"""Distributed video prep + prod-parity chunked redenoise nodes.

RayESRGANDistributed: per-frame tiled ESRGAN x2 + lanczos crop spread over the ray
actors; returns the full upscaled IMAGE batch. Frames are independent for the image
upscaler, so slices need no temporal overlap. The H3 video VAE encode stays with the
caller's main-process node (the 17k+5 token contract must not be reimplemented).
Peak memory = one slice per actor + the fp16 frame batch on the main side.

MMH3UltimateUpscaleRay: MMH3UltimateUpscale's prod configuration (temporal split ->
per-chunk reanchored conditioning -> sample -> temporal stitch; no spatial split, no
latent upscale) with every chunk sampled through the ray actors via
actor.custom_sampler_advanced - the same execution path as XFuserSamplerCustomAdvanced.
Requires the ComfyUI-H3-UltimateUpscale-LD pack for the loop helpers (lazy import of
the already-loaded module - no duplicate import, shared Custom types).
"""
import sys

import torch

try:
    import ray
except ImportError:
    ray = None


def _ultimate_helpers():
    """The already-loaded ComfyUI-H3-UltimateUpscale-LD module (single source of its
    Custom types). Raises with a clear message if the pack is not installed."""
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and f.replace("\\", "/").endswith("ComfyUI-H3-UltimateUpscale-LD/nodes.py"):
            return m
    raise ImportError("MMH3UltimateUpscaleRay needs the ComfyUI-H3-UltimateUpscale-LD "
                      "custom node pack loaded (loop helpers + temporal stitch).")


def _ray_sample_piece(piece, cond_i, ray_actors, noise_seed, sampler, sigmas):
    guider = {"ray_actors": ray_actors, "type": "basic", "positive": cond_i}
    workers = ray_actors["workers"]
    futures = [
        w.custom_sampler_advanced.remote(True, int(noise_seed), guider, sampler, sigmas, piece)
        for w in workers
    ]
    results = ray.get(futures)
    outputs = [r[0] if isinstance(r, (list, tuple)) else r for r in results]
    out = outputs[0]
    samples = out["samples"] if isinstance(out, dict) else out
    return samples.to("cpu", non_blocking=False)


class RayESRGANDistributed:
    @classmethod
    def INPUT_TYPES(s):
        import folder_paths
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS", {"tooltip": "Ray actors to submit the slices to."}),
                "images": ("IMAGE", {"tooltip": "Source frames (BHWC)."}),
                "upscale_name": (folder_paths.get_filename_list("upscale_models"),
                                 {"tooltip": "upscale_models folder item."}),
                "width": ("INT", {"default": 2560, "min": 64, "max": 8192, "step": 32,
                                  "tooltip": "Target width after x2 + lanczos crop."}),
                "height": ("INT", {"default": 1440, "min": 64, "max": 8192, "step": 32,
                                   "tooltip": "Target height after x2 + lanczos crop."}),
                "slice_frames": ("INT", {"default": 28, "min": 4, "max": 128,
                                         "tooltip": "Frames per actor slice; slices need no temporal overlap."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Raylight/extra/video"

    def run(self, ray_actors, images, upscale_name, width, height, slice_frames):
        import folder_paths
        gpu_actors = ray_actors["workers"]
        if not gpu_actors:
            raise ValueError("Ray ESRGAN requires at least one Ray worker.")
        upscale_path = folder_paths.get_full_path_or_raise("upscale_models", upscale_name)
        import time
        t0 = time.monotonic()
        total = int(images.shape[0])
        n = len(gpu_actors)
        slices = [(s, min(s + slice_frames, total)) for s in range(0, total, slice_frames)]
        half = images.to(torch.float16).cpu()
        futures = []
        for i, (s, e) in enumerate(slices):
            actor = gpu_actors[i % n]
            futures.append((s, actor.ray_esrgan_partial.remote(
                half[s:e], upscale_path, int(width), int(height))))
        parts = []
        for s, fut in futures:
            parts.append((s, ray.get(fut)))
        parts.sort(key=lambda x: x[0])
        out = torch.cat([p for _, p in parts], dim=0).float()
        print(f"[Raylight][ESRGAN] {total}f x{n} actors in {time.monotonic()-t0:.1f}s", flush=True)
        return (out,)


class MMH3UltimateUpscaleRay:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS", {"tooltip": "Ray actors (model+turbo+shift chain)."}),
                "conditioning": ("CONDITIONING", {"tooltip": "Conditioning of the latent being re-enhanced (re-anchored per chunk)."}),
                "latent": ("LATENT", {"tooltip": "Denoised MiniMax H3 AV latent to enhance."}),
                "noise_seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                       "tooltip": "Base seed; chunk i uses base+i."}),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "temporal_split_param": ("H3_TEMPORAL_PARAM",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "Raylight/extra/video"

    def run(self, ray_actors, conditioning, latent, noise_seed, sampler, sigmas,
            temporal_split_param=None):
        if ray is None:
            raise ValueError("ray is not installed")
        import comfy.nested_tensor
        ld = _ultimate_helpers()
        samples = latent["samples"]
        if not ld.is_h3_av_latent(samples):
            raise ValueError("MMH3UltimateUpscaleRay expects a MiniMax H3 AV latent")
        video = samples.tensors[0]
        audio = samples.tensors[1]
        if video.shape[0] != 1:
            raise ValueError("expected a single-video latent (batch 1)")
        conditioning = ld.normalize_minimax_refs(conditioning)

        tv = video.shape[2]
        ta = audio.shape[-1]
        if temporal_split_param is not None:
            chunk_length = int(temporal_split_param["chunk_length"])
            overlap = int(temporal_split_param["temporal_overlap"])
            bounds, frame_count = ld.compute_segments(tv, chunk_length, overlap)
            anchor_strength = temporal_split_param["anchor_strength"]
        else:
            frame_count = ld.frames_for_tokens(tv)
            bounds = [(0, 0, tv, frame_count)]
            anchor_strength = 0.999

        import time
        t0 = time.monotonic()
        acc_v = None
        acc_a = None
        for i, (k0, f0, k1, f1) in enumerate(bounds):
            chunk_v = video[:, :, k0:k1].contiguous()
            a0, a1 = ld.audio_range(f0, f1)
            a1 = min(a1, ta)
            chunk_a = audio[:, :, :, a0:a1].contiguous()

            cond_i = ld.reanchor_conditioning(conditioning, f0, f1, (chunk_v.shape[3], chunk_v.shape[4]))
            if i > 0 and acc_v is not None:
                cond_i = ld.anchor_conditioning(cond_i, acc_v, f0, anchor_strength)

            piece = {"samples": comfy.nested_tensor.NestedTensor((chunk_v, chunk_a))}
            t_c = time.monotonic()
            out = _ray_sample_piece(piece, cond_i, ray_actors, int(noise_seed) + i, sampler, sigmas)
            print(f"[Raylight][ULTIMATE] chunk {i} [{f0}:{f1}] sampled in {time.monotonic()-t_c:.1f}s", flush=True)
            chunk_out_v = out.tensors[0]

            acc_v, acc_a = ld.temporal_append(acc_v, acc_a, chunk_out_v, chunk_a, i, k0, f0)

        out = {"samples": comfy.nested_tensor.NestedTensor((acc_v, acc_a))}
        print(f"[Raylight][ULTIMATE] loop total: {time.monotonic()-t0:.1f}s", flush=True)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "RayESRGANDistributed": RayESRGANDistributed,
    "MMH3UltimateUpscaleRay": MMH3UltimateUpscaleRay,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RayESRGANDistributed": "Ray ESRGAN Distributed (Ray)",
    "MMH3UltimateUpscaleRay": "MM H3 Ultimate Upscale (Ray USP)",
}
