"""
Profiling script to identify the bottleneck in AEF training step.
Tests: encoder-only, decoder-only, full forward, loss computation.
Compares 64x64 vs 128x128 resolution impact.
"""
from __future__ import annotations

import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch_npu

from src.aef.architecture.aef_module import AlphaEarthFoundations
from src.aef.architecture.encoder import STPEncoder
from src.aef.architecture.decoder import VonMisesFisherDecoder
from src.aef.loss_function import AEFLoss


def time_it(fn, desc, warmup=2, repeats=5):
    """Time a function, with warmup and repeats."""
    device = torch.npu.current_device() if torch.npu.is_available() else "cpu"
    
    for _ in range(warmup):
        fn()
        if torch.npu.is_available():
            torch.npu.synchronize(device)
    
    times = []
    for _ in range(repeats):
        if torch.npu.is_available():
            torch.npu.synchronize(device)
        t0 = time.perf_counter()
        fn()
        if torch.npu.is_available():
            torch.npu.synchronize(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    print(f"  {desc}: {mean:.3f}s ± {std:.3f}s  (min={min(times):.3f}, max={max(times):.3f})")
    return mean


def profile_encoder(B=2, T=16, H=128, W=128, device="npu:0"):
    print(f"\n=== Encoder Profile (B={B}, T={T}, H={H}, W={W}) ===")
    
    encoder = STPEncoder(input_channels=160, d_s=512, d_t=256, d_p=64, num_blocks=6).to(device)
    x = torch.randn(B, T, H, W, 160, device=device)
    ts = torch.randint(0, 1000, (B, T), device=device).float()
    
    time_it(lambda: encoder(x, ts), "Encoder forward")
    
    # Memory
    if torch.npu.is_available():
        torch.npu.reset_peak_memory_stats(device)
        encoder(x, ts)
        torch.npu.synchronize(device)
        mem = torch.npu.max_memory_allocated(device) / 1024**3
        print(f"  Peak memory: {mem:.2f} GB")


def profile_decoder(B=2, H=128, W=128, device="npu:0"):
    print(f"\n=== Decoder Profile (B={B}, H={H}, W={W}) ===")
    
    decoder = VonMisesFisherDecoder(embedding_dim=64, source_dims={"sentinel2": 5}, geometry_dim=16).to(device)
    emb = torch.randn(B, H, W, 64, device=device)
    geo = torch.zeros(B, 16, device=device)
    ts_center = torch.zeros(B, device=device)
    vp = (torch.zeros(B, device=device), torch.ones(B, device=device))
    
    time_it(lambda: decoder(emb, geo, ts_center, vp, "sentinel2", 1), "Decoder forward")
    
    # Profile each MLP layer
    with torch.no_grad():
        B_total = B * H * W
        flat_input = torch.randn(B_total, 144, device=device)
        mlp = decoder.source_decoders["sentinel2"]
        time_it(lambda: mlp(flat_input), f"  Decoder MLP (batch={B_total})")
    
    if torch.npu.is_available():
        torch.npu.reset_peak_memory_stats(device)
        decoder(emb, geo, ts_center, vp, "sentinel2", 1)
        torch.npu.synchronize(device)
        mem = torch.npu.max_memory_allocated(device) / 1024**3
        print(f"  Peak memory: {mem:.2f} GB")


def profile_full_model(B=2, T=16, H=128, W=128, device="npu:0"):
    print(f"\n=== Full Model Profile (B={B}, T={T}, H={H}, W={W}) ===")
    
    model = AlphaEarthFoundations(
        input_sources={"sentinel2": 5},
        decode_sources={"sentinel2": 5},
    ).to(device)
    model.eval()
    
    x = {"sentinel2": torch.randn(B, T, H, W, 5, device=device)}
    ts = {"sentinel2": torch.randint(0, 1000, (B, T), device=device).float()}
    vp = torch.tensor([[0, 1000], [0, 1000]], dtype=torch.float32, device=device)
    
    # Full forward (no grad)
    time_it(lambda: model(x, ts, vp), "Full forward (eval)")
    
    # Forward + backward
    model.train()
    loss_fn = AEFLoss()
    
    def forward_backward():
        out = model(x, ts, vp)
        predictions = {"sentinel2": out["reconstructions"]["sentinel2"][:, 0]}
        targets = {"sentinel2": torch.randn(B, H, W, 5, device=device)}
        masks = {"sentinel2": torch.ones(B, H, W, 1, device=device)}
        losses = loss_fn({
            "embeddings": out["embeddings"],
            "teacher_embeddings": out["teacher_embeddings"],
            "student_embeddings": out["student_embeddings"],
            "predictions": predictions,
            "targets": targets,
            "masks": masks,
        })
        losses["total"].backward()
    
    time_it(forward_backward, "Forward + backward")
    
    if torch.npu.is_available():
        torch.npu.reset_peak_memory_stats(device)
        forward_backward()
        torch.npu.synchronize(device)
        mem = torch.npu.max_memory_allocated(device) / 1024**3
        print(f"  Peak memory: {mem:.2f} GB")


def profile_components(B=2, T=16, H=128, W=128, device="npu:0"):
    print(f"\n=== Component Breakdown (B={B}, T={T}, H={H}, W={W}) ===")
    
    model = AlphaEarthFoundations(
        input_sources={"sentinel2": 5},
        decode_sources={"sentinel2": 5},
    ).to(device)
    model.eval()
    
    x = {"sentinel2": torch.randn(B, T, H, W, 5, device=device)}
    ts = {"sentinel2": torch.randint(0, 1000, (B, T), device=device).float()}
    vp = torch.tensor([[0, 1000], [0, 1000]], dtype=torch.float32, device=device)
    
    # Stack inputs
    x_stacked = model._stack_inputs(x)
    first_src = next(iter(model.input_sources.keys()))
    ts_tensor = ts[first_src]
    
    # 1. Input projection + individual encoders
    def fn_input():
        _ = model._stack_inputs(x)
    time_it(fn_input, "1. Input stack + source encoders")
    
    # 2. Teacher encoder
    def fn_teacher_encoder():
        with torch.no_grad():
            _ = model.encoder(x_stacked, ts_tensor)
    time_it(fn_teacher_encoder, "2. Teacher encoder (no_grad)")
    
    # 3. Student encoder (with perturbation)
    def fn_student_encoder():
        student_srcs, student_ts_dict = model._perturb_inputs(x, ts)
        x_student = model._stack_inputs(student_srcs)
        ts_student = student_ts_dict[first_src]
        _ = model.encoder(x_student, ts_student)
    time_it(fn_student_encoder, "3. Student encoder")
    
    # 4. Summarizer
    with torch.no_grad():
        feats_teacher = model.encoder(x_stacked, ts_tensor)
    def fn_summarizer():
        _ = model.summarizer(feats_teacher, ts_tensor, vp)
    time_it(fn_summarizer, "4. Summarizer")
    
    # 5. Decoder
    with torch.no_grad():
        mu_t = model.summarizer(feats_teacher, ts_tensor, vp)
    geo = torch.zeros(B, 16, device=device, dtype=mu_t.dtype)
    ts_center = ts_tensor.mean(dim=1)
    def fn_decoder():
        _ = model.decoder(mu_t, geo, ts_center, (vp[:, 0], vp[:, 1]), "sentinel2", 1)
    time_it(fn_decoder, "5. Decoder")


def compare_resolution(B=2, T=16, device="npu:0"):
    print("\n" + "="*60)
    print("RESOLUTION COMPARISON: 64x64 vs 128x128")
    print("="*60)
    
    for H, W in [(64, 64), (128, 128)]:
        print(f"\n--- Resolution {H}x{W} ---")
        
        model = AlphaEarthFoundations(
            input_sources={"sentinel2": 5},
            decode_sources={"sentinel2": 5},
        ).to(device)
        model.eval()
        
        x = {"sentinel2": torch.randn(B, T, H, W, 5, device=device)}
        ts = {"sentinel2": torch.randint(0, 1000, (B, T), device=device).float()}
        vp = torch.tensor([[0, 1000], [0, 1000]], dtype=torch.float32, device=device)
        
        time_it(lambda: model(x, ts, vp), f"Full forward {H}x{W}")
        
        if torch.npu.is_available():
            torch.npu.reset_peak_memory_stats(device)
            model(x, ts, vp)
            torch.npu.synchronize(device)
            mem = torch.npu.max_memory_allocated(device) / 1024**3
            print(f"  Peak memory: {mem:.2f} GB")


def main():
    device = "npu:0" if torch.npu.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if torch.npu.is_available():
        print(f"NPU: {torch.npu.get_device_name(0)}")
    
    # 1. Component breakdown at 128x128
    profile_components(B=2, T=16, H=128, W=128, device=device)
    
    # 2. Encoder standalone
    profile_encoder(B=2, T=16, H=128, W=128, device=device)
    
    # 3. Decoder standalone
    profile_decoder(B=2, H=128, W=128, device=device)
    
    # 4. Full model forward+backward
    profile_full_model(B=2, T=16, H=128, W=128, device=device)
    
    # 5. Resolution comparison
    compare_resolution(B=2, T=16, device=device)
    
    print("\n" + "="*60)
    print("Profiling complete")
    print("="*60)


if __name__ == "__main__":
    main()
