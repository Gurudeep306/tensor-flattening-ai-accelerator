#!/usr/bin/env python3
"""
Indian Institute of Technology Tirupati
AI Accelerator Design | Assignment 3
BCHW <-> B x (CHW) Tensor Flattening and Reconstruction

Instructor : Prof. Jaynarayan T. Tudu
Student    : Gurudeep (Roll No: CS25M111)
"""

from pathlib import Path
from typing import Tuple
import torch
import torchvision
import torchvision.transforms as transforms


# ==============================================================================
# 1. FLATTENING ALGORITHM: B x C x H x W  -->  B x (CHW)
# ==============================================================================
def flatten_bchw_to_b_chw(I: torch.Tensor) -> torch.Tensor:
    """
    Convert a 4D tensor I of shape (B, C, H, W) into a 2D flattened tensor F
    of shape (B, C * H * W) from scratch using explicit loops and indexing.

    Address mapping for each batch index b:
        k = c * (H * W) + h * W + w
        F[b, k] = I[b, c, h, w]
    """
    B, C, H, W = I.shape
    HW = H * W
    CHW = C * HW

    F = torch.empty((B, CHW), dtype=I.dtype, device=I.device)

    for b in range(B):
        for c in range(C):
            c_offset = c * HW
            for h in range(H):
                h_offset = c_offset + h * W
                for w in range(W):
                    k = h_offset + w
                    F[b, k] = I[b, c, h, w]

    return F


# ==============================================================================
# 2. RECONSTRUCTION ALGORITHM: B x (CHW)  -->  B x C x H x W
# ==============================================================================
def reconstruct_b_chw_to_bchw(
    F: torch.Tensor, B: int, C: int, H: int, W: int
) -> torch.Tensor:
    """
    Reconstruct a 4D tensor I_hat of shape (B, C, H, W) from a 2D flattened
    tensor F of shape (B, C * H * W) from scratch using inverse index arithmetic.

    Inverse mapping from linear index k in [0, C * H * W):
        c = k // (H * W)
        r = k %  (H * W)
        h = r // W
        w = r %  W
        I_hat[b, c, h, w] = F[b, k]
    """
    HW = H * W
    CHW = C * HW

    I_hat = torch.empty((B, C, H, W), dtype=F.dtype, device=F.device)

    for b in range(B):
        for k in range(CHW):
            c = k // HW
            r = k % HW
            h = r // W
            w = r % W
            I_hat[b, c, h, w] = F[b, k]

    return I_hat


# ==============================================================================
# 3. RECONSTRUCTION ERROR METRICS (E_max and MAE)
# ==============================================================================
def compute_reconstruction_error(
    I: torch.Tensor, I_hat: torch.Tensor
) -> Tuple[torch.Tensor, float, float]:
    """
    Compute element-wise error E = I - I_hat, Maximum Absolute Error (E_max),
    and Mean Absolute Error (MAE) explicitly from scratch.
    """
    B, C, H, W = I.shape
    total_elements = B * C * H * W
    E = torch.empty_like(I)

    e_max = 0.0
    abs_sum = 0.0

    for b in range(B):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    diff = float(I[b, c, h, w].item()) - float(I_hat[b, c, h, w].item())
                    E[b, c, h, w] = diff
                    abs_diff = abs(diff)
                    if abs_diff > e_max:
                        e_max = abs_diff
                    abs_sum += abs_diff

    e_mae = abs_sum / float(total_elements)
    return E, e_max, e_mae


# ==============================================================================
# 4. MANUAL TEST CASE WALKTHROUGH (B = 1, C = 2, H = 2, W = 3)
# ==============================================================================
def run_manual_example() -> None:
    B, C, H, W = 1, 2, 2, 3
    HW = H * W

    I = torch.empty((B, C, H, W), dtype=torch.float32)
    val = 10.0
    for b in range(B):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    I[b, c, h, w] = val
                    val += 10.0

    F = flatten_bchw_to_b_chw(I)
    I_hat = reconstruct_b_chw_to_bchw(F, B, C, H, W)
    _, e_max, e_mae = compute_reconstruction_error(I, I_hat)

    print("=" * 82)
    print("Manual Indexing Example: B = 1, C = 2, H = 2, W = 3  (CHW = 12)")
    print("Forward : k = c * (H * W) + h * W + w = 6*c + 3*h + w")
    print("Inverse : c = k // 6,  r = k % 6,  h = r // 3,  w = r % 3")
    print("=" * 82)
    print(
        f"{'BCHW (b,c,h,w)':<16} | {'k = 6c + 3h + w':<19} | "
        f"{'2D (b,k)':<10} | {'I[b,c,h,w]':<10} | {'Decoded (b,c,h,w)'}"
    )
    print("-" * 82)

    for b in range(B):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    k = c * HW + h * W + w
                    c_d = k // HW
                    r_d = k % HW
                    h_d = r_d // W
                    w_d = r_d % W
                    expr = f"6*{c} + 3*{h} + {w} = {k}"
                    print(
                        f"({b}, {c}, {h}, {w}){'':<4} | {expr:<19} | "
                        f"({b}, {k:>2}){'':<3} | {I[b, c, h, w].item():>8.1f}   | "
                        f"({b}, {c_d}, {h_d}, {w_d})"
                    )
    print("-" * 82)
    print(f"Manual Test Case Error -> E_max = {e_max:.1f}, MAE = {e_mae:.1f}\n")


# ==============================================================================
# 5. DATASET PREPARATION & MULTI-CHANNEL EXPERIMENTS
# ==============================================================================
def load_mnist_sample(batch_size: int = 2) -> torch.Tensor:
    """Load MNIST single-channel (C=1, H=28, W=28) tensor in BCHW format."""
    for root in [Path("../lab_7/Data"), Path("lab_7/Data"), Path("./data")]:
        if (root / "MNIST").exists():
            ds = torchvision.datasets.MNIST(
                root=str(root), train=False, download=False, transform=transforms.ToTensor()
            )
            return torch.stack([ds[i][0] for i in range(batch_size)], dim=0)
    gen = torch.Generator().manual_seed(1)
    return torch.rand((batch_size, 1, 28, 28), generator=gen, dtype=torch.float32)


def load_cifar_sample(batch_size: int = 2) -> torch.Tensor:
    """Load CIFAR-10 three-channel RGB (C=3, H=32, W=32) tensor in BCHW format."""
    for root in [Path("../lab_7/Data"), Path("lab_7/Data"), Path("./data")]:
        if (root / "cifar-10-batches-py").exists():
            ds = torchvision.datasets.CIFAR10(
                root=str(root), train=False, download=False, transform=transforms.ToTensor()
            )
            return torch.stack([ds[i][0] for i in range(batch_size)], dim=0)
    gen = torch.Generator().manual_seed(3)
    return torch.rand((batch_size, 3, 32, 32), generator=gen, dtype=torch.float32)


def generate_synthetic_fmap(B: int, C: int, H: int, W: int) -> torch.Tensor:
    """Generate synthetic multi-channel feature map tensor in BCHW format."""
    gen = torch.Generator().manual_seed(42 + C)
    return torch.randn((B, C, H, W), generator=gen, dtype=torch.float32)


# ==============================================================================
# OPTIONAL REFERENCE VERIFICATION (Isolated from submitted algorithm)
# ==============================================================================
def verify_with_reference_reshape(
    I: torch.Tensor, F_custom: torch.Tensor, I_hat_custom: torch.Tensor
) -> bool:
    """Optional check against library reshape() for independent verification only."""
    B, C, H, W = I.shape
    F_ref = I.reshape(B, C * H * W)
    I_hat_ref = F_ref.reshape(B, C, H, W)
    return bool(torch.equal(F_custom, F_ref) and torch.equal(I_hat_custom, I_hat_ref))


def main() -> None:
    run_manual_example()

    experiments = [
        ("MNIST / sample",    load_mnist_sample(batch_size=2)),
        ("CIFAR / sample",    load_cifar_sample(batch_size=2)),
        ("Synthetic (C=8)",   generate_synthetic_fmap(2, 8,   16, 16)),
        ("Synthetic fmap1",   generate_synthetic_fmap(2, 16,  16, 16)),
        ("Synthetic (C=32)",  generate_synthetic_fmap(2, 32,  16, 16)),
        ("Synthetic fmap2",   generate_synthetic_fmap(2, 64,  16, 16)),
        ("Synthetic fmap3",   generate_synthetic_fmap(2, 128, 16, 16)),
        ("Synthetic (C=256)", generate_synthetic_fmap(2, 256, 16, 16)),
        ("Synthetic fmap4",   generate_synthetic_fmap(2, 500, 16, 16)),
    ]

    print("=" * 78)
    print("Table 1: Experimental results showing E_max and MAE for image/feature tensors")
    print("=" * 78)
    print(f"{'Dataset/Input':<18} | {'B':>3} | {'C':>5} | {'H':>4} | {'W':>4} | {'E_max':>10} | {'MAE':>10}")
    print("-" * 78)

    for name, I in experiments:
        B, C, H, W = I.shape
        F = flatten_bchw_to_b_chw(I)
        I_hat = reconstruct_b_chw_to_bchw(F, B, C, H, W)
        _, e_max, e_mae = compute_reconstruction_error(I, I_hat)
        assert verify_with_reference_reshape(I, F, I_hat), "Verification failed!"

        print(f"{name:<18} | {B:>3} | {C:>5} | {H:>4} | {W:>4} | {e_max:>10.1f} | {e_mae:>10.1f}")

    print("=" * 78)


if __name__ == "__main__":
    main()
