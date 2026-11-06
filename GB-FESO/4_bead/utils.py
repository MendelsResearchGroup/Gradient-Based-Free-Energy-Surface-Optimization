"""Angular basin populations and relative free energies in kBT units."""

from typing import Dict, List, Optional, Sequence, Tuple

import math
import torch


DEFAULT_BASIN_HALF_WIDTH = 0.45

# Each box is ((center_theta123, center_theta234), (half_width_x, half_width_y)).
# half_width scales all boxes relative to the original value of 0.45 radians.
ANGLE_BASINS: Tuple[Dict[str, object], ...] = (
    {
        "name": "A_anti_diagonal",
        "label": "A anti-diagonal",
        "boxes": (
            ((-1.02, 0.98), (0.48, 0.45)),
            ((-0.45, 0.62), (0.35, 0.28)),

            ((1.02, -1.00), (0.47, 0.48)),
            ((0.58, -0.48), (0.33, 0.35)),
        ),
    },
    {
        "name": "B_same_sign",
        "label": "B same-sign",
        "boxes": (
            ((0.70, 0.95), (0.50, 0.47)),
            ((-0.95, -0.95), (0.53, 0.50)),
        ),
    },
    {
        "name": "C_outer",
        "label": "C outer",
        "boxes": (
            ((1.10, 2.02), (0.35, 0.52)),
            ((1.50, 1.62), (0.25, 0.32)),
            ((2.05, 1.02), (0.50, 0.42)),
            ((-2.05, -1.02), (0.50, 0.42)),
            ((-1.50, -1.75), (0.25, 0.29)),
            ((-1.10, -2.02), (0.35, 0.52)),
        ),
    },
)

OUTSIDE_LABEL = "outside"


def _resolve_basin_half_width(half_width: Optional[float]) -> float:
    if half_width is None:
        return DEFAULT_BASIN_HALF_WIDTH
    half_width = float(half_width)
    if half_width <= 0:
        raise ValueError("half_width must be positive")
    return half_width


def _scaled_basin_boxes(basin: Dict[str, object], half_width: Optional[float]):
    half_width = _resolve_basin_half_width(half_width)
    scale = half_width / DEFAULT_BASIN_HALF_WIDTH
    for center, half_size in basin["boxes"]:
        yield center, (float(half_size[0]) * scale, float(half_size[1]) * scale)


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi)."""
    return torch.remainder(x + math.pi, 2.0 * math.pi) - math.pi


def angle_view(samples: torch.Tensor) -> torch.Tensor:
    """Return [theta123, theta234] from [N, 2] angles or [N, 5] geometry."""
    x = torch.as_tensor(samples)
    if x.ndim != 2 or x.shape[1] not in (2, 5):
        raise ValueError(f"Expected samples with shape [N, 2] or [N, 5], got {tuple(x.shape)}")
    return x[:, -2:]


def basin_names(include_outside: bool = True) -> List[str]:
    names = [str(b["name"]) for b in ANGLE_BASINS]
    if include_outside:
        names.append(OUTSIDE_LABEL)
    return names


def assign_angle_basins(
    samples: torch.Tensor,
    half_width: Optional[float] = DEFAULT_BASIN_HALF_WIDTH,
    outside_index: int = -1,
) -> torch.Tensor:
    """
    Assign samples to hard-coded rectangular angle basins.

    Returns a LongTensor with values 0..K-1 for the macro-basins and
    outside_index for samples outside all hard-coded basin boxes.
    """
    half_width = _resolve_basin_half_width(half_width)

    angles = angle_view(samples).detach()
    device = angles.device
    dtype = angles.dtype
    n = angles.shape[0]
    k = len(ANGLE_BASINS)

    best_d2 = torch.full((n, k), float("inf"), dtype=dtype, device=device)

    for basin_idx, basin in enumerate(ANGLE_BASINS):
        box_scores = []
        for center, half_size in _scaled_basin_boxes(basin, half_width):
            center_t = torch.as_tensor(center, dtype=dtype, device=device)
            half_t = torch.as_tensor(half_size, dtype=dtype, device=device)
            delta = wrap_to_pi(angles - center_t)
            inside = (delta.abs() <= half_t).all(dim=-1)
            d2 = (delta / half_t).square().sum(dim=-1)
            box_scores.append(torch.where(inside, d2, torch.full_like(d2, float("inf"))))
        best_d2[:, basin_idx] = torch.stack(box_scores, dim=1).min(dim=1).values

    nearest_d2, nearest = best_d2.min(dim=1)
    assignments = torch.full((n,), int(outside_index), dtype=torch.long, device=device)
    assignments[torch.isfinite(nearest_d2)] = nearest[torch.isfinite(nearest_d2)]
    return assignments


def basin_counts(
    assignments: torch.Tensor,
    include_outside: bool = True,
    outside_index: int = -1,
) -> torch.Tensor:
    """Count assignments in each macro-basin, optionally with an outside bin."""
    a = torch.as_tensor(assignments, dtype=torch.long)
    k = len(ANGLE_BASINS)
    counts = torch.stack([(a == i).sum() for i in range(k)]).to(dtype=torch.float32)
    if include_outside:
        outside = (a == int(outside_index)).sum().to(dtype=torch.float32).view(1)
        counts = torch.cat([counts, outside], dim=0)
    return counts


def basin_probabilities_from_counts(
    counts: torch.Tensor,
    pseudocount: float = 0.0,
) -> torch.Tensor:
    """Convert counts to probabilities with optional additive pseudocount."""
    if pseudocount < 0:
        raise ValueError("pseudocount must be non-negative")
    c = torch.as_tensor(counts, dtype=torch.float32)
    c = c + float(pseudocount)
    total = c.sum()
    if float(total.item()) <= 0.0:
        raise ValueError("Cannot normalize zero total counts")
    return c / total


def basin_probabilities(
    samples: torch.Tensor,
    half_width: Optional[float] = DEFAULT_BASIN_HALF_WIDTH,
    pseudocount: float = 0.0,
    include_outside: bool = True,
) -> torch.Tensor:
    """Assign samples and return basin probabilities."""
    a = assign_angle_basins(samples, half_width=half_width)
    counts = basin_counts(a, include_outside=include_outside)
    return basin_probabilities_from_counts(counts, pseudocount=pseudocount)


def active_basin_indices(
    target_probs: torch.Tensor,
    min_target_prob: float = 0.05,
    include_outside: bool = False,
) -> List[int]:
    """Return target-populated basin indices."""
    p = torch.as_tensor(target_probs, dtype=torch.float32)
    end = len(ANGLE_BASINS) + (1 if include_outside else 0)
    return [i for i in range(min(end, p.numel())) if float(p[i].item()) > min_target_prob]


def total_variation_distance(
    p: torch.Tensor,
    q: torch.Tensor,
) -> float:
    """0.5 * L1 distance between two probability vectors."""
    pp = torch.as_tensor(p, dtype=torch.float32)
    qq = torch.as_tensor(q, dtype=torch.float32)
    if pp.shape != qq.shape:
        raise ValueError(f"Probability shapes differ: {tuple(pp.shape)} vs {tuple(qq.shape)}")
    return float(0.5 * torch.abs(pp - qq).sum().item())


def relative_free_energies(
    probs: torch.Tensor,
    ref_index: Optional[int] = None,
    prob_floor: float = 1e-3,
) -> torch.Tensor:
    """
    Compute ΔF_i/kBT = -log(P_i/P_ref) with a probability floor.

    The returned vector has the same length as probs. The reference basin has 0.
    """
    if prob_floor <= 0:
        raise ValueError("prob_floor must be positive")
    p = torch.clamp(torch.as_tensor(probs, dtype=torch.float32), min=float(prob_floor))
    if ref_index is None:
        ref_index = int(torch.argmax(p).item())
    if ref_index < 0 or ref_index >= p.numel():
        raise ValueError(f"Invalid ref_index={ref_index} for {p.numel()} probabilities")
    return -torch.log(p / p[ref_index])


def delta_f_error_summary(
    target_probs: torch.Tensor,
    generated_probs: torch.Tensor,
    active_indices: Sequence[int],
    ref_index: Optional[int] = None,
    prob_floor: float = 1e-3,
    cap: Optional[float] = 6.0,
) -> Dict[str, object]:
    """
    Finite ΔF target-vs-generated summary for multi-basin targets.

    Only target-active non-reference basins are compared. The reference basin is
    the dominant target basin, so including it would add a trivial zero error and
    artificially lower the mean error.
    """
    active = [int(i) for i in active_indices]
    names = basin_names(include_outside=False)
    if len(active) < 2:
        return {
            "df_mae": None,
            "df_median_abs_error": None,
            "df_abs_error": [],
            "df_target": [],
            "df_generated": [],
            "df_basin_indices": [],
            "df_basin_names": [],
            "df_ref_index": ref_index,
            "df_ref_name": None if ref_index is None else names[int(ref_index)],
        }

    tp = torch.as_tensor(target_probs, dtype=torch.float32)
    gp = torch.as_tensor(generated_probs, dtype=torch.float32)
    if ref_index is None:
        ref_index = max(active, key=lambda i: float(tp[i].item()))

    compared = [i for i in active if i != int(ref_index)]
    if not compared:
        return {
            "df_mae": None,
            "df_median_abs_error": None,
            "df_abs_error": [],
            "df_target": [],
            "df_generated": [],
            "df_basin_indices": [],
            "df_basin_names": [],
            "df_ref_index": int(ref_index),
            "df_ref_name": names[int(ref_index)],
        }

    df_t = relative_free_energies(tp, ref_index=ref_index, prob_floor=prob_floor)
    df_g = relative_free_energies(gp, ref_index=ref_index, prob_floor=prob_floor)
    idx = torch.as_tensor(compared, dtype=torch.long)
    err = torch.abs(df_g[idx] - df_t[idx])
    if cap is not None:
        err = torch.clamp(err, max=float(cap))

    return {
        "df_mae": float(err.mean().item()),
        "df_median_abs_error": float(err.median().item()),
        "df_abs_error": [float(v) for v in err.tolist()],
        "df_target": [float(v) for v in df_t[idx].tolist()],
        "df_generated": [float(v) for v in df_g[idx].tolist()],
        "df_basin_indices": compared,
        "df_basin_names": [names[i] for i in compared],
        "df_ref_index": int(ref_index),
        "df_ref_name": names[int(ref_index)],
    }


def compare_basin_metrics(
    target_samples: torch.Tensor,
    generated_samples: torch.Tensor,
    half_width: Optional[float] = DEFAULT_BASIN_HALF_WIDTH,
    active_target_prob: float = 0.05,
    min_generated_active_prob: float = 0.005,
    delta_f_prob_floor: float = 1e-3,
    delta_f_error_cap: Optional[float] = 6.0,
) -> Dict[str, object]:
    """
    Compare target and generated ensembles using hard-coded angular basins.

    This function is intended to be called inside/after an optimization loop with
    detached target and generated samples. It returns finite metrics and flags
    missing target-active basins instead of producing infinite ΔF values.
    """
    target_assign = assign_angle_basins(target_samples, half_width=half_width)
    gen_assign = assign_angle_basins(generated_samples, half_width=half_width)

    target_counts = basin_counts(target_assign, include_outside=True)
    gen_counts = basin_counts(gen_assign, include_outside=True)
    target_probs = basin_probabilities_from_counts(target_counts)
    gen_probs = basin_probabilities_from_counts(gen_counts)

    active = active_basin_indices(target_probs, min_target_prob=active_target_prob, include_outside=False)
    missing = [i for i in active if float(gen_probs[i].item()) < min_generated_active_prob]
    ref_index = max(active, key=lambda i: float(target_probs[i].item())) if active else None

    df = delta_f_error_summary(
        target_probs=target_probs,
        generated_probs=gen_probs,
        active_indices=active,
        ref_index=ref_index,
        prob_floor=delta_f_prob_floor,
        cap=delta_f_error_cap,
    )

    names = basin_names(include_outside=False)
    dominant = int(torch.argmax(target_probs[: len(ANGLE_BASINS)]).item())
    dominant_retention = float(gen_probs[dominant].item())
    leakage = float(1.0 - dominant_retention)

    return {
        "basin_names": basin_names(include_outside=True),
        "target_counts": [int(v) for v in target_counts.long().tolist()],
        "generated_counts": [int(v) for v in gen_counts.long().tolist()],
        "target_probs": [float(v) for v in target_probs.tolist()],
        "generated_probs": [float(v) for v in gen_probs.tolist()],
        "active_indices": active,
        "active_names": [names[i] for i in active],
        "target_type": "multi_basin" if len(active) >= 2 else "single_basin",
        "missing_active_basin": len(missing) > 0,
        "missing_indices": missing,
        "missing_names": [names[i] for i in missing],
        "total_variation": total_variation_distance(target_probs, gen_probs),
        "dominant_basin_index": dominant,
        "dominant_basin_name": names[dominant],
        "dominant_basin_retention": dominant_retention,
        "leakage": leakage,
        **df,
    }


def format_basin_metrics(metrics: Dict[str, object]) -> str:
    """Compact one-line string for printing inside optimization."""
    tv = float(metrics["total_variation"])
    missing = bool(metrics["missing_active_basin"])
    if metrics.get("target_type") == "multi_basin":
        df = metrics["df_mae"]
        df_s = "NA" if df is None else f"{float(df):.4f}"
        return f"D_TV={tv:.4f} | dF_MAE={df_s} | missing_active={missing}"
    return (
        f"D_TV={tv:.4f} | dominant_retention="
        f"{float(metrics['dominant_basin_retention']):.4f} | leakage={float(metrics['leakage']):.4f}"
    )
