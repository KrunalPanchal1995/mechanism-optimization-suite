import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, PchipInterpolator

INPUT_CSV = r"F:\mechanism-optimization-suite\DATASET_MANIPULATION\DATA_INTERPOLATION\targets.csv"
OUTPUT_CSV = r"F:\mechanism-optimization-suite\DATASET_MANIPULATION\DATA_INTERPOLATION\targets_with_dummy_ntc_saperated_1.csv"
SUMMARY_CSV = r"ntc_region_summary_manual_override.csv"
LNTAU_PLOT = r"temperature_vs_lntau_ntc_manual_override.png"
TAU_PLOT = r"temperature_vs_tau_ntc_manual_override.png"

POINTS_PER_ZONE = 15
ETHYL_SMOOTHING_S = 0.05

# ing_Butanoate generated from shifted Ethyl shape
TEMPLATE_TEMP_SHIFT_K = 38.0
BUTANOATE_OVER_ETHYL_SHIFT_DEX = 0.05
MIN_BUTANOATE_OVER_ETHYL_GAP_DEX = 0.02
BLEND_START_OFFSET_FROM_FIRST_MB_POINT = 6.0

# ==================================================
# Manual override section
# Set USE_MANUAL_* to True if you want to force the
# NTC boundaries instead of auto-detection.
# ==================================================
USE_MANUAL_ETHYL_NTC_BOUNDS = True
MANUAL_ETHYL_NTC_START_K = 746.0
MANUAL_ETHYL_NTC_END_K = 820.0

# Optional direct override for ing_Butanoate. If False,
# ing_Butanoate bounds are taken as shifted Ethyl bounds.
USE_MANUAL_MB_NTC_BOUNDS = False
MANUAL_MB_NTC_START_K = 784.0
MANUAL_MB_NTC_END_K = 840.0


# ==================================================
# Utilities
# ==================================================
def make_prototype_row(group: pd.DataFrame) -> pd.Series:
    row = group.iloc[0].copy()
    row["Pressure_Pa"] = float(group["Pressure_Pa"].median())
    return row


def fit_smoothed_logtau_spline(T: np.ndarray,
                               tau_us: np.ndarray,
                               smoothing_s: float) -> UnivariateSpline:
    return UnivariateSpline(T, np.log(tau_us), s=smoothing_s, k=min(3, len(T) - 1))


def evaluate_logtau_with_linear_boundary_extrapolation(spline,
                                                       T_eval: np.ndarray,
                                                       T_data: np.ndarray) -> np.ndarray:
    T_eval = np.asarray(T_eval, dtype=float)
    T_data = np.asarray(T_data, dtype=float)

    y = np.empty_like(T_eval, dtype=float)
    T_min, T_max = float(np.min(T_data)), float(np.max(T_data))

    inside = (T_eval >= T_min) & (T_eval <= T_max)
    left = T_eval < T_min
    right = T_eval > T_max

    y[inside] = spline(T_eval[inside])

    ds = spline.derivative()
    y_left0 = float(spline(T_min))
    y_right0 = float(spline(T_max))
    m_left = float(ds(T_min))
    m_right = float(ds(T_max))

    y[left] = y_left0 + m_left * (T_eval[left] - T_min)
    y[right] = y_right0 + m_right * (T_eval[right] - T_max)

    return y


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def build_equal_zone_grid(t0: float, t1: float, t2: float, t3: float, points_per_zone: int):
    """
    Build equal number of generated temperatures in:
      before_ntc : [t0, t1)
      ntc        : [t1, t2)
      after_ntc  : [t2, t3]
    """
    if not (t0 < t1 < t2 < t3):
        raise ValueError(f"Invalid zone bounds: {(t0, t1, t2, t3)}")

    zone_before = np.linspace(t0, t1, points_per_zone, endpoint=False)
    zone_ntc = np.linspace(t1, t2, points_per_zone, endpoint=False)
    zone_after = np.linspace(t2, t3, points_per_zone, endpoint=True)

    T_grid = np.concatenate([zone_before, zone_ntc, zone_after])
    zone_labels = (
        ["before_ntc"] * points_per_zone +
        ["ntc"] * points_per_zone +
        ["after_ntc"] * points_per_zone
    )
    return T_grid, zone_labels


def make_rows_from_template(template_row: pd.Series,
                            dataset_id: str,
                            temps: np.ndarray,
                            tau_us: np.ndarray,
                            zone_labels,
                            source_label: str) -> pd.DataFrame:
    rows = []
    for T, tau, zone in zip(temps, tau_us, zone_labels):
        r = template_row.copy()
        r["dataset_ID"] = dataset_id
        r["Temperature_K"] = float(T)
        r["observed"] = float(tau)
        r["obs_unit"] = "us"
        rows.append(r)

    out = pd.DataFrame(rows)
    out["is_dummy"] = True
    out["dummy_source"] = source_label
    out["generated_zone"] = zone_labels
    return out


# ==================================================
# NTC detection / override
# ==================================================
def detect_ntc_region(T: np.ndarray,
                      tau_us: np.ndarray,
                      smoothing_s: float = 0.05,
                      derivative_tol: float = 1e-4,
                      n_fine: int = 2000):
    """
    Auto-detect NTC as the temperature interval where
    d ln(tau) / dT > 0 on the smoothed curve.
    """
    T = np.asarray(T, dtype=float)
    tau_us = np.asarray(tau_us, dtype=float)

    order = np.argsort(T)
    T = T[order]
    tau_us = tau_us[order]

    spline = fit_smoothed_logtau_spline(T, tau_us, smoothing_s)

    Tf = np.linspace(T.min(), T.max(), n_fine)
    yf = spline(Tf)
    dy = spline.derivative()(Tf)

    positive = dy > derivative_tol

    segments = []
    start = None
    for i, flag in enumerate(positive):
        if flag and start is None:
            start = i
        elif (not flag) and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(positive) - 1))

    if not segments:
        raise ValueError("No NTC region detected automatically.")

    best = max(segments, key=lambda seg: Tf[seg[1]] - Tf[seg[0]])
    i0, i1 = best

    left_window_start = max(0, i0 - 120)
    left_window_end = min(len(Tf) - 1, i0 + 120)
    i_min_local = left_window_start + np.argmin(yf[left_window_start:left_window_end + 1])

    right_window_start = max(0, i1 - 120)
    right_window_end = min(len(Tf) - 1, i1 + 120)
    i_max_local = right_window_start + np.argmax(yf[right_window_start:right_window_end + 1])

    t0 = float(T.min())
    t1 = float(Tf[i_min_local])
    t2 = float(Tf[i_max_local])
    t3 = float(T.max())

    if not (t0 < t1 < t2 < t3):
        t1 = float(Tf[i0])
        t2 = float(Tf[i1])

    return {
        "t_min": t0,
        "ntc_start": t1,
        "ntc_end": t2,
        "t_max": t3,
        "spline": spline,
    }


def apply_manual_bounds(auto_info: dict,
                        use_manual: bool,
                        manual_start: float,
                        manual_end: float,
                        label: str):
    """
    Override only ntc_start and ntc_end while keeping t_min, t_max, spline.
    """
    info = auto_info.copy()

    if not use_manual:
        info["boundary_mode"] = "auto"
        return info

    t0 = info["t_min"]
    t3 = info["t_max"]

    if not (t0 < manual_start < manual_end < t3):
        raise ValueError(
            f"Manual {label} NTC bounds must satisfy t_min < start < end < t_max. "
            f"Got t_min={t0}, start={manual_start}, end={manual_end}, t_max={t3}"
        )

    info["ntc_start"] = float(manual_start)
    info["ntc_end"] = float(manual_end)
    info["boundary_mode"] = "manual"
    return info


# ==================================================
# Ethyl generation
# ==================================================
def build_ethyl_dense_balanced(eth: pd.DataFrame):
    eth = eth.sort_values("Temperature_K").copy()

    T_eth = eth["Temperature_K"].to_numpy()
    tau_eth = eth["observed"].to_numpy()

    auto_ntc_info = detect_ntc_region(
        T_eth,
        tau_eth,
        smoothing_s=ETHYL_SMOOTHING_S,
        derivative_tol=1e-4,
        n_fine=3000,
    )

    ntc_info = apply_manual_bounds(
        auto_info=auto_ntc_info,
        use_manual=USE_MANUAL_ETHYL_NTC_BOUNDS,
        manual_start=MANUAL_ETHYL_NTC_START_K,
        manual_end=MANUAL_ETHYL_NTC_END_K,
        label="Ethyl",
    )

    T_grid, zone_labels = build_equal_zone_grid(
        ntc_info["t_min"],
        ntc_info["ntc_start"],
        ntc_info["ntc_end"],
        ntc_info["t_max"],
        POINTS_PER_ZONE,
    )

    y_ln = evaluate_logtau_with_linear_boundary_extrapolation(
        ntc_info["spline"],
        T_grid,
        T_eth,
    )
    tau_dense = np.exp(y_ln)

    proto = make_prototype_row(eth)
    out = make_rows_from_template(
        proto,
        dataset_id="dummy_ing_Ethyl_Butanoate_dense",
        temps=T_grid,
        tau_us=tau_dense,
        zone_labels=zone_labels,
        source_label=(
            f"Smoothed Ethyl ln(tau) spline; NTC bounds mode={ntc_info['boundary_mode']}; "
            "equal points in before_ntc / ntc / after_ntc"
        ),
    )

    return out, ntc_info


# ==================================================
# ing_Butanoate generation
# ==================================================
def build_ing_butanoate_balanced(eth: pd.DataFrame,
                                 mb: pd.DataFrame,
                                 eth_ntc_info: dict):
    eth = eth.sort_values("Temperature_K").copy()
    mb = mb.sort_values("Temperature_K").copy()

    T_eth = eth["Temperature_K"].to_numpy()
    T_mb = mb["Temperature_K"].to_numpy()
    tau_mb = mb["observed"].to_numpy()

    eth_spline = eth_ntc_info["spline"]
    mb_pchip = PchipInterpolator(T_mb, np.log(tau_mb), extrapolate=True)

    # Default ing_Butanoate zone limits = shifted Ethyl limits
    auto_mb_info = {
        "t_min": float(T_eth.min() + TEMPLATE_TEMP_SHIFT_K),
        "ntc_start": float(eth_ntc_info["ntc_start"] + TEMPLATE_TEMP_SHIFT_K),
        "ntc_end": float(eth_ntc_info["ntc_end"] + TEMPLATE_TEMP_SHIFT_K),
        "t_max": float(T_mb.max()),
        "spline": None,
    }

    mb_ntc_info = apply_manual_bounds(
        auto_info=auto_mb_info,
        use_manual=USE_MANUAL_MB_NTC_BOUNDS,
        manual_start=MANUAL_MB_NTC_START_K,
        manual_end=MANUAL_MB_NTC_END_K,
        label="ing_Butanoate",
    )

    t0 = mb_ntc_info["t_min"]
    t1 = mb_ntc_info["ntc_start"]
    t2 = mb_ntc_info["ntc_end"]
    t3 = mb_ntc_info["t_max"]

    T_grid, zone_labels = build_equal_zone_grid(t0, t1, t2, t3, POINTS_PER_ZONE)

    def y_template(T):
        return (
            evaluate_logtau_with_linear_boundary_extrapolation(
                eth_spline,
                np.asarray(T) - TEMPLATE_TEMP_SHIFT_K,
                T_eth,
            )
            + BUTANOATE_OVER_ETHYL_SHIFT_DEX * np.log(10.0)
        )

    T_mb0 = float(T_mb.min())
    T_mb1 = float(T_mb[1]) if len(T_mb) > 1 else float(T_mb.min() + 12.0)
    blend_start = T_mb0 - BLEND_START_OFFSET_FROM_FIRST_MB_POINT
    blend_end = T_mb1

    y_out = np.empty_like(T_grid, dtype=float)

    mask_low = T_grid <= blend_start
    mask_high = T_grid >= blend_end
    mask_blend = (~mask_low) & (~mask_high)

    if np.any(mask_low):
        y_out[mask_low] = y_template(T_grid[mask_low])

    if np.any(mask_high):
        y_out[mask_high] = mb_pchip(T_grid[mask_high])

    if np.any(mask_blend):
        Tb = T_grid[mask_blend]
        w = smoothstep((Tb - blend_start) / (blend_end - blend_start))
        y_low = y_template(Tb)
        y_high = mb_pchip(Tb)
        y_out[mask_blend] = (1.0 - w) * y_low + w * y_high

    # Keep ing_Butanoate slightly above Ethyl in overlap
    overlap_mask = T_grid <= float(T_eth.max())
    if np.any(overlap_mask):
        y_eth_same_T = evaluate_logtau_with_linear_boundary_extrapolation(
            eth_spline,
            T_grid[overlap_mask],
            T_eth,
        )
        gap_ln = MIN_BUTANOATE_OVER_ETHYL_GAP_DEX * np.log(10.0)
        y_out[overlap_mask] = np.maximum(y_out[overlap_mask], y_eth_same_T + gap_ln)

    tau_dense = np.exp(y_out)

    proto = make_prototype_row(mb)
    out = make_rows_from_template(
        proto,
        dataset_id="dummy_ing_Butanoate_NTC_dense",
        temps=T_grid,
        tau_us=tau_dense,
        zone_labels=zone_labels,
        source_label=(
            f"Shifted Ethyl template smoothly blended into measured ing_Butanoate; "
            f"NTC bounds mode={mb_ntc_info['boundary_mode']}; "
            "equal points in before_ntc / ntc / after_ntc"
        ),
    )

    mb_ntc_info["blend_start"] = blend_start
    mb_ntc_info["blend_end"] = blend_end

    return out, mb_ntc_info


# ==================================================
# Plotting helpers
# ==================================================
def add_zone_markers(ax, bounds: dict, y_text: float, prefix: str):
    """
    Mark regions as before NTC / NTC / after NTC using boundary lines + labels.
    """
    t0 = bounds["t_min"]
    t1 = bounds["ntc_start"]
    t2 = bounds["ntc_end"]
    t3 = bounds["t_max"]

    # boundary lines
    ax.axvline(t1, linestyle="--", alpha=0.5)
    ax.axvline(t2, linestyle="--", alpha=0.5)

    # text labels at zone centers
    x_before = 0.5 * (t0 + t1)
    x_ntc = 0.5 * (t1 + t2)
    x_after = 0.5 * (t2 + t3)

    ax.text(x_before, y_text, f"{prefix} before NTC", ha="center", va="bottom", fontsize=9)
    ax.text(x_ntc, y_text, f"{prefix} NTC", ha="center", va="bottom", fontsize=9)
    ax.text(x_after, y_text, f"{prefix} after NTC", ha="center", va="bottom", fontsize=9)


def plot_temperature_vs_lntau(eth_original, but_original, eth_dummy, but_dummy,
                              eth_ntc_info, mb_ntc_info,
                              save_path=LNTAU_PLOT):
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(eth_original["Temperature_K"], np.log(eth_original["observed"]), "o", label="Ethyl original")
    ax.plot(eth_dummy["Temperature_K"], np.log(eth_dummy["observed"]), "-", label="Ethyl generated")
    ax.plot(but_original["Temperature_K"], np.log(but_original["observed"]), "s", label="ing_Butanoate original")
    ax.plot(but_dummy["Temperature_K"], np.log(but_dummy["observed"]), "-", label="ing_Butanoate generated")

    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin

    add_zone_markers(ax, eth_ntc_info, ymin + 0.90 * yr, "Ethyl")
    add_zone_markers(ax, mb_ntc_info, ymin + 0.82 * yr, "MB")

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("ln(tau)")
    ax.set_title("Temperature vs ln(tau): Original and Manual-Override Generated Data")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.show()


def plot_temperature_vs_tau_semilog(eth_original, but_original, eth_dummy, but_dummy,
                                    eth_ntc_info, mb_ntc_info,
                                    save_path=TAU_PLOT):
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.semilogy(eth_original["Temperature_K"], eth_original["observed"], "o", label="Ethyl original")
    ax.semilogy(eth_dummy["Temperature_K"], eth_dummy["observed"], "-", label="Ethyl generated")
    ax.semilogy(but_original["Temperature_K"], but_original["observed"], "s", label="ing_Butanoate original")
    ax.semilogy(but_dummy["Temperature_K"], but_dummy["observed"], "-", label="ing_Butanoate generated")

    ymin, ymax = ax.get_ylim()
    # Put labels in log space using multiplicative placement
    y_text_eth = ymin * (ymax / ymin) ** 0.92
    y_text_mb = ymin * (ymax / ymin) ** 0.84

    add_zone_markers(ax, eth_ntc_info, y_text_eth, "Ethyl")
    add_zone_markers(ax, mb_ntc_info, y_text_mb, "MB")

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("tau (us)")
    ax.set_title("Temperature vs tau: Original and Manual-Override Generated Data")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.show()


# ==================================================
# Main
# ==================================================
def main():
    df = pd.read_csv(INPUT_CSV)

    eth = df[df["dataset_ID"] == "ing_Ethyl_Butanoate"].copy()
    mb = df[df["dataset_ID"] == "ing_Butanoate"].copy()

    if eth.empty or mb.empty:
        raise ValueError("Could not find both ing_Ethyl_Butanoate and ing_Butanoate in the input CSV")

    eth_dense, eth_ntc_info = build_ethyl_dense_balanced(eth)
    mb_dense, mb_ntc_info = build_ing_butanoate_balanced(eth, mb, eth_ntc_info)

    original = df.copy()
    original["is_dummy"] = False
    original["dummy_source"] = ""
    original["generated_zone"] = ""

    out = pd.concat([original, eth_dense, mb_dense], ignore_index=True)
    out.to_csv(OUTPUT_CSV, index=False)

    summary = pd.DataFrame([
        {
            "dataset_ID": "ing_Ethyl_Butanoate",
            "t_min": eth_ntc_info["t_min"],
            "ntc_start": eth_ntc_info["ntc_start"],
            "ntc_end": eth_ntc_info["ntc_end"],
            "t_max": eth_ntc_info["t_max"],
            "boundary_mode": eth_ntc_info["boundary_mode"],
        },
        {
            "dataset_ID": "ing_Butanoate_generated",
            "t_min": mb_ntc_info["t_min"],
            "ntc_start": mb_ntc_info["ntc_start"],
            "ntc_end": mb_ntc_info["ntc_end"],
            "t_max": mb_ntc_info["t_max"],
            "boundary_mode": mb_ntc_info["boundary_mode"],
        },
    ])
    summary.to_csv(SUMMARY_CSV, index=False)

    print("Wrote:", OUTPUT_CSV)
    print("Wrote:", SUMMARY_CSV)

    plot_temperature_vs_lntau(
        eth.sort_values("Temperature_K"),
        mb.sort_values("Temperature_K"),
        eth_dense.sort_values("Temperature_K"),
        mb_dense.sort_values("Temperature_K"),
        eth_ntc_info,
        mb_ntc_info,
        save_path=LNTAU_PLOT,
    )
    print("Wrote:", LNTAU_PLOT)

    plot_temperature_vs_tau_semilog(
        eth.sort_values("Temperature_K"),
        mb.sort_values("Temperature_K"),
        eth_dense.sort_values("Temperature_K"),
        mb_dense.sort_values("Temperature_K"),
        eth_ntc_info,
        mb_ntc_info,
        save_path=TAU_PLOT,
    )
    print("Wrote:", TAU_PLOT)


if __name__ == "__main__":
    main()

