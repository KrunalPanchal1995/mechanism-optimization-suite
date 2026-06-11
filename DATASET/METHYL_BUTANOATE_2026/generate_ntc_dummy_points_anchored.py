import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, PchipInterpolator

INPUT_CSV = r"targets.csv"
OUTPUT_CSV = r"targets_with_dummy_ntc_anchored.csv"
LNTAU_PLOT = r"temperature_vs_lntau_anchored.png"
TAU_PLOT = r"temperature_vs_tau_semilog_anchored.png"

# --------------------------------------------------
# User-tunable knobs
# --------------------------------------------------
ETHYL_GRID = np.arange(680.0, 826.0, 5.0)
BUTANOATE_GRID = np.arange(760.0, 911.0, 5.0)

# Ethyl smoothing: mild, because original data may contain noise
ETHYL_SMOOTHING_S = 0.05

# Butanoate construction
TEMPLATE_TEMP_SHIFT_K = 38.0
BUTANOATE_OVER_ETHYL_SHIFT_DEX = 0.06
MIN_BUTANOATE_OVER_ETHYL_GAP_DEX = 0.03

# IMPORTANT:
# Only apply the Ethyl-vs-Butanoate ordering in the low-temperature NTC overlap.
# Do NOT force this in the measured high-temperature Butanoate branch.
ORDER_ENFORCEMENT_MAX_T = 832.82

# Blend template into measured Butanoate branch
BLEND_START_K = 828.0
BLEND_END_K = 845.0

FIX_PRESSURE_WITH_MEDIAN = True


def make_prototype_row(group: pd.DataFrame) -> pd.Series:
    row = group.iloc[0].copy()
    if FIX_PRESSURE_WITH_MEDIAN:
        row["Pressure_Pa"] = float(group["Pressure_Pa"].median())
    return row


def make_rows_from_template(template_row: pd.Series,
                            dataset_id: str,
                            temps: np.ndarray,
                            tau_us: np.ndarray,
                            source_label: str) -> pd.DataFrame:
    rows = []
    for T, tau in zip(temps, tau_us):
        r = template_row.copy()
        r["dataset_ID"] = dataset_id
        r["Temperature_K"] = float(T)
        r["observed"] = float(tau)
        r["obs_unit"] = "us"
        rows.append(r)
    out = pd.DataFrame(rows)
    out["is_dummy"] = True
    out["dummy_source"] = source_label
    return out


def fit_smoothed_logtau_spline(T: np.ndarray,
                               tau_us: np.ndarray,
                               smoothing_s: float) -> UnivariateSpline:
    y_ln = np.log(tau_us)
    return UnivariateSpline(T, y_ln, s=smoothing_s, k=min(3, len(T) - 1))


def evaluate_logtau_with_linear_boundary_extrapolation(spline,
                                                       T_grid: np.ndarray,
                                                       T_data: np.ndarray) -> np.ndarray:
    T_grid = np.asarray(T_grid, dtype=float)
    y_out = np.empty_like(T_grid, dtype=float)

    T_min = float(np.min(T_data))
    T_max = float(np.max(T_data))

    inside = (T_grid >= T_min) & (T_grid <= T_max)
    left = T_grid < T_min
    right = T_grid > T_max

    y_out[inside] = spline(T_grid[inside])

    deriv = spline.derivative()
    y_left0 = float(spline(T_min))
    m_left = float(deriv(T_min))
    y_right0 = float(spline(T_max))
    m_right = float(deriv(T_max))

    y_out[left] = y_left0 + m_left * (T_grid[left] - T_min)
    y_out[right] = y_right0 + m_right * (T_grid[right] - T_max)
    return y_out


def build_ethyl_dense(eth: pd.DataFrame, T_grid: np.ndarray):
    eth = eth.sort_values("Temperature_K").copy()
    T = eth["Temperature_K"].to_numpy()
    tau = eth["observed"].to_numpy()

    spline = fit_smoothed_logtau_spline(T, tau, ETHYL_SMOOTHING_S)
    y_ln = evaluate_logtau_with_linear_boundary_extrapolation(spline, T_grid, T)
    tau_dense = np.exp(y_ln)

    proto = make_prototype_row(eth)
    out = make_rows_from_template(
        proto,
        dataset_id="dummy_ing_Ethyl_Butanoate_dense",
        temps=T_grid,
        tau_us=tau_dense,
        source_label="Smoothed spline in ln(tau) vs T with linear boundary extrapolation",
    )
    return out, spline


def enforce_butanoate_above_ethyl_in_lowT_only(T_vals: np.ndarray,
                                               y_but_ln: np.ndarray,
                                               ethyl_spline,
                                               ethyl_T_data: np.ndarray,
                                               min_gap_dex: float,
                                               max_enforce_T: float) -> np.ndarray:
    """
    Enforce Butanoate > Ethyl only in the low-T overlap / NTC region.
    This avoids distorting the measured high-T Butanoate branch.
    """
    y_out = y_but_ln.copy()
    mask = (~np.isnan(y_out)) & (T_vals <= max_enforce_T)

    if np.any(mask):
        y_eth = evaluate_logtau_with_linear_boundary_extrapolation(
            ethyl_spline,
            T_vals[mask],
            ethyl_T_data,
        )
        gap_ln = min_gap_dex * np.log(10.0)
        y_out[mask] = np.maximum(y_out[mask], y_eth + gap_ln)

    return y_out


def build_butanoate_dense_from_ethyl(eth: pd.DataFrame,
                                     mb: pd.DataFrame,
                                     ethyl_spline,
                                     T_grid: np.ndarray) -> pd.DataFrame:
    eth = eth.sort_values("Temperature_K").copy()
    mb = mb.sort_values("Temperature_K").copy()

    T_eth = eth["Temperature_K"].to_numpy()
    T_mb = mb["Temperature_K"].to_numpy()
    tau_mb = mb["observed"].to_numpy()

    # Exact measured Butanoate branch:
    # PCHIP in ln(tau) passes through the original Butanoate points exactly.
    mb_pchip = PchipInterpolator(T_mb, np.log(tau_mb), extrapolate=True)

    # Ethyl-based low-T template for Butanoate
    template_T = T_eth + TEMPLATE_TEMP_SHIFT_K
    template_y_ln = evaluate_logtau_with_linear_boundary_extrapolation(
        ethyl_spline,
        T_eth,
        T_eth,
    ) + BUTANOATE_OVER_ETHYL_SHIFT_DEX * np.log(10.0)

    template_pchip = PchipInterpolator(template_T, template_y_ln, extrapolate=True)

    y_out = np.full_like(T_grid, np.nan, dtype=float)

    # Region 1: low-T template
    mask_template = T_grid < BLEND_START_K
    if np.any(mask_template):
        y_out[mask_template] = template_pchip(T_grid[mask_template])

    # Region 2: blend from template to exact measured Butanoate branch
    mask_blend = (T_grid >= BLEND_START_K) & (T_grid <= BLEND_END_K)
    if np.any(mask_blend):
        Tb = T_grid[mask_blend]
        y_tmp = template_pchip(Tb)
        y_meas = mb_pchip(Tb)
        w = (Tb - BLEND_START_K) / (BLEND_END_K - BLEND_START_K)
        y_out[mask_blend] = (1.0 - w) * y_tmp + w * y_meas

    # Region 3: measured Butanoate branch, anchored to original points
    mask_measured = T_grid > BLEND_END_K
    if np.any(mask_measured):
        y_out[mask_measured] = mb_pchip(T_grid[mask_measured])

    # Only enforce species ordering where it belongs: low-T / NTC overlap
    y_out = enforce_butanoate_above_ethyl_in_lowT_only(
        T_vals=T_grid,
        y_but_ln=y_out,
        ethyl_spline=ethyl_spline,
        ethyl_T_data=T_eth,
        min_gap_dex=MIN_BUTANOATE_OVER_ETHYL_GAP_DEX,
        max_enforce_T=ORDER_ENFORCEMENT_MAX_T,
    )

    tau_dense = np.exp(y_out)

    proto = make_prototype_row(mb)
    out = make_rows_from_template(
        proto,
        dataset_id="dummy_ing_Butanoate_NTC_dense",
        temps=T_grid,
        tau_us=tau_dense,
        source_label=(
            f"Low-T branch from shifted Ethyl template (+{TEMPLATE_TEMP_SHIFT_K:.1f} K, "
            f"+{BUTANOATE_OVER_ETHYL_SHIFT_DEX:.3f} dex), blended into exact Butanoate "
            f"PCHIP branch from {BLEND_START_K:.1f} to {BLEND_END_K:.1f} K; "
            f"Butanoate-over-Ethyl ordering enforced only up to {ORDER_ENFORCEMENT_MAX_T:.2f} K"
        ),
    )
    return out


def plot_temperature_vs_lntau(eth_original, but_original, eth_dummy, but_dummy, save_path=LNTAU_PLOT):
    plt.figure(figsize=(8, 5))
    plt.plot(eth_original["Temperature_K"], np.log(eth_original["observed"]), "o", label="Ethyl original")
    plt.plot(eth_dummy["Temperature_K"], np.log(eth_dummy["observed"]), "-", label="Ethyl generated")
    plt.plot(but_original["Temperature_K"], np.log(but_original["observed"]), "s", label="ing_Butanoate original")
    plt.plot(but_dummy["Temperature_K"], np.log(but_dummy["observed"]), "-", label="ing_Butanoate generated")
    plt.xlabel("Temperature (K)")
    plt.ylabel("ln(tau)")
    plt.title("Temperature vs ln(tau): Original and Anchored Generated Data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


def plot_temperature_vs_tau_semilog(eth_original, but_original, eth_dummy, but_dummy, save_path=TAU_PLOT):
    plt.figure(figsize=(8, 5))
    plt.semilogy(eth_original["Temperature_K"], eth_original["observed"], "o", label="Ethyl original")
    plt.semilogy(eth_dummy["Temperature_K"], eth_dummy["observed"], "-", label="Ethyl generated")
    plt.semilogy(but_original["Temperature_K"], but_original["observed"], "s", label="ing_Butanoate original")
    plt.semilogy(but_dummy["Temperature_K"], but_dummy["observed"], "-", label="ing_Butanoate generated")
    plt.xlabel("Temperature (K)")
    plt.ylabel("tau (us)")
    plt.title("Temperature vs tau: Original and Anchored Generated Data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


def main():
    df = pd.read_csv(INPUT_CSV)

    eth = df[df["dataset_ID"] == "ing_Ethyl_Butanoate"].copy()
    mb = df[df["dataset_ID"] == "ing_Butanoate"].copy()

    if eth.empty or mb.empty:
        raise ValueError("Could not find both ing_Ethyl_Butanoate and ing_Butanoate in the input CSV")

    eth_dense, ethyl_spline = build_ethyl_dense(eth, ETHYL_GRID)
    mb_dense = build_butanoate_dense_from_ethyl(eth, mb, ethyl_spline, BUTANOATE_GRID)

    original = df.copy()
    original["is_dummy"] = False
    original["dummy_source"] = ""

    out = pd.concat([original, eth_dense, mb_dense], ignore_index=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print("Wrote:", OUTPUT_CSV)

    plot_temperature_vs_lntau(
        eth.sort_values("Temperature_K"),
        mb.sort_values("Temperature_K"),
        eth_dense.sort_values("Temperature_K"),
        mb_dense.sort_values("Temperature_K"),
        save_path=LNTAU_PLOT,
    )
    print("Wrote:", LNTAU_PLOT)

    plot_temperature_vs_tau_semilog(
        eth.sort_values("Temperature_K"),
        mb.sort_values("Temperature_K"),
        eth_dense.sort_values("Temperature_K"),
        mb_dense.sort_values("Temperature_K"),
        save_path=TAU_PLOT,
    )
    print("Wrote:", TAU_PLOT)


if __name__ == "__main__":
    main()
