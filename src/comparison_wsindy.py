"""
WSINDy / SINDy workflows on MC triad simulation bundles (equipart, cascade, dual_cascade).

Plotting helpers (``plot_wsindy_*``, ``run_wsindy_prediction_plots``, ``run_equipart_wsindy_fit_plots``)
live in this module.

Non-interactive::

    python src/comparison_wsindy.py --batch --all
    python src/comparison_wsindy.py --batch --params_name equipart --noise_level 1.0 --wsindy_only
    python src/comparison_wsindy.py --batch --params_name equipart --plot_wsindy --no_standardize

Interactive (no ``--batch``): options 1–3 as before, plus **4** for the same plot diagnostics as ``--plot_wsindy``.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ``config`` lives next to this file (``src/``)
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from config import create_main_parser

# Defaults aligned with ``config.create_main_parser()`` (same as 1stage_deterministic / parse_args).
_cfg_args = create_main_parser().parse_args([])


def _load_wsindy_class():
    here = Path(__file__).resolve().parent
    path = here / "utils" / "wsindy.py"
    spec = importlib.util.spec_from_file_location("turbulentfex_wsindy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.WSINDy


WSINDy = _load_wsindy_class()


def plot_wsindy_fit_comparison(
    t_data: np.ndarray,
    x_data: np.ndarray,
    x_wsindy_full: np.ndarray,
    x_wsindy_enforced: Optional[np.ndarray],
    *,
    rmse_full: Optional[float] = None,
    rmse_enforced: Optional[float] = None,
    save_path: Optional[str] = None,
    title: str = "",
) -> Optional[str]:
    """Three panels: ensemble mean vs WSINDy (full library) and optional enforced-triad simulation."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True, constrained_layout=True)
    labels = ("x1", "x2", "x3")
    for i in range(3):
        ax = axes[i]
        ax.plot(t_data, x_data[:, i], label="ensemble mean (data)", color="C0", lw=1.5)
        lf = f"WSINDy full (RMSE={rmse_full:.4g})" if rmse_full is not None else "WSINDy full"
        ax.plot(t_data, x_wsindy_full[:, i], label=lf, color="C1", lw=1.2, ls="--")
        if x_wsindy_enforced is not None:
            le = (
                f"enforced triad (RMSE={rmse_enforced:.4g})"
                if rmse_enforced is not None
                else "enforced triad"
            )
            ax.plot(t_data, x_wsindy_enforced[:, i], label=le, color="C2", lw=1.0, ls=":")
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("t")
    if title:
        fig.suptitle(title, fontsize=10)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    return None


def plot_wsindy_prediction_extended(
    t_data: np.ndarray,
    x_data: np.ndarray,
    t_long: np.ndarray,
    x_det_long: np.ndarray,
    x_stoch_paths: np.ndarray,
    *,
    t_mark_end: Optional[float] = None,
    save_path: Optional[str] = None,
    title: str = "",
    annotation: Optional[str] = None,
) -> Optional[str]:
    """Data vs deterministic WSINDy on ``t_long`` and stochastic paths (Heun RK2 + MC noise)."""
    if t_mark_end is None:
        t_mark_end = float(np.max(t_data))
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    labels = ("x1", "x2", "x3")
    n_paths = x_stoch_paths.shape[0]
    for i in range(3):
        ax = axes[i]
        ax.plot(t_data, x_data[:, i], label="ensemble mean (data)", color="C0", lw=1.8)
        ax.plot(t_long, x_det_long[:, i], label="WSINDy det (IVP)", color="C1", lw=1.3, ls="--")
        mean_s = np.nanmean(x_stoch_paths[:, :, i], axis=0)
        ref = np.nanmax(
            np.abs(
                np.concatenate(
                    [x_data[:, i].ravel(), x_det_long[:, i].ravel()]
                )
            )
        )
        ref = max(float(ref), 1e-12)
        stoch_peak = np.nanmax(np.abs(mean_s))
        stoch_ok = np.isfinite(stoch_peak) and stoch_peak < 100.0 * ref and np.all(
            np.isfinite(x_stoch_paths[:, :, i])
        )
        if stoch_ok:
            ax.plot(
                t_long,
                mean_s,
                label="stoch mean (Heun RK2 + MC noise)",
                color="C2",
                lw=1.2,
            )
            for p in range(min(n_paths, 24)):
                ax.plot(
                    t_long,
                    x_stoch_paths[p, :, i],
                    color="C3",
                    alpha=0.12,
                    lw=0.6,
                )
        else:
            ax.text(
                0.02,
                0.98,
                "stoch Heun RK2 diverged\n(omitted; learned drift+noise unstable)",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="C3",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
        y_baseline = [x_data[:, i], x_det_long[:, i]]
        if stoch_ok:
            y_baseline.append(mean_s)
            y_baseline.append(x_stoch_paths[:, :, i].ravel())
        y_baseline = np.concatenate(y_baseline)
        y_lo, y_hi = np.nanpercentile(y_baseline, [1.0, 99.0])
        pad = 0.08 * max(y_hi - y_lo, ref * 1e-6)
        ax.set_ylim(y_lo - pad, y_hi + pad)
        ax.axvline(t_mark_end, color="0.4", ls=":", lw=1)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("t")
    if annotation:
        fig.text(0.02, 0.02, annotation, fontsize=8, family="monospace", va="bottom")
    if title:
        fig.suptitle(title, fontsize=10)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
    return None


def triad_noise_params(params_name: str, n_tm_indices: int) -> Dict[str, np.ndarray]:
    """
    Noise matrices matching ``MC_triad.params_init`` (SS, SSt, tmS) without importing ``MC_triad``
    (that would load ``utils`` → ``Controller`` → ``config.parse_args()`` under CLI).

    ``tmS`` is padded/truncated to length ``n_tm_indices`` (index used at time step ``k``).
    """
    Dt = 1e-2
    Nt = int(round(10 / Dt))
    if params_name == "equipart":
        g = np.diag([0.2, 0.1, 0.1])
        req = 2.5
        ss = req * np.sqrt(2 * g)
        sst = np.zeros((3, 3))
        tm_s = np.zeros(Nt)
    elif params_name == "cascade":
        ss = np.diag([np.sqrt(10), np.sqrt(10 ** (-2)), np.sqrt(10 ** (-2))])
        sst = np.zeros((3, 3))
        tm_s = np.zeros(Nt)
    elif params_name == "dual_cascade":
        ss = np.diag([np.sqrt(10), np.sqrt(10 ** (-2)), np.sqrt(10 ** (-2))])
        sst = np.zeros((3, 3))
        tm_s = np.zeros(Nt)
    elif params_name == "periodic_cascade":
        ss = np.diag([np.sqrt(10), np.sqrt(10 ** (-2)), np.sqrt(10 ** (-2))])
        sst = np.diag([np.sqrt(1), np.sqrt(2), np.sqrt(2)])
        tm_s = np.zeros(Nt)
    elif params_name == "random_cascade_deterministic":
        ss = np.diag([np.sqrt(1), np.sqrt(10 ** (-2)), np.sqrt(10 ** (-2))])
        sst = np.diag([np.sqrt(1), np.sqrt(2), np.sqrt(2)])
        tm_s = np.zeros(Nt)
    elif params_name == "random_cascade":
        ss = np.diag([np.sqrt(10), np.sqrt(10 ** (-2)), np.sqrt(10 ** (-2))])
        sst = np.diag([np.sqrt(1), np.sqrt(2), np.sqrt(2)])
        fr = 2 * np.pi / 2
        theta = fr / (2 * np.pi)
        sigma = np.sqrt(2 * theta)
        rng = np.random.default_rng(42)
        tm_s_full = np.zeros(Nt + 1)
        for j in range(Nt):
            dW1 = np.sqrt(Dt / 4) * rng.standard_normal(4)
            winc = np.sum(dW1)
            tm_s_full[j + 1] = tm_s_full[j] - theta * tm_s_full[j] * Dt + sigma * winc
        tm_s = 0.8 * tm_s_full
    else:
        raise ValueError(f"Unknown params_name for noise: {params_name}")

    if tm_s.size < n_tm_indices:
        pad = n_tm_indices - tm_s.size
        tm_s = np.pad(tm_s, (0, pad), mode="constant", constant_values=0.0)
    elif tm_s.size > n_tm_indices:
        tm_s = tm_s[:n_tm_indices]
    return {"SS": ss, "SSt": sst, "tmS": tm_s}


def heun_rk2_wsindy(
    model: Any,
    x0: np.ndarray,
    t_grid: np.ndarray,
    noise_level: float,
    params: Dict[str, Any],
    rng: np.random.Generator,
    n_paths: int,
) -> np.ndarray:
    """
    Heun (RK2) for additive noise: same diffusion increment as ``MC_triad_direct`` /
    Euler–Maruyama, but drift uses trapezoid ``(f(t,x)+f(t+dt,x*))/2`` with
    ``x* = x + f(t,x) dt + dW``. Improves drift accuracy vs plain EM; instability
    can still occur for stiff learned models.
    """
    t_grid = np.asarray(t_grid, dtype=float)
    dt = float(t_grid[1] - t_grid[0])
    n_t = t_grid.shape[0]
    u = np.tile(np.asarray(x0, dtype=float), (n_paths, 1))
    out = np.zeros((n_paths, n_t, 3))
    out[:, 0, :] = u
    tm_s = np.asarray(params["tmS"]).reshape(-1)
    sqrt_dt = np.sqrt(dt)
    for k in range(n_t - 1):
        tk = float(t_grid[k])
        tk1 = float(t_grid[k + 1])
        if tm_s.size == 0:
            s = 0.0
        else:
            idx = min(k, tm_s.size - 1)
            s = float(tm_s[idx]) ** 2
        ss_eff = params["SS"] + s * (params["SSt"] - params["SS"])
        f0 = np.stack([model.rhs(tk, u[i]) for i in range(n_paths)])
        winc = rng.standard_normal((n_paths, 3))
        dW = sqrt_dt * noise_level * (winc @ ss_eff)
        u_star = u + dt * f0 + dW
        f1 = np.stack([model.rhs(tk1, u_star[i]) for i in range(n_paths)])
        u = u + 0.5 * dt * (f0 + f1) + dW
        out[:, k + 1, :] = u
    return out


def run_wsindy_prediction_plots(params_name: str, args: argparse.Namespace) -> None:
    """
    Fit WSINDy on the **physical** ensemble mean, save:
      - ``wsindy_fit_comparison.png`` (data vs full vs optional enforced)
      - ``wsindy_prediction_extended.png`` (deterministic + stochastic to ``prediction_t_max``)

    Stochastic noise matches ``MC_triad_direct`` (``SS`` / ``tmS``-modulated diffusion).
    """
    triad_root = Path(args.triad_root) if args.triad_root else None
    npz = default_npz_path(params_name, args.noise_level, triad_root)
    if not npz.is_file():
        print(f"[WARN] plot_wsindy: missing {npz}")
        return

    t, x, meta = mean_trajectory_from_npz(npz)
    polys = np.arange(0, args.poly_max + 1)
    model = fit_wsindy(
        t,
        x,
        polys=polys,
        L=args.L,
        overlap=args.overlap,
        ws_ld=args.ws_ld,
        ws_gamma=args.ws_gamma,
        ws_scale_theta=args.ws_scale_theta,
        ws_use_gls=args.ws_use_gls,
        time_poly_max=args.time_poly_max,
    )
    coef_full = np.asarray(model.coef).copy()
    x0 = x[0].copy()
    x_full = simulate_wsindy(model, x0, t)
    rmse_full = _rmse(x_full, x)
    x_enf: Optional[np.ndarray] = None
    rmse_enf: Optional[float] = None
    if args.triad_interaction_only:
        model.coef = coef_full.copy()
        enforce_triad_one_interaction_structure(
            model,
            keep_constant=args.keep_constant_terms,
            keep_all_linear=True,
        )
        x_enf = simulate_wsindy(model, x0, t)
        rmse_enf = _rmse(x_enf, x)

    out_dir = npz.parent
    title_base = f"{params_name} noise_{args.noise_level}"
    plot_wsindy_fit_comparison(
        t,
        x,
        x_full,
        x_enf,
        rmse_full=rmse_full,
        rmse_enforced=rmse_enf,
        save_path=str(out_dir / "wsindy_fit_comparison.png"),
        title=title_base + " — WSINDy vs ensemble mean",
    )

    dt = float(meta["dt"])
    t_max = float(args.prediction_t_max)
    t_long = np.arange(0.0, t_max + 0.5 * dt, dt)
    model.coef = coef_full.copy()
    if args.triad_interaction_only:
        enforce_triad_one_interaction_structure(
            model,
            keep_constant=args.keep_constant_terms,
            keep_all_linear=True,
        )
    x_det_long = simulate_wsindy(model, x0, t_long)

    # Stochastic rollout: same MC diffusion as ``MC_triad_direct`` on the **data time grid**
    # only (learned drift + noise is often unstable when extrapolated far beyond fit horizon).
    t_data_end = float(t[-1])
    n_st = int(np.searchsorted(t_long, t_data_end, side="right"))
    n_st = max(n_st, 2)
    t_stoch = t_long[:n_st]
    params = triad_noise_params(params_name, max(len(t_stoch) - 1, 1))
    rng = np.random.default_rng(int(args.plot_seed))
    x_stoch = heun_rk2_wsindy(
        model,
        x0,
        t_stoch,
        float(args.noise_level),
        params,
        rng,
        int(args.stoch_paths),
    )
    # Pad stochastic paths to ``t_long`` length with NaN past ``t_data_end`` for plotting overlay.
    n_tl = len(t_long)
    x_stoch_pad = np.full((x_stoch.shape[0], n_tl, 3), np.nan)
    x_stoch_pad[:, : x_stoch.shape[1], :] = x_stoch

    i_end = int(np.argmin(np.abs(t_long - t_max)))
    st_mean_end = np.nanmean(x_stoch_pad[:, i_end, :], axis=0)
    ann = (
        f"t_end={t_long[i_end]:.4g}  det={np.array2string(x_det_long[i_end], precision=4)}\n"
        f"stoch_mean@{t_max:.3g} (NaN if no stoch past data)={np.array2string(st_mean_end, precision=4)}"
    )
    print(f"[INFO] WSINDy det at t≈{t_max}: {x_det_long[i_end]}")
    print(
        f"[INFO] Stochastic paths: MC noise on t∈[0,{t_data_end:.4g}] only; "
        f"past that, plot shows deterministic WSINDy only."
    )

    plot_wsindy_prediction_extended(
        t,
        x,
        t_long,
        x_det_long,
        x_stoch_pad,
        t_mark_end=float(t[-1]),
        save_path=str(out_dir / "wsindy_prediction_extended.png"),
        title=title_base + f" — det to t={t_max}; stoch+noise on t≤{t_data_end:.3g} only",
        annotation=ann,
    )
    print(f"[INFO] Wrote {out_dir / 'wsindy_fit_comparison.png'}")
    print(f"[INFO] Wrote {out_dir / 'wsindy_prediction_extended.png'}")


TRIAD_CASES = (
    "equipart",
    "cascade",
    "dual_cascade",
    "periodic_cascade",
    "random_cascade",
    "random_cascade_deterministic",
)


def default_npz_path(
    params_name: str,
    noise_level: float = 1.0,
    triad_root: Optional[Path | str] = None,
) -> Path:
    """Path to ``simulation_results_noise_{level}.npz`` for a triad case."""
    if triad_root is None:
        triad_root = Path(__file__).resolve().parent / "Example" / "MC_triad"
    else:
        triad_root = Path(triad_root)
    return triad_root / "Results" / params_name / f"noise_{noise_level}" / f"simulation_results_noise_{noise_level}.npz"


def noise_result_dir(
    params_name: str,
    noise_level: float,
    triad_root: Optional[Path | str] = None,
) -> Path:
    """Directory ``.../Results/<case>/noise_<level>/`` (npz and sidecar text files live here)."""
    return default_npz_path(params_name, noise_level, triad_root).parent


def mean_trajectory_from_npz(
    npz_path: Path | str,
    *,
    max_traj: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Return ``(t, x)`` with ``x`` shape ``(n_times, 3)`` (ensemble mean) and metadata.

    Drops trajectories with any non-finite values, then averages.
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path)
    dataset = data["dataset"]
    if dataset.ndim != 3 or dataset.shape[1] != 3:
        raise ValueError(f"Expected dataset (N, 3, T+1); got {dataset.shape}")

    finite_rows = np.isfinite(dataset).all(axis=(1, 2))
    valid = dataset[finite_rows]
    meta = {
        "npz_path": str(npz_path),
        "n_total": int(dataset.shape[0]),
        "n_valid": int(valid.shape[0]),
        "n_dropped": int(dataset.shape[0] - valid.shape[0]),
    }
    if valid.shape[0] == 0:
        raise ValueError("No finite trajectories in dataset; regenerate or fix npz.")

    if max_traj is not None and valid.shape[0] > max_traj:
        rng = np.random.default_rng(0)
        idx = rng.choice(valid.shape[0], size=max_traj, replace=False)
        valid = valid[idx]

    x_mean = np.mean(valid, axis=0).T
    nt = x_mean.shape[0]
    # MC_triad default: T=10, uniform Dt, dataset length Nt+1 → dt = T/(Nt+1 - 1)
    t_horizon = 10.0
    dt = t_horizon / max(nt - 1, 1)
    t = np.arange(nt, dtype=float) * dt
    meta["dt"] = dt
    meta["t_span"] = (float(t[0]), float(t[-1]))
    return t, x_mean, meta


def _standardize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.mean(x, axis=0)
    sig = np.std(x, axis=0)
    sig = np.where(sig < 1e-12, 1.0, sig)
    z = (x - mu) / sig
    return z, mu, sig


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE over indices where both arrays are finite (learned ODEs may blow up)."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return float("nan")
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return float("nan")
    d = a[m] - b[m]
    return float(np.sqrt(np.mean(d**2)))


def fit_wsindy(
    t: np.ndarray,
    x: np.ndarray,
    *,
    polys: np.ndarray | None = None,
    L: int = 30,
    overlap: float = 0.7,
    ws_ld: float = 1e-3,
    ws_gamma: float = 0.0,
    ws_scale_theta: int = 0,
    ws_use_gls: float = 1e-12,
    time_poly_max: int = 0,
) -> Any:
    if polys is None:
        polys = np.arange(0, 3)
    model = WSINDy(
        polys=polys,
        time_polys=np.arange(1, int(time_poly_max) + 1) if int(time_poly_max) > 0 else [],
        ld=ws_ld,
        gamma=ws_gamma,
        scaled_theta=ws_scale_theta,
        useGLS=ws_use_gls,
    )
    model.getWSindyUniform(x, t, L=L, overlap=overlap)
    return model


def fit_wsindy_adaptive(
    t: np.ndarray,
    x: np.ndarray,
    *,
    polys: Optional[np.ndarray] = None,
    r_whm: int = 30,
    s: int = 16,
    K: int = 200,
    p: int = 2,
    tau_p: int = 16,
    ws_ld: float = 1e-3,
    ws_gamma: float = 0.0,
    ws_scale_theta: int = 0,
    ws_use_gls: float = 1e-12,
    time_poly_max: int = 0,
) -> Any:
    """
    Joint weak identification for all three state components using adaptive test functions
    (``getWsindyAdaptive``). Same coupled ``(x1,x2,x3)`` data as uniform WSINDy; used as the
    integrated WSINDy refinement step (menu choice 2).
    """
    if polys is None:
        polys = np.arange(0, 3)
    model = WSINDy(
        polys=polys,
        time_polys=np.arange(1, int(time_poly_max) + 1) if int(time_poly_max) > 0 else [],
        ld=ws_ld,
        gamma=ws_gamma,
        scaled_theta=ws_scale_theta,
        useGLS=ws_use_gls,
    )
    model.getWsindyAdaptive(x, t, r_whm=r_whm, s=s, K=K, p=p, tau_p=tau_p)
    return model


def simulate_wsindy(
    model: Any,
    x0: np.ndarray,
    t: np.ndarray,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> np.ndarray:
    t_span = np.array([float(t[0]), float(t[-1])])
    return model.simulate(x0, t_span=t_span, t_eval=t, rtol=rtol, atol=atol)


def format_wsindy_dimension_rhs(
    coef_col: np.ndarray,
    tags: np.ndarray,
    *,
    threshold: float = 1e-9,
) -> str:
    """Turn one column of ``model.coef`` and tag rows into a readable polynomial string."""
    term_strs: List[Tuple[float, str]] = []
    for k in range(coef_col.shape[0]):
        c = float(coef_col[k])
        if not np.isfinite(c) or abs(c) < threshold:
            continue
        trow = tags[k]
        if np.max(np.abs(np.imag(trow))) > 0:
            continue
        e = np.asarray(np.real(trow), dtype=float)
        e_int = np.rint(e).astype(int)
        if np.max(np.abs(e - e_int)) > 1e-8:
            continue
        factors: List[str] = []
        for j in range(min(3, e_int.shape[0])):
            if e_int[j] == 0:
                continue
            name = f"x{j+1}"
            if e_int[j] == 1:
                factors.append(name)
            else:
                factors.append(f"{name}^{int(e_int[j])}")
        if e_int.shape[0] > 3 and e_int[3] != 0:
            if e_int[3] == 1:
                factors.append("t")
            else:
                factors.append(f"t^{int(e_int[3])}")
        mono = "*".join(factors)
        term_strs.append((c, mono))

    if not term_strs:
        return "0"

    parts: List[str] = []
    for i, (c, mono) in enumerate(term_strs):
        if mono:
            tok = f"{abs(c):.14g}*{mono}"
        else:
            tok = f"{abs(c):.14g}"
        if i == 0:
            parts.append(f"-{tok}" if c < 0 else tok)
        else:
            parts.append(f" - {tok}" if c < 0 else f" + {tok}")
    return "".join(parts)


def wsindy_model_to_dimension_lines(
    model: Any,
    *,
    threshold: float = 1e-9,
) -> List[str]:
    """Three lines ``dimension_d: ...`` compatible with ``final_expressions.txt`` style."""
    coef = np.asarray(model.coef)
    tags = np.asarray(model.tags)
    lines = []
    for d in range(coef.shape[1]):
        rhs = format_wsindy_dimension_rhs(coef[:, d], tags, threshold=threshold)
        lines.append(f"dimension_{d + 1}: {rhs}")
    return lines


def enforce_triad_one_interaction_structure(
    model: Any,
    *,
    keep_constant: bool = False,
    keep_all_linear: bool = True,
) -> Any:
    """
    Enforce a triad-structured sparse RHS on an already fitted WSINDy model.

    Keeps, per dimension:
      - dim1: x1, x2, x3, and x2*x3
      - dim2: x1, x2, x3, and x1*x3
      - dim3: x1, x2, x3, and x1*x2
    Optionally keeps constant terms.
    """
    coef = np.asarray(model.coef).copy()
    tags = np.asarray(model.tags)
    if coef.ndim != 2 or tags.ndim != 2 or tags.shape[1] < 3:
        return model

    tags_int = np.rint(np.real(tags)).astype(int)
    allowed_per_dim = [{(0, 1, 1)}, {(1, 0, 1)}, {(1, 1, 0)}]
    if keep_all_linear:
        linear_terms = {(1, 0, 0), (0, 1, 0), (0, 0, 1)}
        for s in allowed_per_dim:
            s.update(linear_terms)
    if keep_constant:
        for s in allowed_per_dim:
            s.add((0, 0, 0))

    has_time_dim = tags_int.shape[1] > 3
    for d in range(min(3, coef.shape[1])):
        keep_mask = np.zeros(tags_int.shape[0], dtype=bool)
        for k in range(tags_int.shape[0]):
            x_exp = tuple(tags_int[k, :3].tolist())
            t_exp = int(tags_int[k, 3]) if has_time_dim else 0
            # Keep explicit forcing basis terms t..t^k (pure time only, no x*t cross terms).
            if has_time_dim and t_exp > 0 and x_exp == (0, 0, 0):
                keep_mask[k] = True
                continue
            # Keep selected x-library terms (requires t exponent 0 when time basis exists).
            if t_exp == 0 and x_exp in allowed_per_dim[d]:
                keep_mask[k] = True
        coef[~keep_mask, d] = 0.0

    model.coef = coef
    return model


def save_wsindy_expression_file(
    model: Any,
    out_path: Path,
    *,
    threshold: float = 1e-9,
    note: str = "",
    title: str = "Final Expressions (WSINDy, ensemble-mean trajectory, physical x1, x2, x3):",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = wsindy_model_to_dimension_lines(model, threshold=threshold)
    text = [
        title,
        "==================================================",
    ]
    if note:
        text.append(note)
        text.append("==================================================")
    text.extend(lines)
    text.append("")
    text.append("Export completed successfully!")
    out_path.write_text("\n".join(text), encoding="utf-8")


def save_wsindy_per_dimension_files(
    model: Any,
    out_dir: Path | str,
    *,
    threshold: float = 1e-9,
    meta_note: str = "",
    name_suffix: str = "",
) -> List[Path]:
    """
    Write ``wsindy_expression_1.txt`` … ``wsindy_expression_3.txt`` (one RHS per file).

    Uses one joint WSINDy fit (same ``model`` as the full 3D vector field); each file holds
    the discovered polynomial for ``dx_i/dt`` in x1,x2,x3 coordinates.
    ``name_suffix`` is inserted before ``.txt``, e.g. ``_integrated`` → ``wsindy_expression_1_integrated.txt``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coef = np.asarray(model.coef)
    tags = np.asarray(model.tags)
    paths: List[Path] = []
    for d in range(coef.shape[1]):
        rhs = format_wsindy_dimension_rhs(coef[:, d], tags, threshold=threshold)
        p = out_dir / f"wsindy_expression_{d + 1}{name_suffix}.txt"
        body = [
            f"WSINDy RHS for dimension {d + 1} (ensemble-mean data, physical x1, x2, x3):",
            "==================================================",
        ]
        if meta_note:
            body.append(meta_note)
            body.append("==================================================")
        body.append(f"dimension_{d + 1}: {rhs}")
        body.append("")
        body.append("Export completed successfully!")
        p.write_text("\n".join(body), encoding="utf-8")
        paths.append(p)
    return paths


def export_wsindy_per_dimension_from_npz(
    npz_path: Path | str,
    out_dir: Path | str,
    *,
    polys: Optional[np.ndarray] = None,
    L: int = 30,
    overlap: float = 0.7,
    threshold: float = 1e-9,
    ws_ld: float = 1e-3,
    ws_gamma: float = 0.0,
    ws_scale_theta: int = 0,
    ws_use_gls: float = 1e-12,
    time_poly_max: int = 0,
    triad_interaction_only: bool = False,
    keep_constant_terms: bool = False,
) -> Tuple[Any, List[Path]]:
    """Fit WSINDy on raw ensemble mean; save ``wsindy_expression_{1,2,3}.txt`` into ``out_dir``."""
    t, x_raw, meta = mean_trajectory_from_npz(npz_path)
    model = fit_wsindy(
        t,
        x_raw,
        polys=polys,
        L=L,
        overlap=overlap,
        ws_ld=ws_ld,
        ws_gamma=ws_gamma,
        ws_scale_theta=ws_scale_theta,
        ws_use_gls=ws_use_gls,
        time_poly_max=time_poly_max,
    )
    if triad_interaction_only:
        model = enforce_triad_one_interaction_structure(
            model, keep_constant=keep_constant_terms, keep_all_linear=True
        )
    note = (
        f"npz: {meta['npz_path']}\n"
        f"trajectories: valid {meta['n_valid']} / total {meta['n_total']} "
        f"(dropped {meta['n_dropped']})"
    )
    paths = save_wsindy_per_dimension_files(
        model, out_dir, threshold=threshold, meta_note=note
    )
    return model, paths


def export_wsindy_expressions_from_npz(
    npz_path: Path | str,
    out_path: Path | str,
    *,
    polys: Optional[np.ndarray] = None,
    L: int = 30,
    overlap: float = 0.7,
    threshold: float = 1e-9,
    note: str = "",
    ws_ld: float = 1e-3,
    ws_gamma: float = 0.0,
    ws_scale_theta: int = 0,
    ws_use_gls: float = 1e-12,
    time_poly_max: int = 0,
    triad_interaction_only: bool = False,
    keep_constant_terms: bool = False,
) -> Any:
    """
    Fit WSINDy on the **raw** (physical) ensemble mean so printed coefficients match x1,x2,x3
    coordinates, comparable to ``final_expressions.txt`` from FEX.
    """
    t, x_raw, meta = mean_trajectory_from_npz(npz_path)
    model = fit_wsindy(
        t,
        x_raw,
        polys=polys,
        L=L,
        overlap=overlap,
        ws_ld=ws_ld,
        ws_gamma=ws_gamma,
        ws_scale_theta=ws_scale_theta,
        ws_use_gls=ws_use_gls,
        time_poly_max=time_poly_max,
    )
    if triad_interaction_only:
        model = enforce_triad_one_interaction_structure(
            model, keep_constant=keep_constant_terms, keep_all_linear=True
        )
    note_extra = (
        f"npz: {meta['npz_path']}\n"
        f"trajectories: valid {meta['n_valid']} / total {meta['n_total']} "
        f"(dropped {meta['n_dropped']})"
    )
    full_note = note_extra if not note else f"{note_extra}\n{note}"
    save_wsindy_expression_file(model, out_path, threshold=threshold, note=full_note)
    return model


def export_integrated_wsindy_finetune_from_npz(
    npz_path: Path | str,
    out_dir: Path | str,
    *,
    polys: Optional[np.ndarray] = None,
    r_whm: int = 30,
    s: int = 16,
    K: int = 200,
    p: int = 2,
    tau_p: int = 16,
    threshold: float = 1e-9,
    ws_ld: float = 1e-3,
    ws_gamma: float = 0.0,
    ws_scale_theta: int = 0,
    ws_use_gls: float = 1e-12,
    time_poly_max: int = 0,
    triad_interaction_only: bool = False,
    keep_constant_terms: bool = False,
) -> Tuple[Any, Path, List[Path]]:
    """
    Integrated WSINDy-only refinement: adaptive weak form on the **raw** ensemble mean,
    joint (x1,x2,x3). Writes ``finite_expression_wsindy_integrated.txt`` and
    ``wsindy_expression_{1,2,3}_integrated.txt``.
    """
    t, x_raw, meta = mean_trajectory_from_npz(npz_path)
    model = fit_wsindy_adaptive(
        t,
        x_raw,
        polys=polys,
        r_whm=r_whm,
        s=s,
        K=K,
        p=p,
        tau_p=tau_p,
        ws_ld=ws_ld,
        ws_gamma=ws_gamma,
        ws_scale_theta=ws_scale_theta,
        ws_use_gls=ws_use_gls,
        time_poly_max=time_poly_max,
    )
    if triad_interaction_only:
        model = enforce_triad_one_interaction_structure(
            model, keep_constant=keep_constant_terms, keep_all_linear=True
        )
    note = (
        f"npz: {meta['npz_path']}\n"
        f"trajectories: valid {meta['n_valid']} / total {meta['n_total']} "
        f"(dropped {meta['n_dropped']})\n"
        f"method: getWsindyAdaptive (joint 3D)"
    )
    out_dir = Path(out_dir)
    combined = out_dir / "finite_expression_wsindy_integrated.txt"
    save_wsindy_expression_file(
        model,
        combined,
        threshold=threshold,
        note=note,
        title="Final Expressions (WSINDy adaptive integrated, ensemble-mean, physical x1, x2, x3):",
    )
    paths = save_wsindy_per_dimension_files(
        model,
        out_dir,
        threshold=threshold,
        meta_note=note,
        name_suffix="_integrated",
    )
    return model, combined, paths


def fit_sindy_pysindy(
    t: np.ndarray,
    x: np.ndarray,
    *,
    poly_degree: int = 2,
    stlsq_max_iter: int = 20,
):
    import pysindy as ps

    poly_library = ps.PolynomialLibrary(
        include_interaction=True, degree=poly_degree
    )
    optimizer = ps.STLSQ(copy_X=False, max_iter=stlsq_max_iter, ridge_kw=None)
    model = ps.SINDy(feature_library=poly_library, optimizer=optimizer)
    model.fit(x, t=t)
    return model


def simulate_sindy(
    model,
    x0: np.ndarray,
    t: np.ndarray,
    *,
    integrator_kws: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    if integrator_kws is None:
        dt = float(np.min(np.diff(t[np.isfinite(t)])))
        integrator_kws = {"method": "LSODA", "max_step": dt, "rtol": 1e-6, "atol": 1e-8}
    return model.simulate(x0, t=t, integrator_kws=integrator_kws)


def run_case(
    params_name: str,
    *,
    noise_level: float = 1.0,
    triad_root: Optional[Path] = None,
    standardize: bool = True,
    polys: Optional[np.ndarray] = None,
    L: int = 30,
    overlap: float = 0.7,
    run_sindy: bool = True,
    poly_degree: int = 2,
    ws_ld: float = 1e-3,
    ws_gamma: float = 0.0,
    ws_scale_theta: int = 0,
    ws_use_gls: float = 1e-12,
    time_poly_max: int = 0,
) -> Dict[str, Any]:
    """Fit WSINDy (and optionally SINDy) on the ensemble mean; return metrics and models."""
    if params_name not in TRIAD_CASES:
        raise ValueError(f"params_name must be one of {TRIAD_CASES}")

    path = default_npz_path(params_name, noise_level, triad_root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing simulation bundle: {path}")

    t, x_raw, meta = mean_trajectory_from_npz(path)
    x = x_raw
    if standardize:
        x, mu, sig = _standardize(x_raw)

    x0 = x[0].copy()

    out: Dict[str, Any] = {"meta": meta, "params_name": params_name, "standardize": standardize}

    ws_model = fit_wsindy(
        t,
        x,
        polys=polys,
        L=L,
        overlap=overlap,
        ws_ld=ws_ld,
        ws_gamma=ws_gamma,
        ws_scale_theta=ws_scale_theta,
        ws_use_gls=ws_use_gls,
        time_poly_max=time_poly_max,
    )
    ws_sim = simulate_wsindy(ws_model, x0, t)
    out["wsindy"] = {
        "model": ws_model,
        "rmse_mean_traj": _rmse(ws_sim, x),
    }

    if run_sindy:
        try:
            ps = __import__("pysindy", fromlist=["*"])
            _ = ps
        except ImportError:
            out["sindy"] = None
            out["sindy_note"] = "pysindy not installed; skipped (pip install pysindy)."
        else:
            sd_model = fit_sindy_pysindy(t, x, poly_degree=poly_degree)
            try:
                sd_sim = simulate_sindy(sd_model, x0, t)
                sd_rmse = _rmse(sd_sim, x)
            except (ValueError, RuntimeError, ArithmeticError) as e:
                sd_sim = None
                sd_rmse = float("nan")
                out["sindy_error"] = str(e)
            out["sindy"] = {
                "model": sd_model,
                "rmse_mean_traj": sd_rmse,
            }

    return out


def run_interactive_menu(args: argparse.Namespace) -> None:
    """Loop like 1stage style: choices 1–4 (no quit)."""
    print(
        "[INFO] Interactive menu — use --batch for non-interactive CLI. "
        "Flags --params_name, --noise_level, … apply below."
    )
    while True:
        print("\n" + "=" * 60)
        print("WSINDY: OPTIONS")
        print("=" * 60)
        print("1. WSINDy vs PySINDy comparison (RMSE on ensemble mean)")
        print("2. Integrated WSINDy finetune: adaptive weak form, joint (x1,x2,x3) [getWsindyAdaptive]")
        print("3. Uniform WSINDy export: finite_expression_wsindy.txt + wsindy_expression_1/2/3.txt")
        print(
            "4. Plot diagnostics: wsindy_fit_comparison.png + wsindy_prediction_extended.png "
            "(fit vs data, det + stochastic rollout; flags: --prediction_t_max, --stoch_paths, …)"
        )
        print("=" * 60)
        print(
            f"Case: {args.params_name} | noise_{args.noise_level} | "
            f"poly_max={args.poly_max}, L={args.L}, overlap={args.overlap}"
        )
        print("=" * 60)

        while True:
            choice = input("\nChoose option (1, 2, 3, or 4):").strip()
            if choice in ("1", "2", "3", "4"):
                break
            print("Please enter '1', '2', '3', or '4'.")

        triad_root = Path(args.triad_root) if args.triad_root else None
        polys = np.arange(0, args.poly_max + 1)
        npz = default_npz_path(args.params_name, args.noise_level, triad_root)
        if choice == "1":
            a = argparse.Namespace(**vars(args))
            a.save_wsindy_expressions = False
            a.save_wsindy_per_dimension = False
            a.all = False
            run_all_cli(a)
        elif choice == "2":
            if not npz.is_file():
                print(f"[ERROR] Missing simulation bundle: {npz}")
                continue
            _, combined, paths = export_integrated_wsindy_finetune_from_npz(
                npz,
                npz.parent,
                polys=polys,
                r_whm=args.r_whm,
                s=args.wsindy_adaptive_s,
                K=args.K_adaptive,
                p=args.wsindy_adaptive_p,
                tau_p=args.tau_p,
                threshold=args.expression_threshold,
                ws_ld=args.ws_ld,
                ws_gamma=args.ws_gamma,
                ws_scale_theta=args.ws_scale_theta,
                ws_use_gls=args.ws_use_gls,
                time_poly_max=args.time_poly_max,
                triad_interaction_only=args.triad_interaction_only,
                keep_constant_terms=args.keep_constant_terms,
            )
            print(f"[INFO] Wrote {combined}")
            for p in paths:
                print(f"[INFO] Wrote {p}")
        elif choice == "3":
            if not npz.is_file():
                print(f"[ERROR] Missing simulation bundle: {npz}")
                continue
            _, paths = export_wsindy_per_dimension_from_npz(
                npz,
                npz.parent,
                polys=polys,
                L=args.L,
                overlap=args.overlap,
                threshold=args.expression_threshold,
                ws_ld=args.ws_ld,
                ws_gamma=args.ws_gamma,
                ws_scale_theta=args.ws_scale_theta,
                ws_use_gls=args.ws_use_gls,
                time_poly_max=args.time_poly_max,
                triad_interaction_only=args.triad_interaction_only,
                keep_constant_terms=args.keep_constant_terms,
            )
            for p in paths:
                print(f"[INFO] Wrote {p}")
            out_combined = Path(args.wsindy_expression_out) if args.wsindy_expression_out else (
                npz.parent / "finite_expression_wsindy.txt"
            )
            export_wsindy_expressions_from_npz(
                npz,
                out_combined,
                polys=polys,
                L=args.L,
                overlap=args.overlap,
                threshold=args.expression_threshold,
                ws_ld=args.ws_ld,
                ws_gamma=args.ws_gamma,
                ws_scale_theta=args.ws_scale_theta,
                ws_use_gls=args.ws_use_gls,
                time_poly_max=args.time_poly_max,
                triad_interaction_only=args.triad_interaction_only,
                keep_constant_terms=args.keep_constant_terms,
            )
            print(f"[INFO] Wrote {out_combined}")
        elif choice == "4":
            if not npz.is_file():
                print(f"[ERROR] Missing simulation bundle: {npz}")
                continue
            run_wsindy_prediction_plots(args.params_name, args)


def print_summary(result: Dict[str, Any]) -> None:
    m = result["meta"]
    print(f"\n=== {result['params_name']} ===")
    print(f"npz: {m['npz_path']}")
    print(f"trajectories: valid {m['n_valid']} / total {m['n_total']} (dropped {m['n_dropped']})")
    print(f"standardize: {result['standardize']}")
    w = result["wsindy"]
    print(f"WSINDy RMSE vs mean trajectory: {w['rmse_mean_traj']:.6g}")
    if result.get("sindy") is not None:
        print(f"SINDy RMSE vs mean trajectory: {result['sindy']['rmse_mean_traj']:.6g}")
        if result.get("sindy_error"):
            print(f"SINDy simulate note: {result['sindy_error'][:200]}")
    elif result.get("sindy_note"):
        print(result["sindy_note"])


def run_all_cli(args: argparse.Namespace) -> None:
    cases: List[str]
    if args.all:
        cases = list(TRIAD_CASES)
    else:
        cases = [args.params_name]

    triad_root = Path(args.triad_root) if args.triad_root else None
    polys = np.arange(0, args.poly_max + 1) if args.poly_max is not None else None

    for name in cases:
        if getattr(args, "save_wsindy_expressions", False):
            npz = default_npz_path(name, args.noise_level, triad_root)
            out = Path(args.wsindy_expression_out) if getattr(
                args, "wsindy_expression_out", None
            ) else (npz.parent / "finite_expression_wsindy.txt")
            export_wsindy_expressions_from_npz(
                npz,
                out,
                polys=polys,
                L=args.L,
                overlap=args.overlap,
                threshold=args.expression_threshold,
                ws_ld=args.ws_ld,
                ws_gamma=args.ws_gamma,
                ws_scale_theta=args.ws_scale_theta,
                ws_use_gls=args.ws_use_gls,
                time_poly_max=args.time_poly_max,
                triad_interaction_only=args.triad_interaction_only,
                keep_constant_terms=args.keep_constant_terms,
            )
            print(f"[INFO] Wrote WSINDy expressions to {out}")

        if getattr(args, "save_wsindy_per_dimension", False):
            npz = default_npz_path(name, args.noise_level, triad_root)
            _, paths = export_wsindy_per_dimension_from_npz(
                npz,
                npz.parent,
                polys=polys,
                L=args.L,
                overlap=args.overlap,
                threshold=args.expression_threshold,
                ws_ld=args.ws_ld,
                ws_gamma=args.ws_gamma,
                ws_scale_theta=args.ws_scale_theta,
                ws_use_gls=args.ws_use_gls,
                time_poly_max=args.time_poly_max,
                triad_interaction_only=args.triad_interaction_only,
                keep_constant_terms=args.keep_constant_terms,
            )
            for p in paths:
                print(f"[INFO] Wrote WSINDy per-dimension file {p}")

        r = run_case(
            name,
            noise_level=args.noise_level,
            triad_root=triad_root,
            standardize=not args.no_standardize,
            polys=polys,
            L=args.L,
            overlap=args.overlap,
            run_sindy=not args.wsindy_only,
            poly_degree=args.sindy_degree,
            ws_ld=args.ws_ld,
            ws_gamma=args.ws_gamma,
            ws_scale_theta=args.ws_scale_theta,
            ws_use_gls=args.ws_use_gls,
            time_poly_max=args.time_poly_max,
        )
        print_summary(r)

        if getattr(args, "plot_wsindy", False):
            run_wsindy_prediction_plots(name, args)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="WSINDy / SINDy comparison on MC triad npz bundles."
    )
    p.add_argument(
        "--params_name",
        type=str,
        choices=list(TRIAD_CASES),
        default=_cfg_args.params_name,
        help="Single triad case (ignored if --all); default from config.create_main_parser().",
    )
    p.add_argument("--all", action="store_true", help="Run all supported triad cases.")
    p.add_argument(
        "--noise_level",
        type=float,
        default=_cfg_args.NOISE_LEVEL,
        help="Default matches config --NOISE_LEVEL.",
    )
    p.add_argument(
        "--triad_root",
        type=str,
        default=None,
        help="Override MC_triad directory (default: src/Example/MC_triad).",
    )
    p.add_argument(
        "--no_standardize",
        action="store_true",
        help="Use raw ensemble mean (default: per-dimension z-score).",
    )
    p.add_argument(
        "--poly_max",
        type=int,
        default=2,
        help="Max monomial pool index passed to WSINDy as np.arange(0, poly_max+1).",
    )
    p.add_argument("--L", type=int, default=30, help="WSINDy uniform test function support length.")
    p.add_argument("--overlap", type=float, default=0.7, help="WSINDy uniform overlap.")
    p.add_argument(
        "--ws_ld",
        type=float,
        default=1e-3,
        help="WSINDy sequential threshold (sparsifyDynamics ld).",
    )
    p.add_argument(
        "--ws_gamma",
        type=float,
        default=0.0,
        help="WSINDy Tikhonov regularization gamma (0 means none).",
    )
    p.add_argument(
        "--ws_scale_theta",
        type=int,
        default=0,
        help="WSINDy theta-column normalization norm order (0 disables scaling).",
    )
    p.add_argument(
        "--ws_use_gls",
        type=float,
        default=1e-12,
        help="WSINDy GLS stabilization floor (useGLS in WSINDy).",
    )
    p.add_argument(
        "--time_poly_max",
        type=int,
        default=0,
        help="Add explicit time basis terms t..t^k to WSINDy library (k=time_poly_max).",
    )
    p.add_argument(
        "--triad_interaction_only",
        action="store_true",
        help="Keep only triad terms per dimension: x1,x2,x3 + one cross interaction.",
    )
    p.add_argument(
        "--keep_constant_terms",
        action="store_true",
        help="When --triad_interaction_only is used, also keep constant terms.",
    )
    p.add_argument(
        "--wsindy_only",
        action="store_true",
        help="Skip PySINDy (no pip dependency).",
    )
    p.add_argument("--sindy_degree", type=int, default=2, help="Polynomial degree for PySINDy.")
    p.add_argument(
        "--save_wsindy_expressions",
        action="store_true",
        help="After fitting on raw ensemble mean, write finite_expression_wsindy.txt next to the npz.",
    )
    p.add_argument(
        "--wsindy_expression_out",
        type=str,
        default=None,
        help="Override output path (default: <noise_dir>/finite_expression_wsindy.txt).",
    )
    p.add_argument(
        "--expression_threshold",
        type=float,
        default=1e-9,
        help="Drop |coefficient| below this when printing.",
    )
    p.add_argument(
        "--save_wsindy_per_dimension",
        action="store_true",
        help="Write wsindy_expression_1.txt, _2, _3 next to the npz (joint WSINDy fit).",
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Non-interactive mode: run exports/comparison from flags (default is interactive menu).",
    )
    p.add_argument(
        "--r_whm",
        type=int,
        default=30,
        help="getWsindyAdaptive r_whm (menu choice 2 / integrated WSINDy).",
    )
    p.add_argument(
        "--K_adaptive",
        type=int,
        default=200,
        help="getWsindyAdaptive K (number of adaptive grid points).",
    )
    p.add_argument(
        "--wsindy_adaptive_s",
        type=int,
        default=16,
        help="getWsindyAdaptive inner index gap parameter s.",
    )
    p.add_argument(
        "--wsindy_adaptive_p",
        type=int,
        default=2,
        help="getWsindyAdaptive basis degree p.",
    )
    p.add_argument(
        "--tau_p",
        type=int,
        default=16,
        help="getWsindyAdaptive tau_p (test-function width control).",
    )
    p.add_argument(
        "--plot_wsindy",
        action="store_true",
        help="Write wsindy_fit_comparison.png and wsindy_prediction_extended.png next to the npz.",
    )
    p.add_argument(
        "--prediction_t_max",
        type=float,
        default=20.0,
        help="Extend WSINDy IVP and stochastic rollout to this time (uses same dt as npz).",
    )
    p.add_argument(
        "--stoch_paths",
        type=int,
        default=48,
        help="Number of Heun RK2 (additive noise) sample paths for plot_wsindy.",
    )
    p.add_argument(
        "--plot_seed",
        type=int,
        default=0,
        help="RNG seed for stochastic paths in plot_wsindy.",
    )
    return p


def run_equipart_wsindy_fit_plots(
    noise_level: float = 1.0,
    ws_ld: float = 0.03,
    ws_gamma: float = 0.001,
    triad_interaction_only: bool = True,
    prediction_t_max: float = 20.0,
    stoch_paths: int = 32,
    plot_seed: int = 0,
) -> None:
    """Equipart-only: same PNGs as ``--plot_wsindy`` (convenience for scripts)."""
    argv = [
        "--batch",
        "--params_name",
        "equipart",
        "--noise_level",
        str(noise_level),
        "--wsindy_only",
        "--plot_wsindy",
        "--no_standardize",
        "--ws_ld",
        str(ws_ld),
        "--ws_gamma",
        str(ws_gamma),
    ]
    if triad_interaction_only:
        argv.append("--triad_interaction_only")
    argv.extend(
        [
            "--prediction_t_max",
            str(prediction_t_max),
            "--stoch_paths",
            str(stoch_paths),
            "--plot_seed",
            str(plot_seed),
        ]
    )
    args = build_arg_parser().parse_args(argv)
    run_wsindy_prediction_plots("equipart", args)


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.batch:
        run_all_cli(args)
    else:
        run_interactive_menu(args)


if __name__ == "__main__":
    main()
