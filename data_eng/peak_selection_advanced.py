#!/usr/bin/env python3
"""
Peak / pulse extraction + feature building + parameter scan (ready for GMVAE/PSD).

Designed for large "trigger-window" records containing multiple pulses.
Input: NPZ with key "wfs" shaped (n_records, n_samples).

Core ideas
- Robust baseline subtraction per record (tail percentile or chunked drift removal).
- Peak finding with scipy.signal.find_peaks using *noise-adaptive* thresholds.
- Optional alignment to constant-fraction crossing (rising edge).
- Pulse window extraction (fixed pre/post around peak or around rising-edge).
- Quality cuts + pile-up flagging.
- Feature table output (Parquet/CSV).
- Grid-scan of parameters using an *unsupervised proxy*: how well features separate
  "early-in-window" pulses (PNS dominated) from "late-in-window" pulses (background dominated).

This is meant to be a clean starting point. Tune defaults in Config below.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks, peak_widths


# -----------------------------
# Configuration
# -----------------------------

@dataclass(frozen=True)
class BaselineConfig:
    # Use tail region to estimate baseline and noise:
    tail_fraction: float = 0.30        # last 30% of samples
    tail_percentile: float = 10.0      # robust baseline = percentile(tail)
    # Optional drift check with chunked percentile baselines:
    enable_drift_check: bool = True
    drift_chunk: int = 2000            # samples per chunk
    drift_percentile: float = 10.0
    drift_ptp_threshold: float = 50.0  # ADC counts peak-to-peak across chunks


@dataclass(frozen=True)
class PeakFindConfig:
    # find_peaks parameters, in units of baseline sigma where applicable
    min_height_sigma: float = 6.0
    min_prom_sigma: float = 4.0
    min_distance: int = 30            # samples between peaks
    min_width: int = 2               # samples; acts as deglitch
    max_peaks_per_record: int = 200  # hard cap to avoid runaway noise triggers


@dataclass(frozen=True)
class AlignConfig:
    mode: str = "cfd"               # "peak" or "cfd" (constant-fraction on rising edge)
    cfd_fraction: float = 0.2
    cfd_search_back: int = 600       # samples before peak to search for crossing


@dataclass(frozen=True)
class WindowConfig:
    # pre: int = 40                    # samples before anchor
    # post: int = 200                  # samples after anchor
    ## version 1
    # pre: int = 10                    # v1
    # post: int = 128                  # v1
    ## version 2
    # pre: int = 20                  # v2
    # post: int = 320                # v2
    ## version 3
    # pre: int = 20                 # v3
    # post: int = 500                # v3
    ## version 4
    # pre: int = 30                 # v4
    # post: int = 530               # v4
    ## version 5
    pre: int = 20                 # v5
    post: int = 250               # v5
    anchor: str = "cfd"             # "peak" or "cfd" (must match AlignConfig.mode if using cfd)


@dataclass(frozen=True)
class CutsConfig:
    # Amplitude cut:
    min_amp_sigma: float = 8.0
    # Edge safety: reject if window would clip record
    require_full_window: bool = True
    # Saturation cut (optional): reject if any sample is at/near ADC max
    enable_saturation_cut: bool = False
    adc_max: float = 16383.0
    saturation_margin: float = 1.0
    # Pile-up flagging: second peak above a fraction of primary within the window
    pileup_secondary_frac: float = 0.20
    pileup_secondary_min_sigma: float = 5.0
    # Shape sanity: rise-time bounds in samples (computed 10%->90%)
    enable_risetime_cut: bool = True
    rise_min: int = 1
    rise_max: int = 400


@dataclass(frozen=True)
class ScanProxyConfig:
    # Define "early" and "late" regions in the record (fractions of total samples)
    early_fraction: float = 0.20
    late_fraction: float = 0.30      # last 30% is "late/background"
    # Minimum statistics to accept a scan point:
    min_pulses_total: int = 500
    min_pulses_per_class: int = 100


@dataclass(frozen=True)
class Config:
    baseline: BaselineConfig = BaselineConfig()
    peak: PeakFindConfig = PeakFindConfig()
    align: AlignConfig = AlignConfig()
    window: WindowConfig = WindowConfig()
    cuts: CutsConfig = CutsConfig()
    scan_proxy: ScanProxyConfig = ScanProxyConfig()

    # If you know dt (ns/sample), set it. Otherwise leave None and features remain in samples.
    dt_ns: Optional[float] = None


# -----------------------------
# Helpers
# -----------------------------

def robust_tail_baseline_sigma(x: np.ndarray, tail_fraction: float, tail_percentile: float) -> Tuple[float, float]:
    """Return (baseline, sigma) from the tail region."""
    n = len(x)
    start = int((1.0 - tail_fraction) * n)
    tail = x[start:]
    baseline = float(np.percentile(tail, tail_percentile))
    # robust sigma: MAD scaled to sigma for Gaussian
    mad = float(np.median(np.abs(tail - np.median(tail))))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(tail))
    sigma = max(sigma, 1e-6)
    return baseline, sigma


def estimate_baseline_chunks(record: np.ndarray, chunk: int, q: float) -> np.ndarray:
    n = len(record)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    return np.asarray([np.percentile(record[i:i + chunk], q) for i in range(0, n, chunk)], dtype=np.float64)


def interpolate_baseline_chunks(bchunks: np.ndarray, n_samples: int, chunk: int) -> np.ndarray:
    if len(bchunks) == 0:
        return np.zeros(n_samples, dtype=np.float64)
    centers = np.arange(len(bchunks), dtype=np.float64) * chunk + (chunk - 1) / 2
    centers = np.clip(centers, 0, n_samples - 1)
    return np.interp(np.arange(n_samples, dtype=np.float64), centers, bchunks)


def constant_fraction_crossing_index(y: np.ndarray, peak_idx: int, frac: float, search_back: int) -> Optional[int]:
    """First rising-edge crossing of frac*peak height when scanning backward from peak."""
    peak_val = float(y[peak_idx])
    if peak_val <= 0:
        return None
    thr = frac * peak_val
    start = max(0, peak_idx - search_back)
    seg = y[start:peak_idx + 1]
    above = seg >= thr
    if not np.any(above):
        return None
    k = int(np.argmax(above))
    return int(start + k)


def safe_slice(x: np.ndarray, lo: int, hi: int) -> Optional[np.ndarray]:
    if lo < 0 or hi > len(x) or lo >= hi:
        return None
    return x[lo:hi]


def area_trapz(y: np.ndarray) -> float:
    # sample spacing cancels out in ratios; keep in samples unless dt_ns used elsewhere
    return float(np.trapezoid(y))


def compute_rise_time(y: np.ndarray, peak_idx: int, frac_lo: float = 0.1, frac_hi: float = 0.9, search_back: int = 800) -> Optional[int]:
    """Rise time in samples between frac_lo and frac_hi crossings."""
    peak_val = float(y[peak_idx])
    if peak_val <= 0:
        return None
    lo_thr = frac_lo * peak_val
    hi_thr = frac_hi * peak_val
    start = max(0, peak_idx - search_back)
    seg = y[start:peak_idx + 1]

    above_lo = np.where(seg >= lo_thr)[0]
    above_hi = np.where(seg >= hi_thr)[0]
    if len(above_lo) == 0 or len(above_hi) == 0:
        return None
    t_lo = int(start + above_lo[0])
    t_hi = int(start + above_hi[0])
    rt = t_hi - t_lo
    return rt if rt >= 0 else None


# -----------------------------
# Core processing
# -----------------------------

def preprocess_record(x: np.ndarray, cfg: Config) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Baseline subtract. If enabled, do chunked baseline drift subtraction; otherwise constant.
    Returns baseline-sub record + diagnostics.
    """
    base_tail, sigma_tail = robust_tail_baseline_sigma(
        x,
        tail_fraction=cfg.baseline.tail_fraction,
        tail_percentile=cfg.baseline.tail_percentile,
    )

    drift_flag = False
    drift_ptp = 0.0
    baseline_per_sample = None

    if cfg.baseline.enable_drift_check:
        bchunks = estimate_baseline_chunks(x, chunk=cfg.baseline.drift_chunk, q=cfg.baseline.drift_percentile)
        drift_ptp = float(np.ptp(bchunks)) if len(bchunks) else 0.0
        drift_flag = drift_ptp > cfg.baseline.drift_ptp_threshold
        baseline_per_sample = interpolate_baseline_chunks(bchunks, n_samples=len(x), chunk=cfg.baseline.drift_chunk)
        y = x.astype(np.float64) - baseline_per_sample
    else:
        y = x.astype(np.float64) - base_tail

    diag = dict(
        baseline_tail=base_tail,
        sigma_tail=sigma_tail,
        drift_ptp=drift_ptp,
        drift_flag=float(drift_flag),
    )
    return y, diag


def detect_peaks(y: np.ndarray, sigma: float, cfg: PeakFindConfig) -> np.ndarray:
    """Return peak indices (sorted)."""
    height = cfg.min_height_sigma * sigma
    prom = cfg.min_prom_sigma * sigma
    peaks, props = find_peaks(
        y,
        height=height,
        prominence=prom,
        distance=cfg.min_distance,
        width=cfg.min_width,
    )
    if len(peaks) > cfg.max_peaks_per_record:
        # keep the strongest by prominence
        order = np.argsort(props["prominences"])[::-1][: cfg.max_peaks_per_record]
        peaks = np.sort(peaks[order])
    return peaks.astype(int)


def extract_pulse(
    x_raw: np.ndarray,
    y: np.ndarray,
    peak_idx: int,
    sigma: float,
    rec_idx: int,
    cfg: Config,
) -> Optional[Dict[str, float]]:
    """Extract one pulse around anchor. Returns feature dict or None if rejected."""
    # choose anchor index
    anchor_idx = peak_idx
    if cfg.align.mode.lower() == "cfd" or cfg.window.anchor.lower() == "cfd":
        cfd_idx = constant_fraction_crossing_index(
            y, peak_idx=peak_idx, frac=cfg.align.cfd_fraction, search_back=cfg.align.cfd_search_back
        )
        if cfd_idx is None:
            return None
        anchor_idx = cfd_idx

    lo = anchor_idx - cfg.window.pre
    hi = anchor_idx + cfg.window.post
    seg = safe_slice(y, lo, hi)
    # Saturation must be evaluated on the *raw ADC* (before any baseline subtraction).
    seg_raw = safe_slice(x_raw, lo, hi)
    if seg is None:
        return None if cfg.cuts.require_full_window else None
    if seg_raw is None:
        return None if cfg.cuts.require_full_window else None

    amp = float(y[peak_idx])
    if amp < cfg.cuts.min_amp_sigma * sigma:
        return None

    # Saturation flag (high-rail only): mark if any *raw* sample in the extracted window hits ADCmax (within margin).
    # Using raw avoids baseline subtraction hiding clipping.
    sat_level = cfg.cuts.adc_max - cfg.cuts.saturation_margin
    is_saturated = float(np.any(seg_raw >= sat_level))
    # Optional hard rejection
    if cfg.cuts.enable_saturation_cut and is_saturated > 0.5:
        return None

    # Secondary peak / pile-up inside the extracted segment:
    seg_sigma = max(float(np.std(seg[-max(10, len(seg)//4):])), sigma)  # conservative
    sec_height = max(cfg.cuts.pileup_secondary_min_sigma * seg_sigma, cfg.cuts.pileup_secondary_frac * amp)
    sec_peaks, sec_props = find_peaks(seg, height=sec_height, distance=cfg.peak.min_distance, width=cfg.peak.min_width)
    # Count peaks in segment that are not the main peak (may be outside segment center, so we compare by absolute index)
    # Map seg indices to record indices:
    sec_abs = sec_peaks + lo
    sec_abs = sec_abs[sec_abs != peak_idx]
    is_pileup = float(len(sec_abs) > 0)

    # Rise time cut
    rise = compute_rise_time(y, peak_idx=peak_idx)
    if cfg.cuts.enable_risetime_cut:
        if rise is None or rise < cfg.cuts.rise_min or rise > cfg.cuts.rise_max:
            return None

    # Width estimate at half max (in samples)
    try:
        wres = peak_widths(y, [peak_idx], rel_height=0.5)
        fwhm = float(wres[0][0])
    except Exception:
        fwhm = np.nan

    # Charge features on the extracted segment
    q_total = area_trapz(seg)
    # Tail start relative to peak within the segment: use a conservative offset from peak location
    peak_in_seg = peak_idx - lo
    tail_start = int(min(len(seg) - 1, peak_in_seg + max(5, int(0.10 * (cfg.window.post)))))
    tail = seg[tail_start:]
    q_tail = area_trapz(tail)
    ttr = float(q_tail / q_total) if q_total != 0 else np.nan

    feat = dict(
        record=rec_idx,
        peak_idx=int(peak_idx),
        anchor_idx=int(anchor_idx),
        lo=int(lo),
        hi=int(hi),
        amp=amp,
        sigma=sigma,
        snr=amp / sigma,
        rise_samples=float(rise) if rise is not None else np.nan,
        fwhm_samples=fwhm,
        q_total=q_total,
        q_tail=q_tail,
        ttr=ttr,
        is_pileup=is_pileup,
        is_saturated=is_saturated,
    )
    # Optional time units
    if cfg.dt_ns is not None:
        dt = float(cfg.dt_ns)
        feat["peak_time_ns"] = float(peak_idx * dt)
        feat["anchor_time_ns"] = float(anchor_idx * dt)
        feat["rise_ns"] = float(rise * dt) if rise is not None else np.nan
        feat["fwhm_ns"] = float(fwhm * dt) if np.isfinite(fwhm) else np.nan

    return feat


def build_feature_table(wfs: np.ndarray, cfg: Config, limit_records: Optional[int] = None) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    nrec = wfs.shape[0] if limit_records is None else min(limit_records, wfs.shape[0])

    cutflow = dict(
        records_total=nrec,
        records_drift_flagged=0,
        peaks_detected=0,
        pulses_kept=0,
    )

    for i in tqdm(range(nrec), desc="Records"):
        x = wfs[i]
        y, diag = preprocess_record(x, cfg)
        if diag["drift_flag"] > 0.5:
            cutflow["records_drift_flagged"] += 1

        peaks = detect_peaks(y, sigma=diag["sigma_tail"], cfg=cfg.peak)
        cutflow["peaks_detected"] += int(len(peaks))

        for p in peaks:
            feat = extract_pulse(x, y, peak_idx=int(p), sigma=diag["sigma_tail"], rec_idx=i, cfg=cfg)
            if feat is None:
                continue
            feat.update(diag)
            rows.append(feat)
            cutflow["pulses_kept"] += 1

    df = pd.DataFrame(rows)
    df.attrs["cutflow"] = cutflow
    df.attrs["config"] = asdict(cfg)
    return df


# -----------------------------
# Scan proxy (unsupervised)
# -----------------------------

def roc_auc_from_scores(y_true: np.ndarray, score: np.ndarray) -> float:
    """Compute AUC using rank statistic (no sklearn dependency)."""
    y_true = y_true.astype(int)
    score = score.astype(float)
    pos = score[y_true == 1]
    neg = score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    # Mann–Whitney U / AUC
    ranks = score.argsort().argsort().astype(float) + 1.0
    sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
    n_pos = float(len(pos))
    n_neg = float(len(neg))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


# def scan_parameters(wfs: np.ndarray, base_cfg: Config, grid: Dict[str, List[float | int]], limit_records: Optional[int]) -> pd.DataFrame:
#     """
#     Scan a small grid of parameters and score them by how well they separate early vs late pulses.
#     Proxy labels:
#       early = peak_idx < early_fraction * n_samples  (PNS-rich)
#       late  = peak_idx > (1 - late_fraction) * n_samples (background-rich)
#     Score: AUC using TTR as score (higher TTR tends to be neutron-like in many scintillators,
#     but we do NOT assume direction; we take max(AUC, 1-AUC) to make it direction-free).
#     """
#     n_samples = wfs.shape[1]
#     early_cut = int(base_cfg.scan_proxy.early_fraction * n_samples)
#     late_cut = int((1.0 - base_cfg.scan_proxy.late_fraction) * n_samples)

#     # Build cartesian product of grid
#     keys = list(grid.keys())
#     values = [grid[k] for k in keys]

#     results = []
#     for combo in tqdm(list(product(*values)), desc="Scan points"):
#         cfg_dict = asdict(base_cfg)

#         # apply combo
#         for k, v in zip(keys, combo):
#             # support dotted keys like "window.pre"
#             parts = k.split(".")
#             d = cfg_dict
#             for part in parts[:-1]:
#                 d = d[part]
#             d[parts[-1]] = v

#         # rehydrate Config (dataclasses) from dict
#         cfg = Config(
#             baseline=BaselineConfig(**cfg_dict["baseline"]),
#             peak=PeakFindConfig(**cfg_dict["peak"]),
#             align=AlignConfig(**cfg_dict["align"]),
#             window=WindowConfig(**cfg_dict["window"]),
#             cuts=CutsConfig(**cfg_dict["cuts"]),
#             scan_proxy=ScanProxyConfig(**cfg_dict["scan_proxy"]),
#             dt_ns=cfg_dict.get("dt_ns", None),
#         )

#         df = build_feature_table(wfs, cfg, limit_records=limit_records)
#         if len(df) < cfg.scan_proxy.min_pulses_total:
#             continue

#         # early/late labeling
#         y_true = np.full(len(df), -1, dtype=int)
#         y_true[df["peak_idx"].to_numpy() < early_cut] = 1
#         y_true[df["peak_idx"].to_numpy() > late_cut] = 0
#         keep = y_true >= 0
#         y_true = y_true[keep]
#         if len(y_true) < cfg.scan_proxy.min_pulses_total:
#             continue

#         # require per-class counts
#         n_pos = int(np.sum(y_true == 1))
#         n_neg = int(np.sum(y_true == 0))
#         if n_pos < cfg.scan_proxy.min_pulses_per_class or n_neg < cfg.scan_proxy.min_pulses_per_class:
#             continue

#         score = df.loc[keep, "ttr"].to_numpy()
#         # remove NaNs
#         m = np.isfinite(score)
#         y2 = y_true[m]
#         s2 = score[m]
#         if len(s2) < cfg.scan_proxy.min_pulses_total:
#             continue

#         auc = roc_auc_from_scores(y2, s2)
#         auc_dirfree = float(max(auc, 1.0 - auc)) if np.isfinite(auc) else np.nan

#         kept = len(df)
#         pileup_frac = float(np.mean(df["is_pileup"].to_numpy())) if kept else np.nan

#         res = dict(
#             auc_ttr=auc_dirfree,
#             pulses_kept=kept,
#             pileup_frac=pileup_frac,
#             early_cut=early_cut,
#             late_cut=late_cut,
#         )
#         for k, v in zip(keys, combo):
#             res[k] = v
#         results.append(res)

#     out = pd.DataFrame(results).sort_values(["auc_ttr", "pulses_kept"], ascending=[False, False])
#     return out

def scan_parameters(
    wfs: np.ndarray,
    base_cfg: Config,
    grid: Dict[str, List[float | int]],
    limit_records: Optional[int],
) -> pd.DataFrame:
    """
    Scan a grid of parameters and score them by how well they separate early vs late pulses.

    Proxy labels:
      early = peak_idx < early_fraction * n_samples  (PNS-rich)
      late  = peak_idx > (1 - late_fraction) * n_samples (background-rich)

    Base separation metric:
      AUC of TTR as a score (direction-free: max(AUC, 1-AUC))

    Added diagnostics (to avoid pathological window choices):
      - tail_ok_frac: fraction of pulses with *usable tail* (enough room after peak + finite, nonzero TTR)
      - rise_ok_frac: fraction of pulses where peak is not too close to the left window edge
      - score = auc_dirfree * tail_ok_frac * rise_ok_frac

    Output is sorted by score, then pulses_kept.
    """
    n_samples = wfs.shape[1]
    early_cut = int(base_cfg.scan_proxy.early_fraction * n_samples)
    late_cut = int((1.0 - base_cfg.scan_proxy.late_fraction) * n_samples)

    # Cartesian product of grid
    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    results = []
    for combo in tqdm(list(product(*values)), desc="Scan points"):
        cfg_dict = asdict(base_cfg)

        # apply combo (supports dotted keys like "window.pre")
        for k, v in zip(keys, combo):
            parts = k.split(".")
            d = cfg_dict
            for part in parts[:-1]:
                d = d[part]
            d[parts[-1]] = v

        # rehydrate Config
        cfg = Config(
            baseline=BaselineConfig(**cfg_dict["baseline"]),
            peak=PeakFindConfig(**cfg_dict["peak"]),
            align=AlignConfig(**cfg_dict["align"]),
            window=WindowConfig(**cfg_dict["window"]),
            cuts=CutsConfig(**cfg_dict["cuts"]),
            scan_proxy=ScanProxyConfig(**cfg_dict["scan_proxy"]),
            dt_ns=cfg_dict.get("dt_ns", None),
        )

        df = build_feature_table(wfs, cfg, limit_records=limit_records)
        kept = len(df)
        if kept < cfg.scan_proxy.min_pulses_total:
            continue

        # early/late labeling
        y_true = np.full(kept, -1, dtype=int)
        peak_idx = df["peak_idx"].to_numpy()
        y_true[peak_idx < early_cut] = 1
        y_true[peak_idx > late_cut] = 0

        keep = y_true >= 0
        if int(np.sum(keep)) < cfg.scan_proxy.min_pulses_total:
            continue

        y_true = y_true[keep]

        # require per-class counts
        n_pos = int(np.sum(y_true == 1))
        n_neg = int(np.sum(y_true == 0))
        if n_pos < cfg.scan_proxy.min_pulses_per_class or n_neg < cfg.scan_proxy.min_pulses_per_class:
            continue

        # scores = ttr
        ttr = df.loc[keep, "ttr"].to_numpy()
        finite = np.isfinite(ttr)
        if int(np.sum(finite)) < cfg.scan_proxy.min_pulses_total:
            continue

        y2 = y_true[finite]
        s2 = ttr[finite]

        auc = roc_auc_from_scores(y2, s2)
        auc_dirfree = float(max(auc, 1.0 - auc)) if np.isfinite(auc) else np.nan
        if not np.isfinite(auc_dirfree):
            continue

        # --------- Added diagnostics ----------
        # Peak position inside window
        lo = df.loc[keep, "lo"].to_numpy()[finite]
        hi = df.loc[keep, "hi"].to_numpy()[finite]
        pk = df.loc[keep, "peak_idx"].to_numpy()[finite]

        peak_rel = pk - lo                  # samples from window start to peak
        right_margin = hi - pk              # samples from peak to window end

        # Heuristics: require some baseline before rise, and enough samples after peak for tail.
        # (These are intentionally simple and robust.)
        min_pre_margin = max(10, int(0.25 * cfg.window.pre))     # at least 10 samples before peak
        min_tail_margin = max(80, int(0.40 * cfg.window.post))   # at least ~80 samples after peak

        rise_ok = peak_rel >= min_pre_margin

        # Tail is "ok" if there's enough room after peak AND TTR is positive (not zero-floor)
        tail_ok = (right_margin >= min_tail_margin) & (s2 > 0)

        rise_ok_frac = float(np.mean(rise_ok)) if len(rise_ok) else np.nan
        tail_ok_frac = float(np.mean(tail_ok)) if len(tail_ok) else np.nan

        # Composite score: separation * usability
        score = float(auc_dirfree * rise_ok_frac * tail_ok_frac)
        # -------------------------------------

        pileup_frac = float(np.mean(df["is_pileup"].to_numpy())) if kept else np.nan
        sat_frac = float(np.mean(df["is_saturated"].to_numpy())) if ("is_saturated" in df.columns and kept) else np.nan

        res = dict(
            score=score,
            auc_ttr=auc_dirfree,
            rise_ok_frac=rise_ok_frac,
            tail_ok_frac=tail_ok_frac,
            pulses_kept=kept,
            pulses_used=int(np.sum(keep) if keep is not None else 0),
            pileup_frac=pileup_frac,
            sat_frac=sat_frac,
            early_cut=early_cut,
            late_cut=late_cut,
            min_pre_margin=min_pre_margin,
            min_tail_margin=min_tail_margin,
        )
        for k, v in zip(keys, combo):
            res[k] = v

        results.append(res)

    out = pd.DataFrame(results).sort_values(["score", "pulses_kept"], ascending=[False, False])
    return out

# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pulse extraction and parameter scan for SiPM trigger-window waveforms.")
    p.add_argument("--input", type=str, required=True, help="Path to .npz containing key 'wfs'")
    p.add_argument("--out", type=str, default="pulses.parquet", help="Output file: .parquet or .csv")
    p.add_argument("--limit-records", type=int, default=None, help="Process only first N records (for quick tests)")
    p.add_argument("--dt-ns", type=float, default=None, help="Optional sampling period (ns/sample) for time features")

    p.add_argument("--scan", action="store_true", help="Run parameter scan instead of single extraction")
    p.add_argument("--scan-out", type=str, default="scan_results.csv", help="Scan results output (.csv/.parquet)")
    p.add_argument("--grid", type=str, default=None,
                   help="JSON dict of scan grid. Example: "
                        "'{\"window.pre\":[20,30],\"window.post\":[150,200],\"peak.min_height_sigma\":[5,6,7]}'")

    # Minimal override convenience flags
    p.add_argument("--pre", type=int, default=None, help="Override window.pre")
    p.add_argument("--post", type=int, default=None, help="Override window.post")
    p.add_argument("--min-height-sigma", type=float, default=None, help="Override peak.min_height_sigma")
    p.add_argument("--min-prom-sigma", type=float, default=None, help="Override peak.min_prom_sigma")
    return p.parse_args()


def save_df(df: pd.DataFrame, path: str) -> None:
    path = str(path)
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif path.endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unknown output extension for {path}. Use .parquet or .csv")


def main() -> None:
    args = parse_args()
    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(inp)

    npz = np.load(str(inp), allow_pickle=True)
    if "wfs" not in npz:
        raise KeyError(f"{inp} does not contain key 'wfs'. Keys: {list(npz.keys())}")
    wfs = npz["wfs"]

    cfg = Config(dt_ns=args.dt_ns)

    # Apply simple overrides
    cfg_dict = asdict(cfg)
    if args.pre is not None:
        cfg_dict["window"]["pre"] = int(args.pre)
    if args.post is not None:
        cfg_dict["window"]["post"] = int(args.post)
    if args.min_height_sigma is not None:
        cfg_dict["peak"]["min_height_sigma"] = float(args.min_height_sigma)
    if args.min_prom_sigma is not None:
        cfg_dict["peak"]["min_prom_sigma"] = float(args.min_prom_sigma)

    cfg = Config(
        baseline=BaselineConfig(**cfg_dict["baseline"]),
        peak=PeakFindConfig(**cfg_dict["peak"]),
        align=AlignConfig(**cfg_dict["align"]),
        window=WindowConfig(**cfg_dict["window"]),
        cuts=CutsConfig(**cfg_dict["cuts"]),
        scan_proxy=ScanProxyConfig(**cfg_dict["scan_proxy"]),
        dt_ns=cfg_dict.get("dt_ns", None),
    )

    if args.scan:
        if args.grid is None:
            # sane default scan
            grid = {
                "window.pre": [20, 30, 40],
                "window.post": [150, 200, 260],
                "peak.min_height_sigma": [5.0, 6.0, 7.0],
                "peak.min_prom_sigma": [3.0, 4.0, 5.0],
                "cuts.pileup_secondary_frac": [0.15, 0.20, 0.25],
            }
        else:
            grid = json.loads(args.grid)
        scan_df = scan_parameters(wfs, cfg, grid=grid, limit_records=args.limit_records)
        save_df(scan_df, args.scan_out)
        print(f"\nScan saved to: {args.scan_out}")
        if len(scan_df):
            print("\nTop 10 scan points:")
            print(scan_df.head(10).to_string(index=False))
        else:
            print("\nNo scan points passed minimum-statistics filters.")
        return

    df = build_feature_table(wfs, cfg, limit_records=args.limit_records)
    save_df(df, args.out)

    print(f"\nSaved features to: {args.out}")
    if "cutflow" in df.attrs:
        print("\nCutflow:")
        for k, v in df.attrs["cutflow"].items():
            print(f"  {k}: {v}")
    print("\nColumns:")
    print(list(df.columns))


if __name__ == "__main__":
    main()
