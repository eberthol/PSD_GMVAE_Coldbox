import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import awkward as ak
import tabulate as tab
from tqdm import tqdm
from scipy.signal import find_peaks

# ---------- baseline subtraction ----------
def estimate_baseline_chunks(record: np.ndarray, chunk: int = 2000, q: float = 10.0) -> np.ndarray:
    """Robust baseline per chunk using percentile q.

    Divide the trigger record into chunks (the last one may be shorter than the provided value)
    In each chunk, compute the percentile: e.g. q=10 returns the value below which 10% of the samples in that chunk lie

    Return all chunk baselines as an array
    """
    n = len(record)
    return np.asarray(
        [np.percentile(record[i:i + chunk], q) for i in range(0, n, chunk)],
        dtype=np.float64
    )

def interpolate_baseline_chunks(baseline_chunks: np.ndarray, n_samples: int, chunk: int) -> np.ndarray:
    """Convert per-chunk baseline estimates into a per-sample baseline (piecewise-linear interpolation)."""
    centers = np.arange(len(baseline_chunks), dtype=np.float64) * chunk + (chunk - 1) / 2
    centers = np.clip(centers, 0, n_samples - 1)
    return np.interp(np.arange(n_samples, dtype=np.float64), centers, baseline_chunks)

def fancy_baseline_subtraction_record(
    rec: np.ndarray,
    *,
    chunk: int,
    q: float,
    drift_ptp_threshold: float,
):
    """
    Baseline subtraction for a single trigger record.

    Returns:
      baseline_sub_rec
      baseline_per_sample
      drift_ptp
      flagged
    """
    bchunks = estimate_baseline_chunks(rec, chunk=chunk, q=q)

    # check how much the baseline changes accross the trigger record (ptp of the baseline estimates)
    drift_ptp = float(np.ptp(bchunks)) if len(bchunks) else 0.0 # np.ptp returns peak-to-peak values in an array along a specific axis
    flagged = drift_ptp > drift_ptp_threshold

    baseline_per_sample = interpolate_baseline_chunks( bchunks, n_samples=len(rec), chunk=chunk)

    baseline_sub_rec = rec - baseline_per_sample

    return baseline_sub_rec, baseline_per_sample, drift_ptp, flagged

# ---------- alignment on rising-edge ----------
def constant_fraction_crossing_index(y: np.ndarray, peak_idx: int, frac: float = 0.2, search_back: int = 400):
    """First rising-edge crossing of frac*peak height. Returns int index or None.
            1.	Takes the baseline-subtracted waveform
            2.	Looks backward from the peak index (limit for search is controlled by search_back)
            3.	Finds the first sample where the waveform crosses: threshold = align_frac x peak_height
    """
    peak_val = y[peak_idx]
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

# ---------- build dataset ----------
def local_max(arr: np.ndarray, idx: int, pad: int = 2) -> float:
    lo = max(0, idx - pad)
    hi = min(len(arr), idx + pad + 1)
    return float(np.max(arr[lo:hi]))

def _print_cutflow(c):
    peaks = c["peaks_detected"]
    def pct(x):
        return 0.0 if peaks == 0 else 100.0 * x / peaks

    print("\n=== Cutflow summary ===")
    print(f"Records total:            {c['records_total']}")
    print(f"Records drift-flagged:    {c['records_flagged_drift']}")
    print(f"Records skipped (drift):  {c['records_skipped_drift']}")
    print("")
    print(f"Peaks detected:           {peaks}")
    print(f"Removed: failed align:    {c['cut_failed_align']}  ({pct(c['cut_failed_align']):.2f}%)")
    print(f"Removed: saturation:      {c['cut_saturation']}  ({pct(c['cut_saturation']):.2f}%)")
    print(f"Removed: amp_min:         {c['cut_amp_min']}  ({pct(c['cut_amp_min']):.2f}%)")
    print(f"Removed: amp_max:         {c['cut_amp_max']}  ({pct(c['cut_amp_max']):.2f}%)")
    print(f"Peaks kept after cuts:    {c['peaks_after_cuts']}  "
          f"({0.0 if peaks==0 else 100.0*c['peaks_after_cuts']/peaks:.2f}%)")
    print("")
    print(f"Removed: edge window:     {c['cut_edge_window']}  ({pct(c['cut_edge_window']):.2f}%)")
    print(f"Removed: peak outside win:{c['cut_peak_outside_window']}  ({pct(c['cut_peak_outside_window']):.2f}%)")
    print("")
    print(f"Final pulses extracted:   {c['pulses_extracted']}  "
          f"({0.0 if peaks==0 else 100.0*c['pulses_extracted']/peaks:.2f}% of detected)")
    print("========================\n")

def build_pulse_dataset(
    raw_data: np.ndarray,
    *,
    pre: int = 40,
    post: int = 216,

    # Baseline
    baseline_chunk: int = 2000,
    baseline_q: float = 10.0,
    drift_ptp_threshold: float = 200.0, # if the peak to peak variation of the baseline in a given trigger record is above the threshold, then the record is flagged as unstable

    # Peak detection on baseline-subtracted waveform
    height: float = 80.0,
    prominence: float = 150.0,
    distance: int = 150,

    # Alignment
    use_pos: str = "align",      # "align" (CFD) or "argmax"
    align_frac: float = 0.2,
    align_search_back: int = 400,
    drop_failed_align: bool = True,

    # Cuts
    ADCsat: float | None = None,      # drop pulses if peak_raw_localmax >= ADCsat
    peak_raw_pad: int = 2,            # local max pad for raw peak
    amp_min: float | None = None,     # cut on baseline-subtracted peak height at argmax
    amp_max: float | None = None,     # cut on baseline-subtracted peak height at argmax

    # Record filter
    skip_flagged_records: bool = False,

    # Storage
    store_raw_window: bool = True,
    store_sub_window: bool = True,
    store_baseline_interpolation: bool = False,   # usually False (big); enable for debugging
    dtype=np.float32,

    # Reporting
    verbose: bool = True,
):
    """
    Builds a pulse-level dataset directly from raw_data

    Returns dict with keys:
      pulses_raw: (N,L) raw windows (optional)
      pulses_sub: (N,L) baseline-sub windows (optional)
      record_index: (N,)
      pos: (N,) window center index (align_idx or peak_idx)
      peak_idx: (N,) argmax index in trigger
      align_idx: (N,) CFD crossing index (or -1)
      peak_sub: (N,) baseline-sub peak height at argmax
      peak_raw: (N,) raw ADC at argmax sample
      peak_raw_localmax: (N,) raw local max around argmax (pad=peak_raw_pad)
      drift_ptp: (N_records,) per-record drift ptp
      flagged_drift: (N_records,) per-record flag
      (optional) baseline_interpolations_per_record: list of arrays if store_baseline_interpolation=True
    """

    use_pos = str(use_pos).strip().lower()
    if use_pos not in ("align", "argmax"):
        raise ValueError("use_pos must be 'align' or 'argmax'")

    n_records, n_samples = raw_data.shape
    L = pre + post

    # Cutflow
    cutflow = {
        "records_total": 0,
        "records_flagged_drift": 0,
        "records_skipped_drift": 0,

        "peaks_detected": 0,
        "peaks_after_cuts": 0,

        "cut_failed_align": 0,
        "cut_saturation": 0,
        "cut_amp_min": 0,
        "cut_amp_max": 0,

        "cut_edge_window": 0,
        "cut_peak_outside_window": 0,

        "pulses_extracted": 0,
    }

    # Per-record diagnostics
    drift_ptp_arr = np.zeros(n_records, dtype=np.float32)
    flagged_arr = np.zeros(n_records, dtype=bool)
    baseline_interpolations = [] if store_baseline_interpolation else None

    # Pulse-level outputs
    pulses_raw = []
    pulses_sub = []
    record_index = []
    pos_used = []
    peak_idx_list = []
    align_idx_list = []
    peak_sub_list = []
    peak_raw_list = []
    peak_raw_lmax_list = []

    for r in range(n_records):
        rec = raw_data[r].astype(np.float64, copy=False)

        # ---- Baseline subtraction ----
        baselineSub_rec, baseline_per_sample, drift_ptp, flagged = fancy_baseline_subtraction_record(
            rec,
            chunk=baseline_chunk,
            q=baseline_q,
            drift_ptp_threshold=drift_ptp_threshold,
        )
        drift_ptp_arr[r] = drift_ptp # store ptp per trigger record
        flagged_arr[r] = flagged

        cutflow["records_total"] += 1
        if flagged:
            cutflow["records_flagged_drift"] += 1
            if skip_flagged_records:
                cutflow["records_skipped_drift"] += 1
                continue

        if skip_flagged_records and flagged:
            continue

        if store_baseline_interpolation:
            baseline_interpolations.append(baseline_per_sample.astype(dtype, copy=False))

        # --- Peak detection ---
        pk_idx, props = find_peaks(
            baselineSub_rec,
            height=height,
            prominence=prominence,
            distance=distance,
        )
        if pk_idx.size == 0:
            continue

        pk_idx = pk_idx.astype(np.int64) # peak positions in the trigger record
        baselineSub_peak_heights = props["peak_heights"].astype(np.float64) # baseline subtracted peak heights

        # Raw peak values at argmax and local max
        pk_raw_heights = rec[pk_idx].astype(np.float64) # raw peak heights
        pk_raw_lmax_heights = np.asarray([local_max(rec, int(p), pad=peak_raw_pad) for p in pk_idx], dtype=np.float64) # more robust estimate than pk_raw

        # --- CFD alignment (computed for all peaks, regardless of use_pos) ---
        align_idx = np.full(len(pk_idx), -1, dtype=np.int64) # alignement array
        for i, p in enumerate(pk_idx): # loop over detected peaks
            a = constant_fraction_crossing_index(baselineSub_rec, int(p), frac=align_frac, search_back=align_search_back)
            align_idx[i] = -1 if a is None else int(a) # -1 is the value if constant_fraction_crossing_index() failed

        # --- Selection mask with cutflow ---
        cutflow["peaks_detected"] += int(len(pk_idx))
        keep = np.ones(len(pk_idx), dtype=bool)

        # 1) drop failed aligns
        if drop_failed_align:
            m = (align_idx >= 0)
            cutflow["cut_failed_align"] += int(np.sum(keep & ~m))
            keep &= m

        # 3) amp_min cut
        if amp_min is not None:
            m = (baselineSub_peak_heights >= float(amp_min))
            cutflow["cut_amp_min"] += int(np.sum(keep & ~m))
            keep &= m

        # 4) amp_max cut
        if amp_max is not None:
            m = (baselineSub_peak_heights <= float(amp_max))
            cutflow["cut_amp_max"] += int(np.sum(keep & ~m))
            keep &= m

        # Apply cuts
        pk_idx = pk_idx[keep]
        baselineSub_peak_heights = baselineSub_peak_heights[keep]
        pk_raw_heights = pk_raw_heights[keep]
        pk_raw_lmax_heights = pk_raw_lmax_heights[keep]
        align_idx = align_idx[keep]

        cutflow["peaks_after_cuts"] += int(len(pk_idx))
        if pk_idx.size == 0:
            continue

        # --- Choose window center index ---
        # not the real center but the point chosen to compute the window
        if use_pos == "align":
            pos = align_idx.copy()
            # If you didn't drop failed aligns, optionally fall back to argmax
            bad = pos < 0
            if np.any(bad):
                pos[bad] = pk_idx[bad]
        else:
            pos = pk_idx
        
        # ---- Extract windows  ----
        for p_center, p_peak, a_idx, ps, pr, prl in zip(
            pos, pk_idx, align_idx, baselineSub_peak_heights, pk_raw_heights, pk_raw_lmax_heights
        ):
            if p_center < 0:
                # happens only if use_pos="align" and you keep failed aligns without fallback
                cutflow["cut_failed_align"] += 1
                continue

            start = int(p_center) - pre
            end = int(p_center) + post
            if start < 0 or end > n_samples:
                cutflow["cut_edge_window"] += 1
                continue

            raw_win = rec[start:end]  # slice once
            # Saturation cut: ANY sample in the extracted raw window hits ADCsat
            if ADCsat is not None and float(np.max(raw_win)) >= float(ADCsat):
                cutflow["cut_saturation"] += 1
                continue

            if store_raw_window:
                pulses_raw.append(raw_win)
            if store_sub_window:
                pulses_sub.append(baselineSub_rec[start:end])

            record_index.append(r)
            pos_used.append(int(p_center))
            peak_idx_list.append(int(p_peak))
            align_idx_list.append(int(a_idx))
            peak_sub_list.append(float(ps))
            peak_raw_list.append(float(pr))
            peak_raw_lmax_list.append(float(prl))

    cutflow["pulses_extracted"] = len(record_index)

    out = {
        "record_index": np.asarray(record_index, dtype=np.int32),
        "pos": np.asarray(pos_used, dtype=np.int32), # not the real center but the point chosen to compute the window, it is either equal to peak_idx or align_idx
        "peak_idx": np.asarray(peak_idx_list, dtype=np.int32),
        "align_idx": np.asarray(align_idx_list, dtype=np.int32),

        "baselineSub_peak_height": np.asarray(peak_sub_list, dtype=dtype),
        "peak_raw_height": np.asarray(peak_raw_list, dtype=dtype),
        "peak_raw_localmax_height": np.asarray(peak_raw_lmax_list, dtype=dtype),

        "drift_ptp_per_record": drift_ptp_arr,   # ptp per trigger record
        "flagged_drift_per_record": flagged_arr, # trigger records flagged for unstable baseline
 
        #--- metadata (from arguments of functions...)
        # window before and after "peak"
        "pre": int(pre),
        "post": int(post),
        "L": int(L), # = pre + post
        # args of fancy_baseline_subtraction_record
        "baseline_chunk": int(baseline_chunk),
        "baseline_q": float(baseline_q),
        "drift_ptp_threshold": float(drift_ptp_threshold),
        # args of find_peaks
        "height": float(height), 
        "prominence": float(prominence), 
        "distance": int(distance),
        # alignment  
        "use_pos": use_pos, # "align" (CFD) or "argmax"
        "align_frac": float(align_frac),
        "align_search_back": int(align_search_back),
        "drop_failed_align": bool(drop_failed_align),

        "ADCsat": None if ADCsat is None else float(ADCsat),
        "amp_min": None if amp_min is None else float(amp_min),
        "amp_max": None if amp_max is None else float(amp_max),

        "cutflow": cutflow,


    }

    if store_raw_window:
        out["pulses_raw"] = np.asarray(pulses_raw, dtype=dtype) if len(pulses_raw) else np.empty((0, L), dtype=dtype)
    if store_sub_window:
        out["pulses_sub"] = np.asarray(pulses_sub, dtype=dtype) if len(pulses_sub) else np.empty((0, L), dtype=dtype)
    if store_baseline_interpolation:
        out["baseline_interpolations_per_record"] = baseline_interpolations

    if verbose:
        _print_cutflow(cutflow)
    
    return out

# ---------- plotting functions ----------

def plot_pulses(
    pulse_ds,
    *,
    waveform: str = "sub",                 # "sub" or "raw"
    use_pos: str = "align",                # "align" or "argmax"

    pos_side: str = "none",                # "before", "after", "between", "none"
    pos_cut=None,                          # int or (lo,hi); ignored if pos_side="none"

    amp_side: str = "none",                # "below", "above", "between", "none"
    amp_cut=None,                          # float or (lo,hi); ignored if amp_side="none"
    amp_source: str = "baselineSub_peak_height",  # key in pulse_ds

    ADCsat: float | None = None,           # optional saturation cut
    sat_source: str = "peak_raw_localmax_height", # key in pulse_ds

    max_pulses_to_plot: int | None = 2000,
    show_mean_band: bool = True,
    title: str | None = None,
    show_plot: bool = True
):
    """
    Plot pulses from pulse_ds with optional cuts on:
      - position in trigger record (align_idx or peak_idx)
      - amplitude (any pulse_ds field, default baselineSub_peak_height)
      - saturation (any pulse_ds field, default peak_raw_localmax_height)

    Uses window info from pulse_ds: pre/post/L. No need to pass them.

    Returns:
      windows: (N_selected, L)
      meta: dict of selected indices and key metadata
      mean: mean value of the selected pulses
      std: 1sigma value of the selected pulses
    """
    waveform = str(waveform).strip().lower()
    if waveform not in ("sub", "raw"):
        raise ValueError("waveform must be 'sub' or 'raw'")

    use_pos = str(use_pos).strip().lower()
    if use_pos not in ("align", "argmax"):
        raise ValueError("use_pos must be 'align' or 'argmax'")

    pos_side = str(pos_side).strip().lower()
    if pos_side not in ("before", "after", "between", "none"):
        raise ValueError("pos_side must be before/after/between/none")

    amp_side = str(amp_side).strip().lower()
    if amp_side not in ("below", "above", "between", "none"):
        raise ValueError("amp_side must be below/above/between/none")

    # ---- window parameters from pulse_ds ----
    pre = int(pulse_ds.get("pre", 0))
    post = int(pulse_ds.get("post", 0))
    L = int(pulse_ds.get("L", pre + post))

    # waveform array
    wf_key = "pulses_sub" if waveform == "sub" else "pulses_raw"
    if wf_key not in pulse_ds:
        raise KeyError(f"pulse_ds has no '{wf_key}'")
    W = np.asarray(pulse_ds[wf_key])

    if L <= 0:
        L = W.shape[1]

    # position array (trigger-record index)
    pos_key = "align_idx" if use_pos == "align" else "peak_idx"
    if pos_key not in pulse_ds:
        raise KeyError(f"pulse_ds has no '{pos_key}'")
    pos_arr = np.asarray(pulse_ds[pos_key], dtype=np.int64)

    # amplitude array
    if amp_side != "none":
        if amp_source not in pulse_ds:
            raise KeyError(f"amp_source='{amp_source}' not found in pulse_ds")
        amp_arr = np.asarray(pulse_ds[amp_source], dtype=np.float64)

    # saturation array
    if ADCsat is not None:
        if sat_source not in pulse_ds:
            raise KeyError(f"sat_source='{sat_source}' not found in pulse_ds")
        sat_arr = np.asarray(pulse_ds[sat_source], dtype=np.float64)

    # ---- build selection mask ----
    mask = np.ones(len(pos_arr), dtype=bool)

    # valid align indices
    if use_pos == "align":
        mask &= (pos_arr >= 0)

    # position cut
    if pos_side != "none":
        if pos_cut is None:
            raise ValueError("pos_cut must be set when pos_side != 'none'")
        if pos_side == "before":
            mask &= (pos_arr < int(pos_cut))
        elif pos_side == "after":
            mask &= (pos_arr > int(pos_cut))
        else:
            lo, hi = pos_cut
            mask &= (pos_arr >= int(lo)) & (pos_arr <= int(hi))

    # amplitude cut
    if amp_side != "none":
        if amp_cut is None:
            raise ValueError("amp_cut must be set when amp_side != 'none'")
        if amp_side == "below":
            mask &= (amp_arr < float(amp_cut))
        elif amp_side == "above":
            mask &= (amp_arr > float(amp_cut))
        else:
            lo, hi = amp_cut
            mask &= (amp_arr >= float(lo)) & (amp_arr <= float(hi))

    # saturation cut
    if ADCsat is not None:
        mask &= (sat_arr < float(ADCsat))

    idx_sel = np.where(mask)[0]
    if idx_sel.size == 0:
        print("No pulses matched the selection (try changing cuts).")
        return np.empty((0, L)), {}

    windows = W[idx_sel]

    # meta
    meta = {
        "index": idx_sel.astype(np.int32),
        "record_index": np.asarray(pulse_ds.get("record_index", -np.ones(len(pos_arr))))[idx_sel].astype(np.int32),
        "pos_in_trigger": pos_arr[idx_sel].astype(np.int32),
        "peak_idx": np.asarray(pulse_ds.get("peak_idx", -np.ones(len(pos_arr))))[idx_sel].astype(np.int32),
        "align_idx": np.asarray(pulse_ds.get("align_idx", -np.ones(len(pos_arr))))[idx_sel].astype(np.int32),
    }
    # include amplitude/sat columns if present
    for k in (amp_source, sat_source):
        if k in pulse_ds:
            meta[k] = np.asarray(pulse_ds[k])[idx_sel]

    # plotting downsample only
    windows_plot = windows
    if max_pulses_to_plot is not None and len(windows) > max_pulses_to_plot:
        take = np.random.choice(len(windows), size=max_pulses_to_plot, replace=False)
        windows_plot = windows[take]

    mean = windows.mean(axis=0)
    std = windows.std(axis=0)

    # ---- plot ----
    if show_plot:
        fig, (ax_all, ax_mean) = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
        t = np.arange(L) - pre

        ax_all.plot(t, windows_plot.T, linewidth=0.6, alpha=0.15)
        ax_all.axvline(0, linestyle="--", linewidth=1)
        ax_all.set_xlabel("Samples relative to window center (0)")
        ax_all.set_ylabel(f"ADC ({'baseline-subtracted' if waveform=='sub' else 'raw'})")
        ax_all.grid(True, alpha=0.25)

        ax_mean.plot(t, mean, linewidth=2.0)
        ax_mean.axvline(0, linestyle="--", linewidth=1)
        if show_mean_band:
            ax_mean.fill_between(t, mean - std, mean + std, alpha=0.15)
        ax_mean.set_title("Mean pulse (±1σ)" if show_mean_band else "Mean pulse")
        ax_mean.set_xlabel("Samples relative to window center (0)")
        ax_mean.set_ylabel(f"ADC ({'baseline-subtracted' if waveform=='sub' else 'raw'})")
        ax_mean.grid(True, alpha=0.25)

        if title is None:
            title = f"Selected pulses | wf={waveform} | pos={pos_key} | N={len(windows)}"
            if pos_side != "none":
                title += f" | {pos_side} {pos_cut}"
            if amp_side != "none":
                title += f" | {amp_source} {amp_side} {amp_cut}"
            if ADCsat is not None:
                title += f" | {sat_source}<{ADCsat}"
        ax_all.set_title(title)
        plt.show()

    return windows, meta, mean, std

def plot_peak_position_histogram(
    pulse_ds,
    *,
    use: str = "align",                 # "align" uses align_idx, "argmax" uses peak_idx
    bins: int | np.ndarray = 200,
    range: tuple[int, int] | None = None,
    density: bool = False,
    title: str | None = None
):
    """
    Histogram of peak positions across all pulses in pulse_ds.
    """
    use = str(use).strip().lower()
    if use not in ("align", "argmax"):
        raise ValueError("use must be 'align' or 'argmax'")

    key = "align_idx" if use == "align" else "peak_idx"
    pos = np.asarray(pulse_ds[key], dtype=np.int64)

    mask = np.ones(len(pos), dtype=bool)
    if key == "align_idx":
        mask &= (pos >= 0)

    pos = pos[mask]
    if pos.size == 0:
        print("No positions to plot after cuts.")
        return pos

    fig, ax = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
    ax.hist(pos, bins=bins, range=range, density=density)
    ax.set_xlabel(f"Peak position in trigger (sample index) [{key}]")
    ax.set_ylabel("Density" if density else "Count")
    ax.grid(True, alpha=0.25)

    if title is None:
        title = f"Histogram of peak positions ({key}), N={len(pos)}"
        # if ADCsat is not None:
        #     title += f" | {sat_source}<{ADCsat}"
    ax.set_title(title)

    plt.show()
    return pos

def plot_record(
    raw_data,
    pulse_ds,
    *,
    record_index: int,
    marker_use: str = "pos",     # "pos", "align_idx", "peak_idx"
    windows: str = "sub",        # "sub" or "raw"
    max_pulses_to_plot: int | None = 200,
    show_baseline_overlay: bool = True,
    show_window_center_line: bool = True,
    allow_baseline_fallback: bool = True,  # set False to *require* stored baseline
):
    """
    Plot for one trigger record.

    Uses stored baseline interpolation if available in pulse_ds:
      pulse_ds["baseline_interpolations_per_record"][ri]

    If not available, can optionally fall back to a simple percentile-per-chunk estimate.
    """
    marker_use = str(marker_use).strip().lower()
    if marker_use not in ("pos", "align_idx", "peak_idx"):
        raise ValueError("marker_use must be 'pos', 'align_idx', or 'peak_idx'")

    windows = str(windows).strip().lower()
    if windows not in ("sub", "raw"):
        raise ValueError("windows must be 'sub' or 'raw'")

    ri = int(record_index)
    rec = np.asarray(raw_data[ri], dtype=np.float64)
    n = rec.size

    # ---- baseline + baseline-subtracted record (prefer stored baseline) ----
    baseline = None
    if "baseline_interpolations_per_record" in pulse_ds:
        b_list = pulse_ds["baseline_interpolations_per_record"]
        if ri < 0 or ri >= len(b_list):
            raise IndexError(f"record_index={ri} out of range for stored baselines (len={len(b_list)})")
        baseline = np.asarray(b_list[ri], dtype=np.float64)
        if baseline.size != n:
            raise ValueError(f"Stored baseline length {baseline.size} != record length {n}")

    if baseline is None:
        if not allow_baseline_fallback:
            raise KeyError(
                "No stored baseline in pulse_ds. Rebuild pulse_ds with store_baseline_interpolation=True "
                "or set allow_baseline_fallback=True."
            )

        # lightweight fallback baseline estimate (same idea as before)
        chunk = int(pulse_ds.get("baseline_chunk", 2000))
        q = float(pulse_ds.get("baseline_q", 10.0))
        b = np.asarray([np.percentile(rec[i:i + chunk], q) for i in range(0, n, chunk)], dtype=np.float64)
        centers = np.arange(len(b), dtype=np.float64) * chunk + (chunk - 1) / 2
        centers = np.clip(centers, 0, n - 1)
        baseline = np.interp(np.arange(n, dtype=np.float64), centers, b)

    y = rec - baseline

    # ---- pulses belonging to this record ----
    rec_of_pulse = np.asarray(pulse_ds["record_index"], dtype=np.int32)
    idx = np.where(rec_of_pulse == ri)[0]
    if idx.size == 0:
        print(f"No pulses in pulse_ds for record_index={ri}")
        return np.empty((0, 0)), idx

    # markers in trigger-record sample indices
    if marker_use == "pos":
        if "pos" not in pulse_ds:
            raise KeyError("pulse_ds has no 'pos' key.")
        marks = np.asarray(pulse_ds["pos"], dtype=np.int64)[idx]
    elif marker_use == "align_idx":
        marks = np.asarray(pulse_ds.get("align_idx", []), dtype=np.int64)[idx]
    else:
        marks = np.asarray(pulse_ds.get("peak_idx", []), dtype=np.int64)[idx]

    marks = marks[(marks >= 0) & (marks < n)]

    # ---- windows from pulse_ds ----
    win_key = "pulses_sub" if windows == "sub" else "pulses_raw"
    if win_key not in pulse_ds:
        raise KeyError(f"pulse_ds has no '{win_key}' key.")
    W = np.asarray(pulse_ds[win_key])[idx]

    W_plot = W
    if max_pulses_to_plot is not None and len(W) > max_pulses_to_plot:
        take = np.random.choice(len(W), size=max_pulses_to_plot, replace=False)
        W_plot = W[take]

    L = W.shape[1]
    pre = int(pulse_ds.get("pre", L // 2))
    t = np.arange(L) - pre

    # ---- Plot layout ----
    fig = plt.figure(figsize=(16, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    ax_raw = fig.add_subplot(gs[0, 0])
    ax_sub = fig.add_subplot(gs[1, 0], sharex=ax_raw)
    ax_win = fig.add_subplot(gs[:, 1])

    # raw + baseline overlay
    ax_raw.plot(rec, linewidth=1)
    if show_baseline_overlay:
        ax_raw.plot(baseline, linewidth=1.5)
    ax_raw.set_title(f"Record {ri}: raw waveform" + (" (baseline overlay)" if show_baseline_overlay else ""))
    ax_raw.set_ylabel("ADC")
    ax_raw.grid(True, alpha=0.25)

    info = (
        f"pulses in record: {len(idx)}\n"
        f"markers plotted: {len(marks)}\n"
        f"marker_use: {marker_use}\n"
        f"windows shown: {len(W_plot)} / {len(W)}\n"
        f"baseline source: {'stored' if 'baseline_interpolations_per_record' in pulse_ds else 'fallback'}"
    )
    ax_raw.text(
        0.01, 0.98, info,
        transform=ax_raw.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", alpha=0.85),
        fontsize=10
    )

    # baseline-subtracted + markers
    ax_sub.plot(y, linewidth=1)
    if len(marks) > 0:
        ax_sub.scatter(marks, y[marks], marker="x", s=40)
    ax_sub.set_title("Baseline-subtracted trigger record + selected markers (✕)")
    ax_sub.set_xlabel("Sample index")
    ax_sub.set_ylabel("ADC (subtracted)")
    ax_sub.grid(True, alpha=0.25)

    # windows overlay
    ax_win.plot(t, W_plot.T, linewidth=0.8, alpha=0.35)
    if show_window_center_line:
        ax_win.axvline(0, linestyle="--", linewidth=1)
    ax_win.set_title(f"Extracted windows superimposed ({win_key})  n={len(W_plot)}")
    ax_win.set_xlabel("Samples relative to window center (0)")
    ax_win.set_ylabel("ADC (" + ("subtracted" if windows == "sub" else "raw") + ")")
    ax_win.grid(True, alpha=0.25)

    plt.show()
    return W, idx

# ---------- diagnostic  ----------
# to use when extracting with the "align" option
# recommend_window_post_from_pulse_ds gives the recommended value for post

def analyze_peak_offset_from_align(
    pulse_ds,
    *,
    ADCsat: float | None = None,
    sat_source: str = "peak_raw_localmax",
    bins: int = 200,
    show_percentiles=(90, 95, 99),
):
    """
    Analyze distribution of (peak_idx - align_idx).

    Returns:
        delta: array of peak offsets (samples)
    """

    if "align_idx" not in pulse_ds or "peak_idx" not in pulse_ds:
        raise KeyError("pulse_ds must contain 'align_idx' and 'peak_idx'")

    align = np.asarray(pulse_ds["align_idx"], dtype=np.int64)
    peak  = np.asarray(pulse_ds["peak_idx"], dtype=np.int64)

    mask = align >= 0  # valid CFD only

    if ADCsat is not None:
        if sat_source not in pulse_ds:
            raise KeyError(f"{sat_source} not in pulse_ds")
        raw = np.asarray(pulse_ds[sat_source], dtype=np.float64)
        mask &= (raw < ADCsat)

    delta = peak[mask] - align[mask]

    if delta.size == 0:
        print("No valid pulses after masking.")
        return delta

    # Plot histogram
    fig, ax = plt.subplots(figsize=(8,4), constrained_layout=True)
    ax.hist(delta, bins=bins)
    ax.set_xlabel("peak_idx - align_idx  (samples)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    # Show percentiles
    pvals = {}
    for p in show_percentiles:
        val = np.percentile(delta, p)
        pvals[p] = val
        ax.axvline(val, linestyle="--")
        ax.text(val, ax.get_ylim()[1]*0.9, f"{p}%", rotation=90)

    ax.set_title(f"Peak offset from CFD alignment  (N={len(delta)})")
    plt.show()

    print("Peak offset statistics:")
    print(f"  min: {delta.min():.1f}")
    print(f"  max: {delta.max():.1f}")
    print(f"  mean: {delta.mean():.1f}")
    for p, v in pvals.items():
        print(f"  {p}th percentile: {v:.1f}")

    return delta

def recommend_window_post_from_pulse_ds(
    pulse_ds,
    *,
    quantile: float = 0.995,      # 0.99, 0.995, 0.999
    margin: int = 10,             # extra samples of safety
    ADCsat: float | None = None,
    sat_source: str = "peak_raw_localmax_height",
    require_align: bool = True,   # only meaningful for CFD; True by default
    min_post: int = 0,
):
    """
    Recommend a 'post' (samples after alignment point) such that the peak is inside
    the window for ~quantile fraction of pulses.

    Uses delta = peak_idx - align_idx (CFD mode).

    Returns:
      post_recommended: int
      stats: dict with delta distribution stats
    """
    if "peak_idx" not in pulse_ds:
        raise KeyError("pulse_ds must contain 'peak_idx'")
    if "align_idx" not in pulse_ds:
        raise KeyError("pulse_ds must contain 'align_idx' (needed for CFD-based recommendation)")

    peak = np.asarray(pulse_ds["peak_idx"], dtype=np.int64)
    align = np.asarray(pulse_ds["align_idx"], dtype=np.int64)

    mask = np.ones(len(peak), dtype=bool)

    if require_align:
        mask &= (align >= 0)

    if ADCsat is not None:
        if sat_source not in pulse_ds:
            raise KeyError(f"sat_source='{sat_source}' not found in pulse_ds")
        mask &= (np.asarray(pulse_ds[sat_source], dtype=np.float64) < float(ADCsat))

    peak = peak[mask]
    align = align[mask]

    if peak.size == 0:
        return int(min_post), {"N": 0}

    delta = peak - align  # samples after CFD crossing where the peak occurs

    # sanity: negative deltas indicate a bug/odd cases
    neg_frac = float(np.mean(delta < 0))
    delta_pos = delta[delta >= 0]
    if delta_pos.size == 0:
        return int(min_post), {"N": int(len(delta)), "neg_frac": neg_frac}

    qv = float(np.quantile(delta_pos, quantile))
    post_rec = int(np.ceil(qv)) + int(margin)
    post_rec = max(int(min_post), post_rec)

    stats = {
        "N": int(len(delta)),
        "N_used": int(len(delta_pos)),
        "quantile": float(quantile),
        "q_value": qv,
        "margin": int(margin),
        "recommended_post": int(post_rec),
        "min": float(delta_pos.min()),
        "max": float(delta_pos.max()),
        "mean": float(delta_pos.mean()),
        "std": float(delta_pos.std()),
        "neg_frac": neg_frac,
        "p95": float(np.percentile(delta_pos, 95)),
        "p99": float(np.percentile(delta_pos, 99)),
    }
    return post_rec, stats




