# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 09:42:02 2026

@author: user, SANG JIN PARK
core.search.narrowing.py

"""
import numpy as np
import pandas as pd
from scipy.stats import norm
import sqlite3
import torch

# ============================================================
#           1. PASS 기반 통계 narrowing
# ============================================================

def compute_statistical_window(pass_rows: pd.DataFrame):
    if pass_rows is None or pass_rows.empty:
        return None

    window = {}

    if "AWG" in pass_rows.columns:
        q1, q3 = pass_rows["AWG"].quantile([0.25, 0.75])
        window["awg_range"] = (int(q1), int(q3))

    if "Parallels" in pass_rows.columns:
        q1, q3 = pass_rows["Parallels"].quantile([0.25, 0.75])
        window["par_range"] = (int(q1), int(q3))

    if "Turns_per_slot_side" in pass_rows.columns:
        q1, q3 = pass_rows["Turns_per_slot_side"].quantile([0.25, 0.75])
        window["turn_range"] = (int(q1), int(q3))

    return window


# ============================================================
#           2. Bayesian narrowing
# ============================================================
def compute_bayesian_window(pass_rows, base_awg, base_par, base_turn):
    """
    PASS 분포 기반 Bayesian narrowing
    """

    if pass_rows is None or pass_rows.empty:
        return None

    window = {}

    # ----------------------
    # AWG
    # ----------------------
    if "AWG" in pass_rows.columns:
        mu = pass_rows["AWG"].mean()
        sigma = pass_rows["AWG"].std() + 1e-6

        probs = [norm.pdf(a, mu, sigma) for a in base_awg]
        weighted = sorted(zip(base_awg, probs), key=lambda x: -x[1])

        top = [x[0] for x in weighted[:max(3, len(weighted)//2)]]
        window["awg_range"] = (min(top), max(top))

    # ----------------------
    # PAR
    # ----------------------
    if "Parallels" in pass_rows.columns:
        mu = pass_rows["Parallels"].mean()
        sigma = pass_rows["Parallels"].std() + 1e-6

        probs = [norm.pdf(p, mu, sigma) for p in base_par]
        weighted = sorted(zip(base_par, probs), key=lambda x: -x[1])

        top = [x[0] for x in weighted[:max(5, len(weighted)//2)]]
        window["par_range"] = (min(top), max(top))

    return window


# ============================================================
#           3. EMF 기반 par_min 안정화
# ============================================================
def compute_par_min_emf(Ke, rpm, Vdc, V_margin=0.9):
    """
    EMF 기반 최소 병렬 수 계산
    """
    omega = 2 * np.pi * rpm / 60
    emf = Ke * omega

    V_limit = Vdc * V_margin

    if emf < V_limit:
        return 1

    ratio = emf / V_limit
    return int(np.ceil(ratio))


# ============================================================
#       4. Thermal narrowing
# ============================================================

def estimate_winding_temp(I_rms_t, R_all, ambient=40.0):
    loss = 3 * I_rms_t**2 * R_all
    deltaT = loss * 0.02
    return ambient + deltaT

def failure_probability(temp_C, J_A_per_mm2):
    """
    ESC 권선 열화/절연 수명 기반 고장 확률 근사 모델
    """
    # Arrhenius + J stress 모델 예시
    thermal_stress = np.exp((temp_C - 120.0) / 25.0)
    current_stress = (J_A_per_mm2 / 8.0) ** 2

    prob = 1.0 - np.exp(-0.001 * thermal_stress * current_stress)
    return prob

# ============================================================
#            5. Self-learning DB
# ============================================================

def save_pass_patterns(df, db_path="bflow_learning.db"):
    import sqlite3
    if df is None or df.empty:
        return

    conn = sqlite3.connect(db_path)      # data/esc_memory.db
    df[["rpm","AWG","Parallels","Turns_per_slot_side"]] \
        .to_sql("pass_patterns", conn, if_exists="append", index=False)
    conn.close()

def load_learning_window(db_path="bflow_learning.db"):
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM pass_patterns", conn)
        conn.close()
    except Exception:
        return None

    if df.empty:
        return None

    return compute_statistical_window(df)


# ============================================================
#           6. Window 적용
# ============================================================
def compute_esc_optimal_window(pass_rows: "pd.DataFrame",
                               min_awg_span: int = 1,
                               min_par_span: int = 3,
                               min_turn_span: int = 5):
    """
    PASS-1 결과 기반 narrowing window 계산

    반환:
        dict or None
        {
            "awg_range": (lo, hi),
            "par_range": (lo, hi),
            "turn_range": (lo, hi),
        }
    """

    if pass_rows is None or pass_rows.empty:
        return None

    window = {}

    # =============================
    #           AWG
    # =============================
    if "AWG" in pass_rows.columns:
        awg_vals = pass_rows["AWG"].astype(int)
        q1, q3 = np.percentile(awg_vals, [25, 75])
        lo = int(np.floor(q1))
        hi = int(np.ceil(q3))

        if hi - lo < min_awg_span:
            mid = int(np.median(awg_vals))
            lo = mid - min_awg_span
            hi = mid + min_awg_span

        window["awg_range"] = (lo, hi)

    # =============================
    #           PARALLELS
    # =============================
    if "Parallels" in pass_rows.columns:
        par_vals = pass_rows["Parallels"].astype(int)
        q1, q3 = np.percentile(par_vals, [25, 75])
        lo = int(np.floor(q1))
        hi = int(np.ceil(q3))

        if hi - lo < min_par_span:
            mid = int(np.median(par_vals))
            lo = mid - min_par_span
            hi = mid + min_par_span

        window["par_range"] = (max(1, lo), hi)

    # =============================
    #           TURN
    # =============================
    if "Turns_per_slot_side" in pass_rows.columns:
        turn_vals = pass_rows["Turns_per_slot_side"].astype(int)
        q1, q3 = np.percentile(turn_vals, [25, 75])
        lo = int(np.floor(q1))
        hi = int(np.ceil(q3))

        if hi - lo < min_turn_span:
            mid = int(np.median(turn_vals))
            lo = mid - min_turn_span
            hi = mid + min_turn_span

        window["turn_range"] = (max(1, lo), hi)

    return window

def apply_window_to_globals(window, engine_module):
    """
    compute_esc_optimal_window 결과를 실제 탐색 범위에 반영
    """

#    global awg_candidates, par_candidates, turn_candidates_base

    if window is None:
        return False
    
    # =============================
    #           AWG
    # =============================
    if "awg_range" in window:
        lo, hi = window["awg_range"]
        engine_module.awg_candidates = [
            a for a in engine_module.awg_candidates if lo <= a <= hi
        ]
    # =============================
    #           PARALLELS
    # =============================
    if "par_range" in window:
        lo, hi = window["par_range"]
        engine_module.par_candidates = [
            p for p in engine_module.par_candidates if lo <= p <= hi
        ]
    # =============================
    #           TURN
    # =============================
    if "turn_range" in window:
        lo, hi = window["turn_range"]
        engine_module.turn_candidates_base = [
            t for t in engine_module.turn_candidates_base if lo <= t <= hi
        ]
    return True

def compute_dynamic_tile_size(base_size=2048):
    if not torch.cuda.is_available():
        return base_size

    free_mem = torch.cuda.mem_get_info()[0]  # bytes
    gb = free_mem / (1024**3)

    if gb > 12:
        return base_size * 4
    elif gb > 6:
        return base_size * 2
    elif gb > 3:
        return base_size
    else:
        return base_size // 2

def compute_auto_search_window(pass_rows):
#    stats = pass_rows.describe()

    if pass_rows is None or pass_rows.empty:
        return None
    return {
        "awg_range": (int(pass_rows["AWG"].min()), int(pass_rows["AWG"].max())),
        "par_range": (int(pass_rows["Parallels"].quantile(0.1)), int(pass_rows["Parallels"].quantile(0.9))),
        "turn_range": (int(pass_rows["Turns_per_slot_side"].quantile(0.1)), int(pass_rows["Turns_per_slot_side"].quantile(0.9))),
    }

def ensure_minimum_search_space(engine_module, par_min_default: int, par_span: int = 12, turn_span: int = 20):

    if len(engine_module.par_candidates) < 8:
        engine_module.par_candidates = list(range(par_min_default, par_min_default + par_span))
        
    if len(engine_module.turn_candidates_base) < 15:
        lo = int(engine_module.NSLOT_USER_RANGE[0])
        engine_module.turn_candidates_base = list(range(lo, lo + turn_span))

def bayesian_narrow(pass_rows, prior_window):
    """실제 구현체 추가: 분포의 중심을 pass_rows의 평균으로 이동"""
    if pass_rows is None or pass_rows.empty:
        return prior_window
    
    # 예시: AWG의 새로운 중심점 계산
    new_window = prior_window.copy()
    if "AWG" in pass_rows.columns:
        new_window["awg_center"] = pass_rows["AWG"].mean()
    return new_window
#    raise NotImplementedError("bayesian_narrow() is not implemented. Use bayesian_update() or compute_bayesian_window().")

def bayesian_update(prior_window, pass_rows):
    posterior = compute_esc_optimal_window(pass_rows)
    return posterior

def auto_tile_limit(search_space_size):
    if search_space_size > 5_000_000:
        return 1_000_000
    if search_space_size > 1_000_000:
        return 500_000
    return None

def auto_tile_size(device_props, search_space_size):
    sm_count = device_props.multi_processor_count
    base = sm_count * 4

    if search_space_size > 5_000_000:
        return base * 2
    return base


def narrow_candidates_by_pass(dist, candidates, keep_ratio=0.5):
    if not dist:
        return candidates

    total = sum(dist.values())
    sorted_items = sorted(dist.items(), key=lambda x: -x[1])

    keep = []
    acc = 0
    for k, v in sorted_items:
        keep.append(k)
        acc += v
        if acc / total >= keep_ratio:
            break

    return [c for c in candidates if c in keep]

def cluster_pass_regions(pass_rows):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2)
    labels = kmeans.fit_predict(pass_rows[["AWG","Parallels","Turns_per_slot_side"]])
    return kmeans.cluster_centers_
