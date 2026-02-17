# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 14:15:50 2025

@author: USER, SANG JIN PARK
"""
# NOTE: avoid importing core.engine at module import time (prevents circular import).
#       Import inside functions when needed.

import torch
from typing import TYPE_CHECKING
import os
import sys
import time
import math
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from configs.config import ENABLE_PROFILING, PROF, AWG_TABLE

# ======================== 기본 환경 =========================================
# 1. 환경 변수 먼저 선언
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
ITYPE  = torch.int32

# 2. Config 정의 및 생성
@dataclass
class RunConfig:
    device: torch.device
    dtype: torch.dtype
    itype: torch.int32
    # 추가적인 파라미터들도 이곳에 통합 가능
# 실행 시점에 생성
cfg = RunConfig(device=DEVICE, dtype=DTYPE, itype=ITYPE)

# 1. 유틸리티 함수
def T(val, config):
    import torch
    import numpy as np

    dev = getattr(config, "device", None)
    dt  = getattr(config, "dtype", None)

    # 1) 이미 Tensor면: 새로 만들지 말고 .to()만
    if isinstance(val, torch.Tensor):
        # 이미 목적 device/dtype이면 그대로 반환(루프 성능/동기화 방지)
        if (dev is None or val.device == dev) and (dt is None or val.dtype == dt):
            return val
        return val.to(device=dev, dtype=dt)

    # 2) numpy면: from_numpy가 빠름 
    if isinstance(val, np.ndarray):
        t = torch.from_numpy(val)
        return t.to(device=dev, dtype=dt)

    # 3) pandas Series 등은 ndarray로
    if hasattr(val, "to_numpy"):
        t = torch.from_numpy(val.to_numpy())
        return t.to(device=dev, dtype=dt)

    # 4) 파이썬 scalar
    if isinstance(val, (int, float, bool)):
        return torch.tensor(val, device=dev, dtype=dt)

    # 5) list/tuple: 너무 큰 경우 방어(진단용)
    if isinstance(val, (list, tuple)) and len(val) > 100000:
        raise ValueError(f"T(): suspicious huge list/tuple len={len(val)} (likely bug)")

    return torch.tensor(val, device=config.device, dtype=config.dtype)

# =================================================================
# [UTIL] row에서 값 꺼내기 방탄 (dict/Series/itertuples 모두 대응)
# =================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Lazy imports to avoid circular dependencies (utils <-> engine)
# -------------------------------------------------------------------
def _lazy_engine():
    from core import engine as _engine  # local import
    return _engine


def _row_get(row, key, default=None):
    """
    row가 dict / pandas.Series / namedtuple(itertuples) 어떤 형태든 안전 접근.
    """
    try:
        if row is None:
            return default
        if isinstance(row, dict): 
            return row[key]
        return getattr(row, key)
    
    except (KeyError, AttributeError) as e:
        # [수정] e를 로깅에 포함하여 어떤 에러가 발생했는지 명시
        logger.debug(f"Key '{key}' retrieval failed ({type(e).__name__}: {e}). Using default: {default}")
        return default
        
    except Exception as e:
        # [수정] 예상치 못한 에러 발생 시 에러 타입과 상세 메시지를 함께 기록
        logger.error(f"Unexpected error in _row_get: [{type(e).__name__}] {e}")
        raise

def _row_get_first(row, keys, default=None):
    for k in keys:
        v = _row_get(row, k, None)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return default

# ======================== 유틸 함수 ========================================
def awg_area_mm2(awg: int) -> float:
    v = AWG_TABLE[int(awg)]
    if isinstance(v, dict):
        return float(v.get("area", 0.0))
    return float(v)
# ======================== GPU 유틸 ========================================
def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def gpu_mem_info_gb():
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        return free_b/1e9, total_b/1e9
    return None, None

def print_gpu_banner():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        free_gb, total_gb = gpu_mem_info_gb()
        print(f"[GPU] Device: {name} | CC {cc[0]}.{cc[1]}")
        if free_gb is not None:
            print(f"[GPU] Memory: free={free_gb:.2f} GB / total={total_gb:.2f} GB")
    else:
        print("[GPU] CUDA not available (running on CPU)")

# Warm-up (프로파일링 ON일 때만 GPU 시간 누적)
if ENABLE_PROFILING:
    print_gpu_banner()
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    w0 = time.perf_counter()
    a = torch.rand(1024, 1024, device=DEVICE)
    b = torch.rand(1024, 1024, device=DEVICE)
    if DEVICE.type == "cuda":
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
    _ = a @ b
    if DEVICE.type == "cuda":
        e1.record()
        torch.cuda.synchronize()
        PROF["gpu_ms_mask"] += float(e0.elapsed_time(e1))
        print(f"[GPU] Warm-up: {e0.elapsed_time(e1)/1000.0:.3f}s")
    else:
        print(f"[CPU] Warm-up: {time.perf_counter()-w0:.3f}s")
    PROF["start_wall"] = time.perf_counter()

# ---- geometry 리스트류 방탄: 사용자가 실수로 float 하나만 넣어도 for-loop가 안 죽게 ----
def _ensure_iterable(x):
    # list/tuple/range/ndarray면 그대로
    if isinstance(x, (list, tuple, range)):
        return x
    try:
        import numpy as _np
        if isinstance(x, _np.ndarray):
            return x
    except Exception:
        pass
    # scalar면 (x,)로 감싸기
    return (x,)

def topk_pretrim_df(
    df: "pd.DataFrame",
    topk: int | None,
    cols: tuple[str, ...] = ("Pcu_W", "J_A_per_mm2"),
) -> "pd.DataFrame":
    """
    df를 cols 기준 오름차순으로 정렬해 상위 topk행만 남긴다.
    - nsmallest가 실패(컬럼 타입/NaN/혼합 등)하면 sort_values + head로 fallback.
    - topk가 None이거나 df가 비어있으면 그대로 반환.
    """
    if topk is None or df is None or len(df) == 0:
        return df

    try:
        return df.nsmallest(int(topk), list(cols))
    except Exception:
        return (
            df.sort_values(list(cols), ascending=[True] * len(cols), kind="mergesort")
              .head(int(topk))
        )

def _is_placeholder_df(df: pd.DataFrame) -> bool:
    return isinstance(df, pd.DataFrame) and (list(df.columns) == ["Note"])
  
def _row_to_min_dict(row):
    """
    _lazy_engine().suggest_margin_fix()가 필요한 키만 최소로 dict화.
    iterrows()/itertuples()/dict 입력 혼재에도 안전.
    """
    # suggest_margin_fix 내부는 _row_get을 쓰도록 패치되어 있으므로
    # 여기서는 dict 변환을 강제하지 않아도 되지만,
    # Series->dict 변환 비용을 줄이기 위해 필요한 것만 추출한다.
    keys = [
        "rpm","Vavail_LL_rms","Ke_scale","Ld_mH","Lq_mH",
        "Kt_rms","T_Nm","Turns_per_slot_side","Parallels",
        "A_wire_mm2","AWG","slot_area_mm2","slot_fill_limit","MLT_mm",
        "J_max_A_per_mm2",
    ]
    out = {}
    for k in keys:
        try:
           # 1) dict 우선
           if isinstance(row, dict):
               out[k] = row.get(k, None)
           # 2) pandas Series/DataFrame row: row[k] 가능
           elif hasattr(row, "get") and hasattr(row, "__getitem__") and not hasattr(row, "_fields"):
               # Series는 보통 _fields가 없고, get()이 있다
               out[k] = row.get(k, None)
           # 3) itertuples() namedtuple: getattr가 정답
           elif hasattr(row, "_fields"):
               out[k] = getattr(row, k)
           # 4) 기타 객체: getattr 시도
           else:
               out[k] = getattr(row, k)
        except Exception:
            # 누락은 suggest_margin_fix에서 fallback 처리
            pass
    return out
 
def _iter_rows_fast(df: pd.DataFrame, k: int):
    """
    itertuples 기반(빠름). df index를 row_index로 유지.
    """
    # index를 별도 컬럼으로 붙여서 itertuples에 포함
    d = df.head(k).copy()
    d = d.reset_index(drop=False).rename(columns={"index": "row_index"})
    # name=None이면 일반 튜플이라 접근이 불편 → namedtuple 유지
    return d.itertuples(index=False)

# ------------------------------------------------------------
# (안전장치) 정의
# ------------------------------------------------------------

def run_fix_batch(df, k=50, target=0.10):

   if df is None or not isinstance(df, pd.DataFrame):
       return pd.DataFrame([{"status":"no_fix","reason":"df_not_dataframe"}])
   if df.empty or _is_placeholder_df(df):
       return pd.DataFrame([{"status":"no_fix","reason":"empty_or_placeholder"}])
    
   # k/target 방어
   try:
       if k is None:
           k = len(df)
       k = int(k)
   except Exception:
       k = 50
   if k <= 0:
       return pd.DataFrame([{"status":"no_fix","reason":"k_nonpositive"}])
   try:
       target = float(target)
   except Exception:
       target = 0.10

   # suggest_margin_fix가 최소한으로 기대하는 키들
   # (여기에서 강제 포함시키면 _row_to_min_dict가 너무 적게 뽑아도 안전)
   REQUIRED_KEYS = (
       "rpm", "rpm_case", "rpm_check",
       "Vavail_LL_rms", "V_LL_max_V",
       "Ke_scale", "Ld_mH", "Lq_mH",
       "Kt_rms", "T_Nm", "T_check_Nm",
       "Turns_per_slot_side", "Parallels", "AWG",
       "slot_area_mm2", "slot_fill_limit", "MLT_mm",
       "J_max_A_per_mm2",
       "A_wire_mm2",
   )

   out = []
   # ✅ iterrows() 대신 itertuples() 사용 (훨씬 빠름)
   for j, r in enumerate(_iter_rows_fast(df, k)):
       # row index 확보 (row_index -> Index -> loop index)
       row_index = getattr(r, "row_index", None)
       if row_index is None:
           row_index = getattr(r, "Index", None)
       if row_index is None:
           row_index = j
       # CPU overhead 최소화: 최소 dict 기반 유지
       d = _row_to_min_dict(r)
       if not isinstance(d, dict):
           d = {}
       # REQUIRED_KEYS를 namedtuple에서 가능한 한 채워 넣기
       # (없으면 d에 없는 상태로 두고 suggest_margin_fix에서 missing 처리)
       for key in REQUIRED_KEYS:
           if key not in d:
               try:
                   v = getattr(r, key)
               except Exception:
                   v = None
               if v is not None:
                   d[key] = v
       # 배치 전체가 죽지 않도록 방어
       try:
           res = _lazy_engine().suggest_margin_fix(d, target_margin_pct=target)
       except Exception as e:
           res = dict(status="error", reason=f"suggest_fix_exception:{type(e).__name__}:{e}")
         
       # res가 dict 또는 None일 수 있으므로 방어
       if res is None:
           res = dict(status="no_fix", reason="None_returned")
       res["row_index"] = row_index
       out.append(res)
   return pd.DataFrame(out)

# 유용한 파생(전/후 비교치) 계산: 가능할 때만 계산
def _safe_diff(a, b):
    try:
        if a is None or b is None: return np.nan
        return b - a
    except Exception:
        return np.nan

def _safe_pct(a, b):
    try:
        if a is None or b is None: return np.nan
        if a == 0: return np.nan
        return 100.0*(b-a)/a
    except Exception:
        return np.nan

def save_rank_and_fixes_workbook(df_ranked: pd.DataFrame,
                                 K: int = 50,
                                 target_margin: float = 0.10,
                                 out_path: str | None = None,
                                 with_worstcheck: bool = True,
                                 worst_cfg: dict | None = None):
    """
    df_ranked 상위 K개에 대해 제안 실행→전후비교→엑셀 저장.
    (기존 do_profile_summary_and_save()와 별개로 추가 결과 파일을 생성)
    """
    from core.physics import worstcase_margin_ok
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"./Ranked_with_Fixes_{ts}.xlsx"

    # 1) 제안 실행
    fix_df = run_fix_batch(df_ranked, k=K, target=target_margin)

    # 2) 병합(전/후 비교)
    orig_sel = df_ranked.reset_index().rename(columns={"index":"row_index"})
    orig_sel = orig_sel[orig_sel["row_index"].isin(fix_df["row_index"])].copy()
    merged = pd.merge(orig_sel, fix_df, on="row_index", how="left", suffixes=("", "_fix"))

    # 비교 파생
    merged["Turns_per_slot_sideΔ"] = merged.apply(
        lambda r: _safe_diff(r.get("Turns_per_slot_side"), r.get("Turns_per_slot_side_new")), axis=1)
    merged["ParallelsΔ"] = merged.apply(
        lambda r: _safe_diff(r.get("Parallels"), r.get("Parallels_new")), axis=1)

    merged["V_LL_req_V_new"] = merged.get("V_LL_req_V_new", np.nan)
    merged["V_LL_margin_V_new"] = merged.get("V_LL_margin_V_new", np.nan)
    merged["V_LL_margin_pct_new"] = merged.get("V_LL_margin_pct_new", np.nan)

    merged["VreqΔ(V)"] = merged.apply(
        lambda r: _safe_diff(r.get("Vreq_LL_rms"), r.get("V_LL_req_V_new")), axis=1)
    merged["VmarginΔ(V)"] = merged.apply(
        lambda r: _safe_diff(r.get("V_margin"), r.get("V_LL_margin_V_new")), axis=1)
    merged["VmarginΔ(%)"] = merged.apply(
        lambda r: _safe_diff(r.get("V_margin_pct"), r.get("V_LL_margin_pct_new")), axis=1)

    merged["R_phase_ohm_new"] = merged.get("R_phase_ohm_new", np.nan)
    merged["Slot_fill_ratio_new"] = merged.get("Slot_fill_ratio_new", np.nan)
    merged["J_A_per_mm2_new"] = merged.get("J_A_per_mm2_new", np.nan)

    merged["R_phaseΔ(ohm)"] = merged.apply(
        lambda r: _safe_diff(r.get("R_phase_ohm"), r.get("R_phase_ohm_new")), axis=1)
    merged["FillΔ"] = merged.apply(
        lambda r: _safe_diff(r.get("Slot_fill_ratio"), r.get("Slot_fill_ratio_new")), axis=1)
    merged["JΔ(A/mm2)"] = merged.apply(
        lambda r: _safe_diff(r.get("J_A_per_mm2"), r.get("J_A_per_mm2_new")), axis=1)

    if "Current_rms_A" in merged.columns:
        merged["Pcu_W_new_est"] = 3.0 * (merged["Current_rms_A"]**2) * merged["R_phase_ohm_new"]
        merged["PcuΔ(W)"] = merged["Pcu_W_new_est"] - merged.get("Pcu_W", np.nan)
        merged["PcuΔ(%)"] = _safe_pct(merged.get("Pcu_W", np.nan), merged["Pcu_W_new_est"])

    # 합격 플래그(새 상태 기준)
    def _pass_new(r, margin_target=target_margin):
        try:
            ok_v = (r.get("V_LL_margin_pct_new", np.nan) is not None) and (r["V_LL_margin_pct_new"] >= margin_target)
            ok_J = ("J_max_A_per_mm2" in r) and (r["J_A_per_mm2_new"] <= r["J_max_A_per_mm2"])
            ok_fill = ("slot_fill_limit" in r) and (r["Slot_fill_ratio_new"] <= r["slot_fill_limit"])
            return bool(ok_v and ok_J and ok_fill)
        except Exception:
            return False

    merged["fix_pass"] = merged.apply(_pass_new, axis=1)
    preview_cols = [
        "row_index","status","reason",
        "rpm","T_Nm","Vdc","m_max","Ke_scale","Ld_mH","Lq_mH","Kt_rms",
        "AWG","Parallels","Turns_per_slot_side","slot_area_mm2","slot_fill_limit","J_max_A_per_mm2",
        "Vavail_LL_rms","Vreq_LL_rms","V_margin","V_margin_pct","R_phase_ohm","Pcu_W","Current_rms_A",
        "Parallels_new","Turns_per_slot_side_new",
        "V_LL_req_V_new","V_LL_margin_V_new","V_LL_margin_pct_new",
        "R_phase_ohm_new","Slot_fill_ratio_new","J_A_per_mm2_new",
        "V_LL_max_V","V_LL_req_V","V_LL_margin_V","V_LL_margin_pct","P_cu_W",
        "V_LL_max_V_new", "P_cu_W_new_est",
        "ParallelsΔ","Turns_per_slot_sideΔ","VreqΔ(V)","VmarginΔ(V)","VmarginΔ(%)","R_phaseΔ(ohm)","FillΔ","JΔ(A/mm2)",
        "Pcu_W_new_est","PcuΔ(W)","PcuΔ(%)","fix_pass"
    ]
    preview = merged.reindex(columns=[c for c in preview_cols if c in merged.columns]).copy()
    preview_ok = preview[preview["fix_pass"]==True].copy()

    # (옵션) Worst-case 동시합격 플래그
    if with_worstcheck:
        wc = worst_cfg if worst_cfg else dict(Vdc=325.0, m_max=0.925, Ke_scale=1.05, R_scale=1.40, L_scale=0.85)
        merged["wc_pass"] = merged.apply(lambda r: _lazy_engine().worstcase_margin_ok(r, wc, target_pct=0.05), axis=1)
        preview["wc_pass"] = merged["wc_pass"]
        preview_ok["wc_pass"] = preview_ok.apply(lambda r: _lazy_engine().worstcase_margin_ok(r, wc, target_pct=0.05), axis=1)

    # 저장
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        # 참고: df_ranked 전체가 필요하면 외부에서 전달/저장하세요.
        preview.to_excel(w, sheet_name="merge_preview", index=False)
        preview_ok.to_excel(w, sheet_name="only_ok_fixes", index=False)
        fix_df.to_excel(w, sheet_name="fix_suggestions", index=False)
    print(f"[SAVE] Ranked-with-fixes workbook: {os.path.abspath(out_path)}")

def run_postprocess_rank_fix_and_worstcase(
    df_candidates: pd.DataFrame,
    topk_rank: int = 100,
    target_margin: float = 0.10,
    worst_cfg: dict | None = None,
    out_rank_path: str | None = None,
    out_fix_path: str | None = None,
):
    """
    1) df_candidates → _lazy_engine().make_rank()로 정렬
    2) 상위 topk_rank만 별도 Excel 저장
    3) auto-fix(suggest_margin_fix) + worst-case check 적용
    4) fix/전후비교 + worst-case 결과를 별도 Excel에 저장
    """

    if worst_cfg is None:
        worst_cfg = dict(Vdc=325.0, m_max=0.925,
                         Ke_scale=1.05, R_scale=1.40, L_scale=0.85)

    # ── 1) 랭킹 및 Top-K ──────────────────────────────────────────────
    if df_candidates is None or not isinstance(df_candidates, pd.DataFrame) or df_candidates.empty or _is_placeholder_df(df_candidates):
        print("[POST][WARN] df_candidates empty/placeholder -> skipping postprocess.")
        return df_candidates, pd.DataFrame()
    
    df_ranked = _lazy_engine().make_rank(df_candidates)
    df_ranked_top = df_ranked.head(topk_rank).copy()

    if out_rank_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_rank_path = f"./RevXX_RankedTop{topk_rank}_{ts}.xlsx"

    os.makedirs(os.path.dirname(os.path.abspath(out_rank_path)) or ".", exist_ok=True)
    df_ranked_top.to_excel(out_rank_path, sheet_name="top_ranked", index=False)
    print(f"[POST] Ranked Top-{topk_rank} saved to {out_rank_path}")

    # ── 2) auto-fix 실행 ───────────────────────────────────────────────
    fix_df = run_fix_batch(df_ranked_top, k=topk_rank, target=target_margin)

    # ── 3) 전후 비교 + worst-case 를 포함해 통합 Workbook 생성 ───────
    if out_fix_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_fix_path = f"./RevXX_RankedWithFixes_{ts}.xlsx"

    save_rank_and_fixes_workbook(
        df_ranked_top,
        K=topk_rank,
        target_margin=target_margin,
        out_path=out_fix_path,
        with_worstcheck=True,
        worst_cfg=worst_cfg
    )
    print(f"[POST] Ranked-with-fixes (with worst-case) saved to {out_fix_path}")

    return df_ranked_top, fix_df

def init_planned_combos_denominator():
    """사전 분모 계산을 실행하고 PROF['combos_planned']에 세팅."""
    total = _lazy_engine().estimate_total_combos_planned()
    PROF["combos_planned"] = int(total)
    return total
# ================== EOF ====================