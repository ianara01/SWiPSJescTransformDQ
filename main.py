# -*- coding: utf-8 -*-
# main.py (helper)
"""
Created on Fri Feb  6 10:24:51 2026

@author: USER, SANG JIN PARK

Unified CLI entrypoint for CoilWindingOptimzation / SWiPSJescTransformDQ

Modes:
  - full      : run_full_pipeline() (classic)
  - adaptive  : setup_rpm_adaptive_envelope_and_run() (recommended)
  - bflow     : run_bflow_full_two_pass() (2-pass filter + rank)
  - femm_gen  : generate FEMM .fem from FW-safe designs (from a saved df_pass2)
  - femm_extract : extract Ld/Lq from FEMM .fem directory -> write JSON/CSV
  - feedback  : apply extracted Ld/Lq feedback to df_pass2 and (optionally) register cache

Design goals:
  - Exactly ONE mode runs per invocation (--mode).
  - Output paths are ALWAYS derived from build_output_paths().
  - Engine runners MUST yield (df_pass1, df_pass2). If upstream still returns None,
    we fallback to reading engine globals (df_pass1/df_pass2) and finally error out.
  - FEMM pipeline is decoupled: you can run sweep first, then femm_gen, then femm_extract, then feedback.

Project layout assumption:
  /configs/config.py
  /core/engine.py
  /core/physics.py (optional for FEMM feedback etc.)
  /core/progress.py
    import 시점에 부작용(프린트/torch warmup/전역 카운터 초기화)을 없앰
    진행률/프로파일 집계용 전역 dict(기존 코드 호환) + 명시적 init 1회 호출로 통일
    engine/utils/main 어디서든 가볍게 import 가능 (순환 import 최소화)
"""

import json
import os
import sys
import time
import threading
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

#import configs.config as C
import core.engine as eng
import core.physics as phys

import configs.config as cfg

from core.engine import autotune_par_candidates_for_revision
from core.engine import run_mode_bflow_pass1, run_mode_bflow_pass2

from winding_spec import WindingConnSpec, lock_coils_per_phase_global
from winding_table import build_winding_table_24s4p, generate_fw_safe_winding_tables

from utils.femm_pipeline import (
    batch_extract_ldlq_from_femm,
    parse_key_from_fem_filename
)
from utils.femm_builder import run_femm_generation, build_fem_from_winding, extract_results_with_dq_transform, generate_design_candidates

from core.winding_table import generate_fw_safe_winding_tables, generate_femm_files_from_windings

import torch
import configs.config as cfg
from core.physics import (
    apply_envelope_for_case, 
    rebuild_awg_par_tensors, 
    kw_rpm_to_torque_nm
)

from core.progress import init_progress

from core.search.rl_agent import DQNAgent, calculate_fill_factor
from core.search.surrogate import DesignSurrogate, run_smart_narrowing, train_surrogate, predict_margin

# Optional (only if you have FEMM installed/usable on the machine):
try:
    from utils.femm_ldlq import I_test as FEMM_I_TEST_DEFAULT
except Exception:
    FEMM_I_TEST_DEFAULT = 0.0

#hp = eng.load_hp_from_yaml_and_globals()
#cases = eng.build_power_torque_cases(hp)
#df_pass1, df_pass2 = eng.setup_rpm_adaptive_envelope_and_run(
#    cases=cases,
#    par_hard_max=cfg.PAR_HARD_MAX,
#)

# =============================================================================
#           Interactive mode chooser
# =============================================================================
def _input_with_timeout(prompt: str, timeout_sec: int | None) -> str | None:
    """
    timeout_sec 안에 입력이 없으면 None 반환.
    timeout_sec=None 이면 무한 대기.
    """
    if timeout_sec is None:
        return input(prompt)

    buf = {"val": None}

    def _reader():
        try:
            buf["val"] = input(prompt)
        except Exception:
            buf["val"] = None

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    t.join(timeout_sec)
    return buf["val"]


def choose_mode_interactively(
    *,
    default_mode: str = "adaptive",
    timeout_sec: int | None = None,
    allow_extended: bool = True,
) -> str:
    """
    - 숫자 1/2/3 입력 방식 + 문자열 입력(full/adaptive/bflow)
    - Enter만 치면 default 선택
    - timeout_sec 내 입력 없으면 default 선택
    """
    base_modes = ["full", "adaptive", "bflow"]
    ext_modes  = ["femm_gen", "femm_extract", "feedback"] if allow_extended else []
    all_modes  = base_modes + ext_modes

    # 메뉴 출력
    print("\n=========================================")
    print(" CoilWindingOptimzation  MENU ")
    print("=========================================")
    print("Choose execution mode:")
    print("  1) full       (전체 파이프라인)")
    print("  2) adaptive   (추천: rpm별 설계공간 줄여 빠르게 PASS 찾기)")
    print("  3) bflow      (2-pass 후보 수렴 - 자동 튜닝 탐색)")
    if allow_extended:
        print("  4) femm_gen       (FW-safe winding → .fem 생성)")
        print("  5) femm_extract   (.fem → Ld/Lq 추출)")
        print("  6) feedback       (Ld/Lq feedback 적용)")
    print("------------------------------")
    if timeout_sec is not None and timeout_sec > 0:
        print(f"Enter for default='{default_mode}'  |  auto-select in {timeout_sec}s")
    else:
        print(f"Enter for default='{default_mode}'")
    print("==============================")

    # 매핑(숫자 입력)
    num_map = {
        "1": "full",
        "2": "adaptive",
        "3": "bflow",
        "4": "aibflow",
        "5": "rl_search",
    }
    if allow_extended:
        num_map.update({
            "6": "femm_gen",
            "7": "femm_extract",
            "8": "feedback",
        })

    while True:
        raw = _input_with_timeout("Select [1/2/3/4/5 or name]: ", timeout_sec)
        if raw is None:
            # timeout
            print(f"\n[MENU] No input. Using default: {default_mode}")
            return default_mode

        s = raw.strip().lower()
        if s == "":
            print(f"[MENU] Using default: {default_mode}")
            return default_mode

        # 숫자 선택
        if s in num_map:
            chosen = num_map[s]
            print(f"[MENU] Selected: {chosen}")
            return chosen

        # 문자열 선택
        if s in all_modes:
            print(f"[MENU] Selected: {s}")
            return s

        print(f"[MENU] Invalid: '{raw}'. Try again.")

# ============================================================
#              CLI (Command Line Interface)
# ============================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("CoilWindingOptimzation CLI")

    ap.add_argument(
        "--mode",
        choices=["full", "adaptive", "bflow", "aibflow", "rl_search", "femm_gen", "femm_extract", "feedback"],
#        required=True,
        help="Execution mode: choose exactly one pipeline step."
    )
    # (선택) 메뉴 입력 타임아웃(초). 0/미지정이면 무한대기.
    ap.add_argument("--menu_timeout", type=int, default=0, help="Interactive menu timeout seconds (0=wait forever).")

    # common outputs
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--stem", default="ESCwinding_candidates_Torch")
    ap.add_argument("--no_excel", action="store_true")
    ap.add_argument("--no_parquet", action="store_true")
    ap.add_argument("--no_csvgz", action="store_true")

    # sweep/bflow common knobs
    ap.add_argument("--motor_type", default="IPM")
    ap.add_argument("--min_margin_pct", type=float, default=0.005)
    ap.add_argument("--passrows_topk", type=int, default=1000)

    # bflow inputs
    ap.add_argument("--rpm_list", type=int, nargs="*", default=[600, 1800, 3600])
    ap.add_argument("--p_list", type=float, nargs="*", default=[1.0, 2.0, 3.0, 4.0, 5.0, 7.0])

    # femm_gen / feedback inputs
    ap.add_argument("--df_in", default=None, help="Input df file path (.xlsx/.parquet/.csv.gz) for femm_gen/feedback.")

    # femm_gen knobs
    ap.add_argument("--femm_dir", default=None, help="Directory containing .fem files (default: <out_dir>/femm)")
    ap.add_argument("--I_test", type=float, default=None, help="Current for Ld/Lq extraction (default: femm_ldlq.I_test)")
    ap.add_argument("--fw_margin_min", type=float, default=0.05)
    
    # femm_extract inputs
    ap.add_argument("--ldlq_out", default=None, help="Output json path for LdLq_DB.")

    # feedback inputs
    ap.add_argument("--ldlq_in", default=None, help="Input LdLq_DB json path.")
    ap.add_argument("--with_fixes", action="store_true")
    ap.add_argument("--fixes_topk", type=int, default=50)
    ap.add_argument("--fixes_target_margin", type=float, default=0.10)
    
    # feedback knobs
    ap.add_argument("--df_pass2", default=None,
                    help="Path to df_pass2 excel/parquet/csv to feed femm_gen/feedback (default: use latest saved by this run if available).")
    ap.add_argument("--ldlq_json", default=None, help="Path to saved ldlq json (from femm_extract).")
    ap.add_argument("--feedback_strategy", default="both", choices=["register", "overwrite", "both"])

    return ap.parse_args()

# ================================================================
#       Output path builder SSOT (single source of truth)
# ================================================================
def build_output_paths(out_dir: str = "results", stem: str = "ESCwinding_candidates") -> Dict[str, str]:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{stem}_{ts}"

    return {
        "OUT_DIR":  str(out_root),
        "OUT_XLSX": str(out_root / f"{base}.xlsx"),
        "OUT_PARQ": str(out_root / f"{base}.parquet"),
        "OUT_CSVGZ": str(out_root / f"{base}.csv.gz"),
        "OUT_JSON": str(out_root / f"{base}.json"),
        "OUT_CSV":   str(out_root / f"{base}.ldlq.csv"),
    }

# =============================================================================
#                   Engine runner normalization
# =============================================================================
def _normalize_engine_return(ret) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Accept:
      - (df_pass1, df_pass2)
      - df (single)
      - None (fallback to engine globals)
    """
    df1 = df2 = None

    # 1) explicit tuple
    if isinstance(ret, tuple) and len(ret) == 2:
        df1, df2 = ret
    # 2) single df
    elif isinstance(ret, pd.DataFrame):
        df2 = ret
    # 3) None -> fallback to engine module globals
    if df1 is None:
        df1 = getattr(eng, "df_pass1", None)
    if df2 is None:
        df2 = getattr(eng, "df_pass2", None)

    # final sanity
    if df1 is not None and not isinstance(df1, pd.DataFrame):
        df1 = None
    if df2 is not None and not isinstance(df2, pd.DataFrame):
        df2 = None

    return df1, df2

def _require_dfs(df1, df2, *, where: str) -> Tuple[pd.DataFrame | None, pd.DataFrame]:
    """
    Require at least df2 exists and non-empty.
    """
    df_src = df2 if (df2 is not None and isinstance(df2, pd.DataFrame) and not df2.empty) else df1
    if df_src is None or not isinstance(df_src, pd.DataFrame) or df_src.empty:
        raise RuntimeError(f"[{where}] produced no DataFrame (df_pass1/df_pass2 are empty).")
    return df1, df2

# ========================================================
# Common: read/write helpers - Common save/report glue
# ========================================================
def save_candidates_bundle(
    *,
    df_pass1: pd.DataFrame | None,
    df_pass2: pd.DataFrame | None,
    out_paths: Dict[str, str],
    save_excel: bool = True,
    save_parquet: bool = True,
    save_csvgz: bool = True,
) -> pd.DataFrame:
    """
    Single place to save the final df (prefers pass2).
    """
    df_final = df_pass2 if (df_pass2 is not None and not df_pass2.empty) else df_pass1
    if df_final is None or df_final.empty:
        try:
            f = getattr(eng, "funnel", None)
            print(f"[SAVE][ERR] df_pass1 rows={0 if df_pass1 is None else len(df_pass1)}, df_pass2 rows={0 if df_pass2 is None else len(df_pass2)}")
            if isinstance(f, dict):
                print(f"[SAVE][ERR] funnel keys sample: pass_all={f.get('pass_all')}, pass_voltage={f.get('pass_voltage')}, pass_fill={f.get('pass_fill')}, pass_J={f.get('pass_J')}")
        except Exception:
            pass
        raise RuntimeError("[SAVE] No rows to save.")

    out_xlsx = out_paths["OUT_XLSX"]
    out_parq = out_paths["OUT_PARQ"]
    out_csvgz = out_paths["OUT_CSVGZ"]

    if save_excel:
        Path(out_xlsx).parent.mkdir(parents=True, exist_ok=True)
        df_final.to_excel(out_xlsx, index=False)
        print(f"[SAVE] Excel  -> {out_xlsx}  ({len(df_final)} rows)")

    if save_parquet:
        try:
            Path(out_parq).parent.mkdir(parents=True, exist_ok=True)
            df_final.to_parquet(out_parq, index=False, compression="zstd")
            print(f"[SAVE] Parquet-> {out_parq}")
        except Exception as e:
            print(f"[SAVE][WARN] Parquet failed: {e}")

    if save_csvgz:
        try:
            Path(out_csvgz).parent.mkdir(parents=True, exist_ok=True)
            df_final.to_csv(out_csvgz, index=False, compression="gzip")
            print(f"[SAVE] CSV.GZ -> {out_csvgz}")
        except Exception as e:
            print(f"[SAVE][WARN] CSV.GZ failed: {e}")

    return df_final

def save_df_bundle(
    df: pd.DataFrame,
    *,
    out_xlsx: str,
    out_parq: str,
    out_csvgz: str,
    save_excel: bool = True,
    save_parquet: bool = True,
    save_csvgz: bool = True,
) -> None:
    Path(out_xlsx).parent.mkdir(parents=True, exist_ok=True)

    if save_excel:
        df.to_excel(out_xlsx, index=False)
        print(f"[SAVE] xlsx  -> {os.path.abspath(out_xlsx)}")

    if save_parquet:
        try:
            df.to_parquet(out_parq, index=False, compression="zstd")
            print(f"[SAVE] parq  -> {os.path.abspath(out_parq)}")
        except Exception as e:
            print(f"[SAVE][WARN] parquet failed: {e}")

    if save_csvgz:
        try:
            df.to_csv(out_csvgz, index=False, compression="gzip")
            print(f"[SAVE] csvgz -> {os.path.abspath(out_csvgz)}")
        except Exception as e:
            print(f"[SAVE][WARN] csvgz failed: {e}")

def _load_df_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    if p.suffix.lower() in (".csv", ".gz"):
        return pd.read_csv(p)
    raise ValueError(f"Unsupported df format: {p.suffix}")

def dump_ldlq_db_json(ldlq_db: Dict[Tuple[int, int, int], Dict[str, float]], out_json: str) -> None:
    # JSON은 tuple key를 그대로 못 쓰므로 문자열 키로 저장
    serial = {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in ldlq_db.items()}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(serial, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] LdLq_DB json -> {os.path.abspath(out_json)}")


def load_ldlq_db_json(path: str) -> Dict[Tuple[int, int, int], Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        serial = json.load(f)
    out: Dict[Tuple[int, int, int], Dict[str, float]] = {}
    for k, v in serial.items():
        a, p, n = (int(x) for x in k.split("_"))
        out[(a, p, n)] = {"Ld_mH": float(v["Ld_mH"]), "Lq_mH": float(v["Lq_mH"])}
    return out

# =====================================================================
# Global wiring (make engine/config see the same OUT_* and save flags)
# =====================================================================
def _set_common_outputs_and_flags(
    *,
    out_paths: Dict[str, str],
    save_to_excel: bool,
    save_to_parquet: bool,
    save_to_csvgz: bool,
) -> None:
    # config 모듈에 세팅
    cfg.OUT_XLSX = out_paths["OUT_XLSX"]
    cfg.OUT_PARQ = out_paths["OUT_PARQ"]
    cfg.OUT_CSVGZ = out_paths["OUT_CSVGZ"]
    cfg.SAVE_TO_EXCEL = bool(save_to_excel)
    cfg.SAVE_TO_PARQUET = bool(save_to_parquet)
    cfg.SAVE_TO_CSVGZ = bool(save_to_csvgz)

    # engine 모듈에도 동일 세팅(엔진 코드가 전역 참조하는 경우 대비)
    eng.OUT_XLSX = out_paths["OUT_XLSX"]
    eng.OUT_PARQ = out_paths["OUT_PARQ"]
    eng.OUT_CSVGZ = out_paths["OUT_CSVGZ"]
    eng.SAVE_TO_EXCEL = bool(save_to_excel)
    eng.SAVE_TO_PARQUET = bool(save_to_parquet)
    eng.SAVE_TO_CSVGZ = bool(save_to_csvgz)


# ============================================================
#               Common reporting/saving funnel
# ============================================================
def _finalize_after_sweep(out_dir: str) -> None:
    """
    run_sweep() 경로(full/adaptive)에서 공통으로 호출할 후처리.
    - do_profile_summary_and_save()가 OUT_* / SAVE_*를 사용하므로, main에서 먼저 세팅되어야 함.
    """
    # worst-case cfg는 engine의 전역 WORST가 있으면 자동 사용
    try:
        eng.do_profile_summary_and_save(
            wc_cfg=getattr(eng, "WORST", None),
            fast_target_pct=0.05,
            exact_target_pct=0.05,
            exact_topk=200,
        )
    except TypeError:
        # 구버전 시그니처 호환
        eng.do_profile_summary_and_save()

    # (선택) fix workbook / 통계 출력은 engine.run_full_pipeline() 내부에서도 수행하지만,
    # adaptive는 여기서만 호출되므로, 필요 시 여기서 추가할 수 있음.
    try:
        eng.print_prof_summary()
    except Exception:
        pass

    # (선택) 최종 파일 위치 안내
    print(f"[DONE] Result path: {os.path.abspath(out_dir)}")


def _finalize_after_bflow(
    df_pass1: Optional[pd.DataFrame],
    df_pass2: Optional[pd.DataFrame],
    out_dir: str,
) -> Optional[pd.DataFrame]:
    """
    bflow 경로는 df를 반환하므로, 공통 저장/리포트 함수를 여기서 처리.
    """
    final_df = None
    if hasattr(eng, "_save_final"):
        final_df = eng._save_final(df_pass1, df_pass2, out_dir)
    else:
        # 최소 안전 저장
        final_df = df_pass2 if (df_pass2 is not None and not df_pass2.empty) else df_pass1
        if final_df is not None and not final_df.empty:
            # OUT_XLSX로 저장(세팅되어 있어야 함)
            final_df.to_excel(eng.OUT_XLSX, index=False)
            print(f"[SAVE] final -> {eng.OUT_XLSX}")

    # bflow에서도 요약/랭킹이 필요하면: do_profile_summary_and_save를 호출
    # (df_candidates를 엔진 전역 results에 쌓는 구조가 아니라면, 이 함수는 의미 없을 수 있음)
    # 필요 시, bflow 전용 요약 함수를 별도 구현하는 게 더 깔끔합니다.
    print(f"[DONE] Result path: {os.path.abspath(out_dir)}")
    return final_df

# ===================================================
#            Common: finalize/report
# ===================================================
def finalize_and_report(
    df_pass1: Optional[pd.DataFrame],
    df_pass2: Optional[pd.DataFrame],
    *,
    out_dir: str,
    out_paths: Dict[str, str],
    save_excel: bool,
    save_parquet: bool,
    save_csvgz: bool,
    stem: str,
) -> pd.DataFrame:
    from core.engine import _save_final

    final_df = _save_final(df_pass1, df_pass2, out_dir)
    if final_df is None or (isinstance(final_df, pd.DataFrame) and final_df.empty):
        raise RuntimeError("[FINAL] No final dataframe to save (empty).")

    save_df_bundle(
        final_df,
        out_xlsx=out_paths["OUT_XLSX"],
        out_parq=out_paths["OUT_PARQ"],
        out_csvgz=out_paths["OUT_CSVGZ"],
        save_excel=save_excel,
        save_parquet=save_parquet,
        save_csvgz=save_csvgz,
    )

    # optional post-report
    try:
        from core.progress import save_pass_heatmaps
        save_pass_heatmaps(final_df, out_prefix=str(Path(out_dir) / stem), pass_margin_pct=0.0)
    except Exception as e:
        print(f"[REPORT][WARN] heatmap skipped: {e}")

    return final_df


# =============================================================================
#               Modes (adaptive/full/bflow): sweep runners
# =============================================================================

def run_mode_full(args, out_paths) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    _set_common_outputs_and_flags(
        out_paths=out_paths,
        save_to_excel=not args.no_excel,
        save_to_parquet=not args.no_parquet,
        save_to_csvgz=not args.no_csvgz,
    )
    # full 모드도 adaptive처럼 5초마다 진행 출력
    from core.progress import set_progress_interval
    set_progress_interval(5.0)

    ret = eng.run_full_pipeline(
        out_xlsx=out_paths["OUT_XLSX"],
        out_parq=out_paths["OUT_PARQ"],
        out_csvgz=out_paths["OUT_CSVGZ"],
        save_to_excel=not args.no_excel,
        save_to_parquet=not args.no_parquet,
        save_to_csvgz=not args.no_csvgz,
    )
    df1, df2 = _normalize_engine_return(ret)
    return df1, df2


def run_mode_adaptive(args, out_paths) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    _set_common_outputs_and_flags(
        out_paths=out_paths,
        save_to_excel=not args.no_excel,
        save_to_parquet=not args.no_parquet,
        save_to_csvgz=not args.no_csvgz,
    )
    from core.progress import set_progress_interval
    # adaptive 모드도 adaptive처럼 5초마다 진행 출력
    set_progress_interval(5.0)

    hp = eng.load_hp_from_yaml_and_globals()
    cases = eng.build_power_torque_cases(hp)
    if not cases:
        raise RuntimeError("[ADAPTIVE] cases is empty. Check rpm_list / P_kW_list / T_Nm_list inputs.")

    ret = eng.setup_rpm_adaptive_envelope_and_run(
        cases=cases,
        par_hard_max=int(getattr(cfg, "PAR_HARD_MAX", 60)),
    )
    return _normalize_engine_return(ret)


def run_mode_bflow(args, out_paths) -> Tuple[pd.DataFrame | None, pd.DataFrame | None]:
    from core.progress import set_progress_interval
    set_progress_interval(5.0)

    # args 쪽 이름이 p_list인지 p_kw_list인지 프로젝트마다 달라서 방탄 처리
    p_list = getattr(args, "p_kw_list", None)
    if p_list is None:
        p_list = getattr(args, "p_list", None)

    # 방탄: p_list가 끝내 None이면 기본값(또는 config 기본) 사용
    if p_list is None:
        p_list = []

    rpm_list = getattr(args, "rpm_list", None)
    if not rpm_list:
        rpm_list = [600, 1800, 3600]

    _set_common_outputs_and_flags(
        out_paths=out_paths,
        save_to_excel=not args.no_excel,
        save_to_parquet=not args.no_parquet,
        save_to_csvgz=not args.no_csvgz,
    )

    # ✅ bflow는 engine 내부가 전역 OUT_XLSX를 참조할 수 있으므로 안전하게 전역 고정
    eng.OUT_XLSX  = out_paths["OUT_XLSX"]
    eng.OUT_PARQ  = out_paths["OUT_PARQ"]
    eng.OUT_CSVGZ = out_paths["OUT_CSVGZ"]

    ret = eng.run_bflow_full_two_pass(
        rpm_list=rpm_list,
        P_kW_list=p_list,           # ✅ 방탄으로 구한 p_list를 사용
        T_Nm_list=None,
        motor_type=args.motor_type,
        min_margin_pct=args.min_margin_pct,
        passrows_topk=args.passrows_topk,
        out_xlsx=out_paths["OUT_XLSX"],
    )
    return _normalize_engine_return(ret)

#final_df = finalize_and_report(
#    df_pass1, df_pass2,
#    out_dir=args.out_dir,
#    out_paths=out_paths,
#   save_excel=not args.no_excel,
#    save_parquet=not args.no_parquet,
#    save_csvgz=not args.no_csvgz,
#    stem=args.stem,
#)


# ==============================================================================
#            AI 기반 Bflow 확장 (Surrogate 모델 통합) run_mode_AI_bflow
# ==============================================================================
def run_mode_aibflow(args, out_paths):
    """
    고도화된 Surrogate 모델(GPR)을 적용한 지능형 Bflow
    """
    print("\n[AI-BFLOW] Phase 1: Global Space Scanning...")
    # 1. 기초 데이터 생성을 위한 Pass 1 실행
    df_pass1, _ = run_mode_bflow_pass1(args, out_paths)

    if df_pass1 is None or df_pass1.empty:
        print("[ERR] Pass 1 failed to generate data.")
        return None, None

    # 2. 고도화된 Surrogate 모델 초기화 및 학습
    # 단순 RandomForest가 아닌 가우시안 프로세스로 설계 공간의 지도를 그림
    surrogate = DesignSurrogate()
    surrogate.train(df_pass1)

    # 3. Smart Narrowing 실행 (UCB 알고리즘 적용)
    # AI가 '성능이 높거나' 혹은 '아직 탐색이 부족하여 유망한' 후보 500개를 추출
    print("[AI-BFLOW] Phase 2: Bayesian Intelligent Narrowing...")
    df_pass1_ai = run_smart_narrowing(surrogate, df_pass1, top_n=args.passrows_topk)

    # 4. 필터링된 정예 후보들에 대해서만 정밀 해석(Pass 2) 수행
    print(f"[AI-BFLOW] Phase 3: Precise Analysis for {len(df_pass1_ai)} candidates...")
    df_pass1_ai, df_pass2 = run_mode_bflow_pass2(args, df_pass1_ai, out_paths)
    
    return df_pass1_ai, df_pass2

# =============================================================================
#                    강화학습 기반 설계 탐색 (RL Search 모드)
# =============================================================================
def evaluate_design_physically(state):
    """
    RL 에이전트가 제안한 단일 설계안(state)의 물리적 타당성 검토
    engine.py의 핵심 물리 로직을 단일 케이스에 대해 수행
    (예시값 반환, 실제 구현 시 physics.apply_envelope_for_case 호출)
    state: (awg, parallels, turns, rpm)
    """
    awg, par, turns, rpm = state
    
    # 1. 목표 출력(kW)으로부터 필요 토크 계산
    # config에 정의된 Target_Power_kW를 기준으로 계산 (예: 1.5kW)
    # 목표 부하 계산
    target_kw = getattr(cfg, "Target_Power_kW", 1.5)
    target_torque = kw_rpm_to_torque_nm(target_kw, rpm)

    # 2. 단일 케이스 해석을 위한 텐서 생성 (Batch Size = 1)
    # core.physics 로직은 텐서 기반이므로 단일 값도 텐서로 변환 필요
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # [중요] rebuild_awg_par_tensors의 로직을 로컬에서 수행
    # 단순 AWG 번호가 아니라 실제 구리 단면적(mm2) 텐서가 물리 엔진에 들어가야 함
    t_awg = torch.tensor([float(awg)], device=device)
    t_area = torch.tensor([awg_area_mm2(int(awg))], device=device) # 단면적 변환 로직
    t_par = torch.tensor([float(par)], device=device)
    t_turns = torch.tensor([float(turns)], device=device)


    # 3. 물리 엔진(Envelope Analysis) 호출
    # apply_envelope_for_case는 전압 제한원과 토크 곡선을 비교하여 마진을 산출함
    try:
        # 물리 엔진 호출 (Batch Size = 1)
        # 여기서 awg_area_tensor 인자에 t_area를 명확히 전달해야 함
        results = apply_envelope_for_case(
            awg_tensor=t_awg,
            awg_area_tensor=t_area, # 이 부분이 누락되면 해석 오류 발생
            par_tensor=t_par,
            turns_tensor=t_turns,
            target_rpm=rpm,
            target_torque_nm=target_torque,
            motor_type=getattr(cfg, "MOTOR_TYPE", "IPM")
        )
        
        # 결과 텐서에서 스칼라 값 추출
        # results는 보통 딕셔너리 형태이며 'margin_pct'와 'fail_prob' 포함
        margin_pct = results['margin_pct'].item()
        
        # 물리적 실패 여부 판정 (예: 전압 초과, 점적률 초과 등)
        # 0.0은 성공, 1.0은 완전 실패를 의미
        fail_prob = 1.0 - results['success_mask'].item() 
        
    except Exception as e:
        # 물리 시뮬레이션 중 오류 발생 시 (예: 불가능한 기하학적 수치)
        print(f"[PHYS_ERR] Evaluation failed for state {state}: {e}")
        return -100.0, 1.0  # 최악의 리워드를 위한 리턴

    return margin_pct, fail_prob

def run_rl_design_search(args):
    """
    Deep Q-Network 기반의 지능형 설계 최적화 탐색
    """
    print("\n[RL-SEARCH] Initializing Deep Q-Network Agent...")
    agent = DQNAgent(action_space=[-1, 0, 1])
    
    # State: [AWG, Par, Turns, RPM], Action: 27 combos
    agent = DQNAgent(state_size=4, action_size=27)

    # 초기 설계 상태 설정
    current_state = (cfg.AWG, cfg.Parallels, cfg.Turns, cfg.Target_RPM)
    batch_size = 32
    history = []

    print("[RL-SEARCH] Exploration Start...")
    for episode in range(getattr(args, "rl_steps", 50)):
        # 1. AI의 결정 (Epsilon-Greedy 탐험 포함)
        action_idx = agent.act(current_state)
        
        # 2. Action Index를 물리적 증감값으로 변환 (-1, 0, 1 조합)
        # 27개 인덱스를 (dAWG, dPar, dTurns)로 해제
        d_awg = (action_idx // 9) - 1
        d_par = ((action_idx // 3) % 3) - 1
        d_turns = (action_idx % 3) - 1
        
        next_state = (
            max(10, min(30, current_state[0] + d_awg)),
            max(1, min(16, current_state[1] + d_par)),
            max(1, min(100, current_state[2] + d_turns)),
            current_state[3] # RPM 고정
        )

        # 3. 물리 엔진을 통한 즉각 평가 (점적률 포함)
        margin, fail = evaluate_design_physically(next_state)
        fill_factor = calculate_fill_factor(next_state[0], next_state[1], next_state[2])
        
        # 4. 고도화된 보상 계산 (제조 가능성 제약 연동)
        reward = agent.get_refined_reward(margin, fail, next_state)
        
        # 5. 경험 저장 및 신경망 학습
        agent.remember(current_state, action_idx, reward, next_state, done=False)
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            
        current_state = next_state
        
        if episode % 10 == 0:
            print(f"  Step {episode:3d} | Margin: {margin:5.2f}% | Fill: {fill_factor*100:4.1f}% | Reward: {reward:6.2f}")
            # 주기적으로 타겟 네트워크 업데이트
            agent.update_target_model()

    return pd.DataFrame(history)


# ==============================================================
#           Modes: femm_gen / femm_extract / feedback
# ==============================================================
def run_mode_femm_gen(args, df_pass2, out_dir):
    """
    모드 4: FEMM 모델 자동 생성 실행부
    """
    print(f"\n[MODE] Starting FEMM Generation Pipeline...")
    
    import configs.config as cfg
    # config의 D_use를 기반으로 반지름(r_mid) 전달
    r_mid = cfg.D_use / 2.0
    
    # 통합 실행 함수 호출 (내부에서 results/femm_models로 저장됨)
    from utils.femm_builder import run_femm_generation
    run_femm_generation(
        df_results=df_pass2,
        target_dir=out_dir,
        r_slot_mid_mm=r_mid
    )
    print(f"[DONE] FEMM generation process completed.")


def run_mode_femm_extract(args, out_paths):
    import glob
    import pandas as pd
    from utils.femm_builder import get_femm_results
    from utils.femm_ldlq import calculate_ld_lq_from_flux
    from utils.femm_ldlq import batch_extract_ldlq_from_femm

    """
    저장된 .ans 파일들로부터 Ld/Lq를 추출하여 df_pass2에 업데이트하고 DB를 생성합니다.
    """

    print("[MODE] Extracting Ld/Lq from FEMM Results...")
    
    # 1. 결과가 저장될 디렉토리 확인
    femm_dir = os.path.join(args.out_dir, "femm_models")
    results = []

    print(f"[MODE] Extracting Ld/Lq from: {femm_dir}")

    # 2. df_pass2의 행을 순회하며 매칭되는 파일 찾기
    for idx, row in df_pass2.iterrows():
        awg = int(row.get("AWG", 0))
        par = int(row.get("Parallels", 1))
        # 생성된 파일 규칙에 맞게 파일명 조립
        ans_name = f"24S4P_AWG{awg:02d}_P{par}_idx{idx}.ans"
        ans_path = os.path.join(femm_dir, ans_name)

        if os.path.exists(ans_path):
            # 파일에서 데이터 추출 (get_femm_results 함수 활용)
            femm_data = get_femm_results(ans_path) 
            
            if femm_data and "all_phases" in femm_data:
                # Park 변환으로 Ld, Lq 계산
                flux_abc = [femm_data["all_phases"]["A"][0], 
                            femm_data["all_phases"]["B"][0], 
                            femm_data["all_phases"]["C"][0]]
                current = femm_data["current"]
                
                Ld, Lq, _, _ = calculate_ld_lq_from_flux(flux_abc, current)
                
                # 결과 리스트 저장
                results.append({
                    "idx": idx,
                    "Ld_mH": Ld * 1000,
                    "Lq_mH": Lq * 1000
                })
                print(f"  > [Extracted] {ans_name}: Ld={Ld*1000:.3f}mH, Lq={Lq*1000:.3f}mH")

    # 3. 데이터프레임과 병합 및 저장
    if results:
        df_ldlq = pd.DataFrame(results).set_index("idx")
        df_final = df_pass2.join(df_ldlq, how="left")
        
        out_excel = os.path.join(args.out_dir, "df_pass2_with_LdLq.xlsx")
        df_final.to_excel(out_excel)
        print(f"[DONE] Extraction complete. Updated Excel: {out_excel}")
        return df_final
    else:
        print("[WARN] No matching .ans files found to extract data.")
        return df_pass2


def run_mode_feedback(args, df_pass2: pd.DataFrame, out_dir: str):
    # load ldlq db
    if args.ldlq_json and os.path.exists(args.ldlq_json):
        with open(args.ldlq_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        ldlq_db = {}
        for k_str, v in raw.items():
            # keys stored as "(17, 3, 20)" string
            k = tuple(int(x) for x in k_str.strip("() ").split(","))
            ldlq_db[k] = {"Ld_mH": float(v["Ld_mH"]), "Lq_mH": float(v["Lq_mH"])}
    else:
        femm_dir = args.femm_dir or str(Path(out_dir) / "femm")
        I_test = float(args.I_test if args.I_test is not None else FEMM_I_TEST_DEFAULT)
        ldlq_db = batch_extract_ldlq_from_femm(femm_dir=femm_dir, I_test=I_test)

    if not ldlq_db:
        print("[FEEDBACK] empty Ld/Lq db -> skip.")
        return df_pass2

    # (b) feedback strategy: register cache vs df overwrite
    strategy = (args.feedback_strategy or "both").lower()

    if strategy in ("register", "both"):
        for k, v in ldlq_db.items():
            phys.register_ldlq_from_femm(k, v["Ld_mH"], v["Lq_mH"])
        print(f"[FEEDBACK] registered {len(ldlq_db)} keys into physics cache.")

    if strategy in ("overwrite", "both"):
        df2 = phys.apply_ldlq_feedback(df_pass2, ldlq_db)
        print("[FEEDBACK] applied df overwrite feedback (Ld_mH/Lq_mH updated where key exists).")
    else:
        df2 = df_pass2

    # save updated df_pass2 (so next femm_gen uses updated Ld/Lq)
    out_path = str(Path(out_dir) / "df_pass2_with_ldlq.xlsx")
    df2.to_excel(out_path, index=False)
    print(f"[FEEDBACK] saved updated df_pass2 -> {out_path}")

    return df2


import math
import configs.config as cfg

class ScrollLoadCoupler:
    @staticmethod
    def check_matching(Ld, Lq, Target_Torque, Target_Speed):
        """
        Ld, Lq를 기반으로 목표 운전점에서 전압/전류 제한 내 구동 가능 여부 판정
        """
        # --- [환경 변수 설정] ---
        Vdc = getattr(cfg, "Vdc", 310)       # 직류단 전압
        Vmax = (Vdc / math.sqrt(3)) * 0.95  # 인버터 전압 이용률 고려 최대 전압
        Imax = getattr(cfg, "Imax", 15)     # 최대 허용 전류 (Arms)
        Pn = cfg.N_poles / 2                # 극쌍수
        Psi_m = getattr(cfg, "Flux_linkage_min", 0.05) # 영구자석 자속 (Wb)
        Rs = getattr(cfg, "R_phase", 0.5)   # 상저항
        
        we = (Target_Speed * 2 * math.pi / 60) * Pn # 전기적 각속도 (rad/s)
        
        # --- [MTPA 기반 Id, Iq 탐색] ---
        # 실제로는 수치해석(Newton-Raphson)이 필요하나, 판정을 위해 단순화된 탐색 수행
        best_match = {"status": "Fail", "efficiency": 0, "Id": 0, "Iq": 0}
        
        # 전류 크기와 위상각을 스윕하며 적합한 운전점 탐색
        for Is in range(1, int(Imax) + 1):
            for beta_deg in range(0, 45, 2): # 진각 제어(Field Weakening) 고려
                beta = math.radians(beta_deg)
                id_curr = -Is * math.sin(beta)
                iq_curr = Is * math.cos(beta)
                
                # 1. 발생 토크 계산
                Te = 1.5 * Pn * (Psi_m * iq_curr + (Ld - Lq) * id_curr * iq_curr)
                
                if Te >= Target_Torque:
                    # 2. 해당 전류에서의 단자 전압 계산
                    Vd = Rs * id_curr - we * Lq * iq_curr
                    Vq = Rs * iq_curr + we * (Ld * id_curr + Psi_m)
                    Va = math.sqrt(Vd**2 + Vq**2)
                    
                    if Va <= Vmax:
                        # 전압/전류 제한 만족 시 효율 추정 (단순 동손 기준)
                        copper_loss = 1.5 * Rs * (id_curr**2 + iq_curr**2)
                        output_power = Te * (Target_Speed * 2 * math.pi / 60)
                        efficiency = (output_power / (output_power + copper_loss)) * 100
                        
                        best_match = {
                            "status": "Pass",
                            "efficiency": round(efficiency, 2),
                            "Id": round(id_curr, 2),
                            "Iq": round(iq_curr, 2),
                            "Voltage": round(Va, 1)
                        }
                        return best_match # 조건을 만족하는 최소 전류 운전점 발견 시 즉시 반환
        
        return best_match


# main.py 에서의 전형적인 활용 흐름
def run_esc_design_platform():
    # Step 1: 12개 후보군 생성 및 FEMM 해석 실행
    candidates = generate_design_candidates()
    for design in candidates:
        build_fem_from_winding(design['winding_table'], design['path'], cfg.R_mid)
    
    # Step 2: [extract_results_batch 활용] 
    # 폴더 내 모든 결과를 스캔하여 물리 데이터 추출 (Pandas DataFrame 반환)
    raw_data_df = extract_results_batch() 
    
    # Step 3: 물리 데이터를 제어 파라미터로 변환 (DQ Transform)
    # 이 과정에서 실시간 부하(Scroll Load) 커플링을 위한 준비 완료
    dq_results = extract_results_with_dq_transform(raw_data_df)
    
    # Step 4: 최적안 선정 및 AI 피드백
    best_idx = dq_results['Salient_Ratio'].idxmax()
    print(f"AI 추천 최적 설계안: {dq_results.loc[best_idx, 'FileName']}")


# ==========================================================================
#                                main
# ==========================================================================

def main():
    print("="*60)
    print("ESC AI Motor Design Platform - Integrated Pipeline")
    print("="*60)

    # --- [Step 1: 결선 스펙 정의 (from winding_spec.py)] ---
    # 병렬 회로 수 설정 (예: A, B, C상 모두 1병렬)
    conn_spec = WindingConnSpec(n_parallel_circuits_per_phase={"A":1, "B":1, "C":1})
    
    # 병렬 회로 수가 1이 아닌 경우, 병렬 회로 수에 따라 코일 수를 조정해야 합니다.
    if any(v > 1 for v in conn_spec.n_parallel_circuits_per_phase.values()):
        print("[WARN] Non-unity parallel circuits detected. Consider adjusting coil count.")

    # 전역 상당 직렬 코일 수 고정 (24S4P 분포권 기준)
    try:
        cph = lock_coils_per_phase_global(conn=conn_spec, n_slots=24, double_layer=True)
        print(f"[INFO] Locked Coils Per Phase: {cph}")
    except ValueError as e:
        print(f"[ERR] Winding Spec Error: {e}")
        return
    
    # --- [Step 2: 설계 후보군 로드 및 권선표 매핑/생성 (from winding_table.py)] ---
    # 후보군 생성 시 winding_table.py의 로직을 사용하여 상세 권선표를 함께 준비합니다.
    print("\n[STEP 2] Preparing Design Candidates & Winding Tables...")
    candidates = generate_design_candidates() 
    
    for design in candidates:
        # 각 후보의 Turns_per_slot_side를 바탕으로 상세 권선표 생성  -  24S4P 전용 권선표 생성 로직 호출
        # build_winding_table_24s4p 함수를 직접 활용
        design['winding_table'] = build_winding_table_24s4p(
            turns_per_slot_side=design.get('turns', 10),
            n_slots=24,
            n_poles=4,
            coil_span_slots=5, # 1-6 권선
            double_layer=True,
            parallels=2 # 위에서 설정한 병렬 수 반영
        )

    # --- [Step 3: FEMM 자동 해석 루프] ---
    print("\n[STEP 3] Running FEMM Automated Analysis...")
    for design in candidates:
        print(f" -> Solving: {design['name']}")
        build_fem_from_winding(
            winding_table=design['winding_table'], 
            file_path=design['fem_path'], 
            r_slot_mid=cfg.R_mid
        )

    # --- [Step 4: DQ 변환 및 데이터 추출] ---
    # Batch 처리를 통해 모든 .ans 파일에서 Ld, Lq 데이터 도출
    print("\n[STEP 4] Extracting DQ Parameters (Ld, Lq)...")
    analysis_df = extract_results_with_dq_transform()

    if analysis_df is None or analysis_df.empty:
        print("[ERR] No analysis data found. Check results/ans folder.")
        return

    # --- [Step 5: 실 부하 적합성 판정 (Scroll Coupling)] ---
    print("\n[STEP 5] Final Suitability Test and Report Generation...")
    # 도출된 DQ 파라미터를 스크롤 압축기 부하 곡선과 비교하여 최적안 선정
    # (이 단계에서 최종 Excel 리포트가 생성됩니다.)
    print("\n[STEP 5] Performing Scroll Load Matching...")
    final_results = []

    for _, row in analysis_df.iterrows():
        # 추출된 Ld, Lq를 부하 엔진에 투입
        suitability = ScrollLoadCoupler.check_matching(
            Ld=row['Ld(H)'], 
            Lq=row['Lq(H)'], 
            Target_Torque=getattr(cfg, "Target_Torque", 1.2),
            Target_Speed=getattr(cfg, "Target_RPM", 3600)
        )
        
        # 데이터 통합
        res_entry = row.to_dict()
        res_entry.update({
            'Load_Match': suitability['status'],
            'Efficiency_Est(%)': suitability['efficiency'],
            'Req_Voltage(V)': suitability['Voltage'],
            'Operating_Id': suitability['Id'],
            'Operating_Iq': suitability['Iq']
        })
        final_results.append(res_entry)

    # --- [Step 6: 리포트 저장 및 최적 설계 출력] ---
    final_df = pd.DataFrame(final_results)
    
    # 결과 폴더 생성 및 엑셀 저장
    os.makedirs("./results", exist_ok=True)
    report_path = "./results/Final_ESC_Design_Report.xlsx"
    final_df.to_excel(report_path, index=False)
    
    # 적합 모델 중 돌극비(Salience)가 가장 높은 모델 추천
    pass_models = final_df[final_df['Load_Match'] == 'Pass']
    
    print("\n" + "="*60)
    if not pass_models.empty:
        best = pass_models.loc[pass_models['Salient_Ratio(Lq/Ld)'].idxmax()]
        print(f"★ OPTIMAL DESIGN: {best['FileName']}")
        print(f" - Ld: {best['Ld(H)']: .6f} / Lq: {best['Lq(H)']: .6f}")
        print(f" - Efficiency: {best['Efficiency_Est(%)']}% at Target Load")
    else:
        print(" [!] No design candidates passed the Load Test.")
    
    print(f"\n[DONE] Full report saved at: {report_path}")
    print("="*60)


    args = parse_args()
    # progress globals are owned by core.progress (single source of truth)
    init_progress(
        ENABLE_PROFILING=bool(getattr(cfg, "ENABLE_PROFILING", False)),
        live_progress=bool(getattr(cfg, "LIVE_PROGRESS", True)),
        progress_every_sec=float(getattr(cfg, "PROGRESS_EVERY_SEC", 3.0)),
        device=getattr(cfg, "DEVICE", None),
    )

    # --- Interactive fallback ---
    if args.mode is None:
        print(
            "\nWhich of the following modes do you want to be executed "
            "(--mode full, --mode adaptive, or --mode bflow)?"
        )
        timeout = None if (args.menu_timeout is None or args.menu_timeout <= 0) else int(args.menu_timeout)
        args.mode = choose_mode_interactively(
            default_mode="adaptive",
            timeout_sec=timeout,
            allow_extended=True,   # femm_* / feedback 메뉴도 띄우기
        )
        while True:
            mode_input = input("Please input one of 3 modes: ").strip().lower()
            if mode_input in ["full", "adaptive", "bflow"]:
                args.mode = mode_input
                break
            else:
                print("Invalid input. Please enter: full, adaptive, or bflow.")
    # ----------------------------
    # (0) 단발성 테스트/디버그가 필요하면 여기서만
    # res = calculate_reverse_power(cfg, 1.0388, 16, 40, T_oper_C=120)
    # print(res)
    os.makedirs(args.out_dir, exist_ok=True)
    out_paths = build_output_paths(
        out_dir=args.out_dir,
        stem=f"{args.stem}_{args.mode}",   #  mode를 파일명에 포함
    )
    
    # [2] 초기화/튜닝 (단 1회)
    autotune_par_candidates_for_revision(
        safety_extra=2,
        auto_raise_hard_max=False,
        keep_user_list_if_ok=False,
    )

    print(f"[RUN] mode={args.mode}")
    print(f"[OUT] dir={os.path.abspath(out_paths['OUT_DIR'])}")
    # ---- ensure SSOT coil_per_phase lock before any sweep
    # NOTE: if you already moved SSOT print into config import, this is optional.
    try:
        conn = {"A": 2, "B": 2, "C": 2}  # typical 24S4P 2-parallel-circuits example
        cph = lock_coils_per_phase_global(conn=None, n_slots=int(getattr(cfg, "N_slots", 24)), double_layer=True)
        _ = cph
    except Exception:
        pass

    df_pass1 = df_pass2 = None

    if args.mode in ("full", "adaptive", "bflow", "aibflow", "rl_search"):
        if args.mode == "full":
            df_pass1, df_pass2 = run_mode_full(args, out_paths)
        elif args.mode == "adaptive":
            df_pass1, df_pass2 = run_mode_adaptive(args, out_paths)
            # ================= DEBUG BLOCK START =================
            print(f"[SAVE][DBG] df_pass1 rows={0 if df_pass1 is None else len(df_pass1)}")
            print(f"[SAVE][DBG] df_pass2 rows={0 if df_pass2 is None else len(df_pass2)}")

            import core.engine as eng
            print(f"[SAVE][DBG] results type={type(eng.results)} "
                f"len={len(eng.results) if isinstance(eng.results,list) else 'n/a'}")

            if isinstance(eng.results, list) and eng.results:
                print("[SAVE][DBG] results[0] type=", type(eng.results[0]))
            # ================= DEBUG BLOCK END =================
        elif args.mode == "bflow":
            df_pass1, df_pass2 = run_mode_bflow(args, out_paths)
        elif args.mode == "aibflow":
            # AI 대리 모델을 활용한 초고속 필터링 모드
            df_pass1, df_pass2 = run_mode_aibflow(args, out_paths)
        elif args.mode == "rl_search":
            # 강화학습 에이전트가 스스로 설계안을 수정하며 최적점을 찾는 모드
            #df_pass1, df_pass2 = run_rl_design_search(args), None  # RL Search는 df_pass1에 히스토리 반환, df_pass2는 없음
            df_rl = run_rl_design_search(args)
            df_rl.to_excel(os.path.join(args.out_dir, "RL_Optimization_Log.xlsx"))

        # common save (engine may already save; this guarantees a consistent artifact)
        df_final = save_candidates_bundle(
            df_pass1=df_pass1, df_pass2=df_pass2, out_paths=out_paths,
            save_excel=not args.no_excel,
            save_parquet=not args.no_parquet,
            save_csvgz=not args.no_csvgz,
        )
        print(f"[DONE] mode={args.mode} rows={len(df_final)} out_dir={args.out_dir}")
        return 0

    # FEMM pipeline modes need df_pass2
    if args.df_pass2:
        df_pass2 = _load_df_any(args.df_pass2)
    else:
        # fallback: try latest saved in out_dir (pick newest .xlsx containing df_pass2_with_ldlq or stem)
        candidates = sorted(Path(args.out_dir).glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            df_pass2 = pd.read_excel(candidates[0])
            print(f"[LOAD] df_pass2 from latest: {candidates[0]}")
        else:
            raise RuntimeError("No df_pass2 provided and no saved excel found in out_dir.")

    if args.mode == "femm_gen":
        run_mode_femm_gen(args, df_pass2=df_pass2, out_dir=args.out_dir)
        return 0

    if args.mode == "femm_extract":
        run_mode_femm_extract(args, out_paths)
        return 0

    if args.mode == "feedback":
        df2 = run_mode_feedback(args, df_pass2=df_pass2, out_dir=args.out_dir)
        # optionally: immediately generate FEMM again based on updated df
        if getattr(args, "fw_margin_min", None) is not None:
            _ = df2
        return 0

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())

"""
사용 예시: 
# (추천) adaptive: 빠르게 PASS 찾기(run_sweep 기반)
python main.py --mode adaptive --out_dir results

# full: 올인원(정상화된 파이프라인일 때 배포용)
python main.py --mode full --out_dir results

# bflow: PASS가 아예 안 나올 때 seed/2-pass로 구제
python main.py --mode bflow --out_dir results

"""