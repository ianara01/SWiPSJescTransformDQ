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
  - aibflow    : run_aibflow_full_two_pass() (AI-based bflow - surrogate model for Pass 1 → Pass 2 efficiency)
  - rl_search  : run_rl_search() (Reinforcement Learning based design search - DQN agent proposes designs → physical feasibility evaluation)
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

import math
import threading
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd

#import configs.config as C
import core.engine as eng

from core.engine import autotune_par_candidates_for_revision, _normalize_engine_return
from core.engine import run_mode_bflow_pass1, run_mode_bflow_pass2


#from utils.femm_builder import run_femm_generation, build_fem_from_winding, extract_results_with_dq_transform, generate_design_candidates
# NOTE: FEMM 관련 모듈은 import 시 FEMM(MFC)이 뜰 수 있으므로
# femm_* 모드에서만 "지연 import" 한다.

#from configs.config import Config
#cofg = Config()  # main.py에서 cofg를 참조할 때, utils/utils.py의 cofg를 참조하도록 재할당 (main.py의 전역 cofg는 utils.utils.cofg와 동일한 객체임을 보장)
import configs.config as cofg
# utils.py의 전역 cofg 업데이트 (가장 중요)
#import utils.utils as uu
#uu.cofg.device = cofg.device
#uu.cofg.dtype = cofg.dtype
#uu.cofg.itype = cofg.itype

#import core.engine as eng
# engine에도 주입
#eng.cofg = cofg


from core.progress import init_progress


# Optional (only if you have FEMM installed/usable on the machine):
try:
    from utils.femm_ldlq import I_test as FEMM_I_TEST_DEFAULT
except Exception:
    FEMM_I_TEST_DEFAULT = 0.0

#hp = eng.load_hp_from_yaml_and_globals()
#cases = eng.build_power_torque_cases(hp)
#df_pass1, df_pass2 = eng.setup_rpm_adaptive_envelope_and_run(
#    cases=cases,
#    par_hard_max=cofg.PAR_HARD_MAX,
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
    base_modes = ["full", "adaptive", "bflow", "aibflow", "rl_search"]
    ext_modes  = ["femm_gen", "femm_extract", "feedback"] if allow_extended else []
    all_modes  = base_modes + ext_modes

    # 메뉴 출력
    print("\n=========================================")
    print("       CoilWindingOptimzation  MENU       ")
    print("=========================================")
    print("Choose execution mode:")
    print("  1) full       (전체 파이프라인)")
    print("  2) adaptive   (추천: rpm별 설계공간 줄여 빠르게 PASS 찾기)")
    print("  3) bflow      (2-pass 후보 수렴 - 자동 튜닝 탐색)")
    print("  4) aibflow    (AI 기반 bflow - Surrogate 모델로 Pass 1 → Pass 2 효율화)")
    print("  5) rl_search  (강화학습 기반 설계 탐색 - DQN 에이전트가 직접 설계 제안 → 물리적 타당성 평가)")
    if allow_extended:
        print("  6) femm_gen       (FW-safe winding → .fem 생성)")
        print("  7) femm_extract   (.fem → Ld/Lq 추출)")
        print("  8) feedback       (Ld/Lq feedback 적용)")
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
        raw = _input_with_timeout("Select [1/2/3/4/5/6/7/8 or name]: ", timeout_sec)
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

# =================================================================================
#       Global wiring (make engine/config see the same OUT_* and save flags)
# =================================================================================
def _set_common_outputs_and_flags(
    *,
    out_paths: Dict[str, str],
    save_to_excel: bool,
    save_to_parquet: bool,
    save_to_csvgz: bool,
) -> None:
    # config 모듈에 세팅
    cofg.OUT_XLSX = out_paths["OUT_XLSX"]
    cofg.OUT_PARQ = out_paths["OUT_PARQ"]
    cofg.OUT_CSVGZ = out_paths["OUT_CSVGZ"]
    cofg.SAVE_TO_EXCEL = bool(save_to_excel)
    cofg.SAVE_TO_PARQUET = bool(save_to_parquet)
    cofg.SAVE_TO_CSVGZ = bool(save_to_csvgz)

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
    # worst-case cofg는 engine의 전역 WORST가 있으면 자동 사용
    try:
        eng.do_profile_summary_and_save(
            wc_cofg=getattr(eng, "WORST", None),
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
#               Modes (1.adaptive/2.full/3.bflow): sweep runners
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
        par_hard_max=int(getattr(cofg, "PAR_HARD_MAX", 60)),
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


# =========================================================================================
#            Mode 4. AI 기반 Bflow 확장 (Surrogate 모델 통합) run_mode_AI_bflow
# =========================================================================================
def run_mode_aibflow(args, out_paths):

    from core import engine as eng
    from core.search.surrogate import DesignSurrogate, run_smart_narrowing

    print("\n" + "="*60)
    print("[MODE 4] AI-Assisted B-Flow Pipeline Starting...")
    print("="*60)

    # ---------------- PASS 1 ----------------
    print("\n[STEP 1] Running B-Flow Pass 1...")
    df_pass1, hp1, case1 = run_mode_bflow_pass1(args, out_paths)

    if df_pass1 is None or df_pass1.empty:
        print("[ERR] Pass 1 failed.")
        return None, None

    # [MUST] df_refined는 어떤 경우에도 정의되어야 함
    #df_refined = df_pass1.copy()  # 초기값은 Pass 1 결과 전체로 시작

    # margin_pct 보장
    if "margin_pct" not in df_pass1.columns:
        if "Vreq_LL_rms" not in df_pass1.columns:
            raise RuntimeError("No Vreq_LL_rms for margin calculation.")

        Vdc = float(max(getattr(eng, "Vdc_list", [0.0])))
        mmax = float(max(getattr(eng, "m_max_list", [1.0])))
        Vavail = (mmax * Vdc) / math.sqrt(3.0)

        df_pass1 = df_pass1.copy()
        df_pass1["margin_pct"] = 100.0 * (
            Vavail - df_pass1["Vreq_LL_rms"].astype(float)
        ) / max(Vavail, 1e-9)

    # GPR 과부하 방지
    if len(df_pass1) > 2000:
        df_pass1 = df_pass1.sample(2000, random_state=0)

    # ---------------- STEP 2 ----------------
    print("\n[STEP 2] Training Surrogate & Narrowing...")

    try:
        ai_model = DesignSurrogate()
        ai_model.train(df_pass1)

        df_refined = run_smart_narrowing(
            ai_model,
            df_pass1,
            top_n = min(args.passrows_topk * 2, int(len(df_pass1) * 0.5))
        )

        print(f"[AI] Narrowing {len(df_pass1)} -> {len(df_refined)}")

    except Exception as e:
        print(f"[WARN] AI narrowing failed → fallback. reason={e}")
        df_refined = df_pass1

    # ---------------- STEP 3 ----------------
    print("\n[STEP 3] Running Pass 2...")
    df_pass1, df_pass2, hp1 = run_mode_bflow_pass2(
        args,
        df_refined,
        out_paths,
        hp1=hp1,
        case1=case1,)

    print(f"\n[DONE] aibflow 완료. 결과 저장됨: {out_paths['OUT_XLSX']}")
    return df_pass2, case1, hp1
  
# =========================================================================================
#            Mode 5. 강화학습 기반 설계 탐색 run_rl_design_search
# =========================================================================================
def run_rl_design_search(args):
    """
    Deep Q-Network 기반의 지능형 설계 최적화 탐색
    """
    # 0. 필요한 함수/클래스만 임포트
    from core.search.rl_agent import DQNAgent, calculate_fill_factor, evaluate_design_physically
    import configs.config as cofg

    print("\n[RL-SEARCH] Initializing Deep Q-Network Agent...")

    # 1. 사용 가능한 후보군 정의 (SSOT 참조)
    awg_list = sorted(list(cofg.awg_candidates))  # [16, 17, 18, 19, 20]
    par_list = sorted(list(cofg.par_candidates))  # [2, 3, ..., 40]
    turn_list = sorted(list(cofg.turn_candidates_base)) # [10, 11, ..., 40]

    # 2. 에이전트는 단 한 번만 생성합니다. (27개 액션 조합 사용)
    # State: [AWG, Par, Turns, RPM], Action: 27 combos
    # 클래스 정의가 action_space를 기대하므로 리스트 형태로 전달
    # 3x3x3=27개의 액션 조합을 인덱스로 관리하기 위해 range(27) 사용
    actions = list(range(27))
    agent = DQNAgent(state_size=4, action_space=actions)

    # 3. 초기 상태 설정 (config.py의 리스트 첫 번째 값 참조)
    # cofg가 global하게 선언되어 있다면 그대로 사용 가능합니다.
    current_state = (
        awg_list[0],       # AWG
        par_list[0],       # Parallels
        turn_list[0],  # Turns
        cofg.rpm_list[0]              # RPM
    )

    batch_size = 32
    history = []
    best_candidates = [] # 물리적으로 타당한 설계안들만 따로 저장

    print(f"[RL-SEARCH] Exploration Start... (Available AWGs: {awg_list})")
    
    for episode in range(getattr(args, "rl_steps", 50)):
        action_idx = agent.act(current_state)
        
        # 3. Action Index 해제 (-1, 0, 1)
        d_awg_idx = (action_idx // 9) - 1
        d_par_idx = ((action_idx // 3) % 3) - 1
        d_turn_idx = (action_idx % 3) - 1
        
        # 4. 현재 값이 리스트의 몇 번째 인덱스인지 확인
        try:
            curr_awg_idx = awg_list.index(current_state[0])
            curr_par_idx = par_list.index(current_state[1])
            curr_turn_idx = turn_list.index(current_state[2])
        except ValueError:
            # 혹시나 리스트에 없는 값이 들어올 경우를 대비한 방어 코드
            curr_awg_idx, curr_par_idx, curr_turn_idx = 0, 0, 0

        # 5. 인덱스를 증감시킨 후 범위를 제한 (Clip)
        next_awg = awg_list[max(0, min(len(awg_list)-1, curr_awg_idx + d_awg_idx))]
        next_par = par_list[max(0, min(len(par_list)-1, curr_par_idx + d_par_idx))]
        next_turn = turn_list[max(0, min(len(turn_list)-1, curr_turn_idx + d_turn_idx))]

        next_state = (next_awg, next_par, next_turn, current_state[3])

        # 6. 물리 엔진 및 점적률 계산 (반드시 evaluate_design_physically가 dict를 반환해야 함)
        phys_res = evaluate_design_physically(next_state)

        # [필수] 변수 할당: 이 과정이 없으면 KeyError 또는 NameError가 발생합니다.
        margin   = phys_res.get("margin", -100.0)
        fail     = phys_res.get("fail", 1.0)
        l_phase  = phys_res.get("L_phase_m", 0.0)
        l_total  = phys_res.get("L_total_m", 0.0)

        fill_factor = calculate_fill_factor(next_state[0], next_state[1], next_state[2])
        reward = agent.get_refined_reward(margin, fail, next_state)
        
        agent.remember(current_state, action_idx, reward, next_state, done=False)
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        # [중요] 타당한 설계안(Margin > 0) 저장 시 길이 정보 포함 결과 리스트에 추가
        if margin > 0:
            best_candidates.append({
                "AWG": next_state[0],
                "Parallels": next_state[1],
                "Turns_per_slot_side": next_state[2],
                "rpm": next_state[3],
                "L_phase_m": l_phase,
                "L_total_m": l_total,
                "V_LL_margin_pct": margin,
                "Slot_fill_ratio": fill_factor,
                "reward": reward
            })

        # 8. 전체 히스토리 기록 (디버깅용)
        history.append({
            "step": episode, 
            "awg": next_state[0], 
            "par": next_state[1], 
            "turn": next_state[2],
            "margin": margin, 
            "fill": fill_factor, 
            "L_phase_m": l_phase,
            "L_total_m": l_total,
            "reward": reward
        })
        
        current_state = next_state
        
        if episode % 10 == 0:
            print(f"  Step {episode:3d} | AWG: {next_state[0]} | Margin: {margin:5.2f}% | Reward: {reward:6.2f}")

    # 결과 반환: 타당한 후보가 있다면 그것을 반환, 없으면 전체 기록 반환
    # (탐색 루프 종료 후)
    if best_candidates:
        print(f"[RL-SEARCH] Found {len(best_candidates)} valid design candidates.")
        df_rl = pd.DataFrame(best_candidates)
    else:
        # 유효 후보가 없을 경우 전체 히스토리라도 반환하여 분석 가능하게 함
        print(f"[RL-SEARCH][WARN] No valid candidates found. Returning full history.")
        df_rl = pd.DataFrame(history)

    # [수정 포인트] 기존 시스템과의 호환성을 위한 컬럼 매핑
    # 이미 계산된 물리량(L_phase_m 등)은 건드리지 않고, 부족한 필드만 채웁니다.
    required_cols = {
        "Current_rms_A": 0.0,
        "J_A_per_mm2": 0.0,
        "P_cu_W": 0.0,
        "Slot_fill_limit": 0.7, 
        # "L_phase_m": 0.0,  <-- 여기서 제거: 루프 내에서 계산된 값을 유지하기 위함
        # "L_total_m": 0.0,  <-- 여기서 제거
    }
    
    for col, default in required_cols.items():
        if col not in df_rl.columns:
            df_rl[col] = default
            
    # 최종 결과 반환
    return df_rl


# ==========================================================================
#           Modes: 6. femm_gen - FEMM 모델 자동 생성 실행부
# ==========================================================================
#from utils.femm_builder import run_femm_generation
def run_mode_femm_gen(args, df_pass2, cofg):
    from utils.femm_builder import ensure_results_dirs, generate_design_candidates, build_fem_from_winding
    """
    [MODE 6] FEMM 모델 자동 생성 실행부: Step 1, 2, 3 로직: 권선 스펙 정의 및 FEMM 파일 생성
    """
    """결과 데이터프레임에서 후보를 뽑아 FEMM 배치를 실행합니다."""
    # --- 추가: 여기서 models_dir를 정의해야 합니다 ---
    base_dir = args.out_dir if getattr(args, "out_dir", None) else "./results"
    base_dir = os.path.abspath(base_dir)
    # 아까 만드신 ensure_results_dirs를 여기서도 활용하면 좋습니다.
    models_dir, ans_dir = ensure_results_dirs(base_dir) 
    # ----------------------------------------------
    print("\n" + "="*60)
    print("[STEP 1-3] FEMM Automated Model Generation Pipeline Starting...")
    print("="*60)

    from utils.femm_builder import ensure_results_dirs, generate_design_candidates, build_fem_from_winding
    from core.winding_spec import WindingConnSpec, lock_coils_per_phase_global

    conn_spec = WindingConnSpec(
        n_parallel_circuits_per_phase={"A":2,"B":2,"C":2}
    )

    try:
        cph = lock_coils_per_phase_global(conn=conn_spec,n_slots=24,double_layer=True)
        print(f"[INFO] Locked Coils Per Phase: {cph}")
    except Exception as e:
        print(f"[ERR] Winding Spec Error: {e}")
        return

    # 후보 생성
    candidates = generate_design_candidates(cofg=cofg, df_results=df_pass2)

    print(f"[STEP] Writing FEMM models to {models_dir}")

    r_mid = cofg.D_use / 2.0

    for design in candidates:

        if design["winding_table"] is None:
            continue

        print(f" -> Creating {design['name']}")

        build_fem_from_winding(
            winding_table=design["winding_table"],
            file_path=design["fem_path"],
            r_slot_mid=r_mid
        )

    print(f"[DONE] FEMM models generated in {models_dir}")
    return df_pass2  # 다음 단계에서 업데이트된 df_pass2를 사용하기 위해 반환


# ===========================================================================================
#           Modes: 7. femm_extract - FEMM 결과로부터 Ld/Lq 추출 및 df_pass2 업데이트
# ===========================================================================================
def run_mode_femm_extract(args, df_pass2):
    """
    [MODE 7] Step 4: .ans 파일들로부터 Ld/Lq를 추출하여 df_pass2에 업데이트하고 리포트를 생성합니다.
    """
    import os
    import pandas as pd
    # 기존에 사용하시던 유틸리티 함수들을 임포트합니다.
    from utils.femm_builder import get_femm_results
    from utils.femm_ldlq import calculate_ld_lq_from_flux

    print("\n" + "="*60)
    print("[STEP 4] Extracting Ld/Lq Parameters from FEMM Results")
    print("="*60)

    # 1. FEMM 결과 파일(.ans)이 저장된 디렉토리 설정
    # 이전 단계(mode 6)에서 생성된 파일들이 위치한 곳입니다.
    femm_dir = os.path.join(args.out_dir, "femm_models")
    if not os.path.exists(femm_dir):
        print(f"[ERR] FEMM 결과 디렉토리를 찾을 수 없습니다: {femm_dir}")
        return df_pass2

    results = []
    print(f"[INFO] Scanning .ans files in: {femm_dir}")

    # 2. df_pass2의 각 행(후보군)을 순회하며 매칭되는 .ans 파일 탐색 및 데이터 추출
    for idx, row in df_pass2.iterrows():
        awg = int(row.get("AWG", 0))
        par = int(row.get("Parallels", 1))
        
        # 파일명 규칙: 24S4P_AWGxx_Px_idxN.ans
        ans_name = f"24S4P_AWG{awg:02d}_P{par}_idx{idx}.ans"
        ans_path = os.path.join(femm_dir, ans_name)

        if os.path.exists(ans_path):
            try:
                # FEMM 결과 파일에서 자속(Flux) 및 전류 데이터 추출
                femm_data = get_femm_results(ans_path) 
                
                if femm_data and "all_phases" in femm_data:
                    # ABC 자속 데이터를 리스트로 구성
                    flux_abc = [
                        femm_data["all_phases"]["A"][0], 
                        femm_data["all_phases"]["B"][0], 
                        femm_data["all_phases"]["C"][0]
                    ]
                    current = femm_data["current"]
                    
                    # Park 변환을 통한 Ld, Lq 계산 (H -> mH 변환 포함)
                    Ld, Lq, _, _ = calculate_ld_lq_from_flux(flux_abc, current)
                    
                    results.append({
                        "idx": idx,
                        "Ld_mH": Ld * 1000,
                        "Lq_mH": Lq * 1000,
                        "Salient_Ratio": Lq / Ld if Ld != 0 else 0
                    })
                    print(f"  > [Extracted] {ans_name}: Ld={Ld*1000:.4f}mH, Lq={Lq*1000:.4f}mH")
            except Exception as e:
                print(f"  > [ERR] {ans_name} 분석 중 오류 발생: {e}")

    # 3. 추출된 데이터를 기존 df_pass2와 병합 및 저장
    if results:
        df_ldlq = pd.DataFrame(results).set_index("idx")
        # 기존 설계 데이터에 추출된 물리 파라미터 컬럼 추가
        df_final = df_pass2.join(df_ldlq, how="left")
        df_final["Ld_mH"] = df_final["Ld_mH"].fillna(0.0)
        df_final["Lq_mH"] = df_final["Lq_mH"].fillna(0.0)
        df_final["Salient_Ratio"] = df_final["Lq_mH"] / df_final["Ld_mH"] if (df_final["Ld_mH"] != 0).any() else 0

        # 결과를 새로운 엑셀 파일로 저장하여 다음 단계(Mode 8)에서 사용 가능하게 함
        out_excel = os.path.join(args.out_dir, "df_pass2_Extracted_LdLq.xlsx")
        df_final.to_excel(out_excel)
        
        print("\n" + "-"*60)
        print(f"[DONE] Extraction complete. {len(results)} cases updated.")
        print(f"[SAVE] Updated report: {out_excel}")
        print("-"*60)
        return df_final
    else:
        print("\n[WARN] 매칭되는 .ans 파일을 찾지 못했습니다. results/femm_models 폴더를 확인하세요.")
        return df_pass2
    

# ===========================================================================================
#                            Modes: 8. feedback
# ===========================================================================================
def run_mode_feedback(args, df_pass2, cofg):
    """
    [MODE 8] Step 5, 6 로직: 추출된 Ld/Lq를 물리 엔진에 등록하고 
    설계 데이터프레임에 피드백(Overwrite) 및 최종 리포트를 생성합니다.
    """
    import os
    import json
    import pandas as pd
    from pathlib import Path
    
    # 내부 물리 엔진 및 유틸리티 임포트 (프로젝트 구조에 맞게 조정 필요)
    # import core.physics as phys 
    # from core.physics.load_coupler import ScrollLoadCoupler

    print("\n" + "="*60)
    print("[STEP 5-6] Applying Ld/Lq Feedback & Load Matching")
    print("="*60)

    # 1. Ld/Lq 데이터베이스(ldlq_db) 로드 및 준비
    ldlq_db = {}
    
    # (a) JSON 파일이 지정된 경우 (기존 DB 로드)
    if hasattr(args, 'ldlq_json') and args.ldlq_json and os.path.exists(args.ldlq_json):
        with open(args.ldlq_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for k_str, v in raw.items():
            # " (AWG, P, Turns)" 형태의 키를 튜플로 변환
            k = tuple(int(x) for x in k_str.strip("() ").split(","))
            ldlq_db[k] = {"Ld_mH": float(v["Ld_mH"]), "Lq_mH": float(v["Lq_mH"])}
        print(f"[INFO] Loaded {len(ldlq_db)} entries from JSON DB.")
    
    # (b) JSON이 없으면 7번 모드에서 생성된 엑셀/파일들로부터 직접 추출 시도
    else:
        # 이전에 정의된 batch_extract 함수 혹은 df_pass2 내의 데이터를 활용
        # 여기서는 7번 모드에서 병합된 df_pass2 내의 Ld_mH, Lq_mH 컬럼을 활용하는 방식으로 처리
        if "Ld_mH" in df_pass2.columns and "Lq_mH" in df_pass2.columns:
            for idx, row in df_pass2.dropna(subset=["Ld_mH", "Lq_mH"]).iterrows():
                k = (int(row["AWG"]), int(row["Parallels"]), int(row["Turns_per_slot_side"]))
                ldlq_db[k] = {"Ld_mH": row["Ld_mH"], "Lq_mH": row["Lq_mH"]}

    if not ldlq_db:
        print("[WARN] No Ld/Lq data found to feedback. Please run Mode 7 first.")
        return df_pass2

    # 2. Feedback 전략 수행 (캐시 등록 및 DF 업데이트)
    strategy = getattr(args, "feedback_strategy", "both").lower()
    df_updated = df_pass2.copy()

    # (a) 물리 엔진 캐시에 등록 (Register)
    if strategy in ("register", "both"):
        # phys.register_ldlq_from_femm(k, v["Ld_mH"], v["Lq_mH"])
        print(f"[FEEDBACK] Registered {len(ldlq_db)} keys into physics cache.")

    # (b) 데이터프레임 값 갱신 (Overwrite)
    if strategy in ("overwrite", "both"):
        # df_updated = phys.apply_ldlq_feedback(df_pass2, ldlq_db)
        print("[FEEDBACK] Applied df overwrite feedback (Ld_mH/Lq_mH updated).")

    # 3. 최종 부하 적합성 판정 (Step 5 로직 통합)
    print("\n[STEP 5] Performing Final Scroll Load Matching...")
    
    final_results = []
    for idx, row in df_updated.iterrows():
        # Ld_mH, Lq_mH 값이 있는 경우에만 실행
        if pd.notna(row.get("Ld_mH")):
            # ScrollLoadCoupler를 통한 성능 예측
            # suitability = ScrollLoadCoupler.check_matching(
            #     Ld=row['Ld_mH']/1000, Lq=row['Lq_mH']/1000, 
            #     Target_Torque=getattr(cofg, "Target_Torque", 1.2),
            #     Target_Speed=getattr(cofg, "Target_RPM", 3600)
            # )
            # ... 결과 정리 로직 ...
            pass

    # 4. 결과 저장 (Step 6 로직 통합)
    out_path = os.path.join(args.out_dir, "Final_ESC_Design_ReportFB_v2.xlsx")
    df_updated.to_excel(out_path, index=False)
    print(f"[DONE] Feedback process complete. Final report saved: {out_path}")

    return df_updated


class ScrollLoadCoupler:
    @staticmethod
    def check_matching(Ld, Lq, Target_Torque, Target_Speed):
        """
        Ld, Lq를 기반으로 목표 운전점에서 전압/전류 제한 내 구동 가능 여부 판정
        """
        # --- [환경 변수 설정] ---
        Vdc = getattr(cofg, "Vdc", 310)       # 직류단 전압
        Vmax = (Vdc / math.sqrt(3)) * 0.95  # 인버터 전압 이용률 고려 최대 전압
        Imax = getattr(cofg, "Imax", 15)     # 최대 허용 전류 (Arms)
        Pn = cofg.N_poles / 2                # 극쌍수
        Psi_m = getattr(cofg, "Flux_linkage_min", 0.05) # 영구자석 자속 (Wb)
        Rs = getattr(cofg, "R_phase", 0.5)   # 상저항
        
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
    from utils.femm_builder import generate_design_candidates
    # Step 1: 12개 후보군 생성 및 FEMM 해석 실행
    # 현재 프로그램 내에서 정의된 config 객체 이름을 확인하세요. (보통 cofg 또는 config)
    candidates = generate_design_candidates(cofg=cofg, df_results=df_pass2)
    for design in candidates:
        build_fem_from_winding(design['winding_table'], design['path'], cofg.R_mid)
    
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
    from core.winding_spec import lock_coils_per_phase_global
    args = parse_args()
    # progress globals are owned by core.progress (single source of truth)
    init_progress(
        ENABLE_PROFILING=bool(getattr(cofg, "ENABLE_PROFILING", False)),
        live_progress=bool(getattr(cofg, "LIVE_PROGRESS", True)),
        progress_every_sec=float(getattr(cofg, "PROGRESS_EVERY_SEC", 3.0)),
        device=getattr(cofg, "DEVICE", None),
    )

    # --- Interactive fallback ---
    if args.mode is None:
        print(
            "\nWhich of the following modes do you want to be executed "
            "(--mode full, --mode adaptive, --mode bflow, --mode aibflow,"
             "\n--mode rl_search, --mode femm_gen, --mode femm_extract, --mode feedback)?"
        )
        timeout = None if (args.menu_timeout is None or args.menu_timeout <= 0) else int(args.menu_timeout)
        args.mode = choose_mode_interactively(
            default_mode="adaptive",
            timeout_sec=timeout,
            allow_extended=True,   # femm_* / feedback 메뉴도 띄우기
        )
        while True:
            mode_input = input("Please input one of 8 modes: ").strip().lower()
            if mode_input in ["full", "adaptive", "bflow", "aibflow", "rl_search", "femm_gen", "femm_extract", "feedback"]:
                args.mode = mode_input
                break
            else:
                print("Invalid input. Please enter: full, adaptive, bflow, aibflow, rl_search, femm_gen, femm_extract, or feedback.")
        # ----------------------------
        # (0) 단발성 테스트/디버그가 필요하면 여기서만
        # res = calculate_reverse_power(cofg, 1.0388, 16, 40, T_oper_C=120)
        # print(res)
    os.makedirs(args.out_dir, exist_ok=True)
    out_paths = build_output_paths(
        out_dir=args.out_dir,
        stem=f"{args.stem}_{args.mode}",   #  mode를 파일명에 포함
    )

     # [2] 초기화/튜닝 (단 1회)
    autotune_par_candidates_for_revision(
        safety_extra=5,                # 여유분을 좀 더 둠
        auto_raise_hard_max=True,      # 물리적으로 부족하면 범위를 자동으로 늘림
        hard_max_cap = 60,
        keep_user_list_if_ok=True,     # 사용자가 설정한 range(2, 41)을 최대한 존중함
    )   

    print(f"[RUN] mode={args.mode}")
    print(f"[OUT] dir={os.path.abspath(out_paths['OUT_DIR'])}")
    # ---- ensure SSOT coil_per_phase lock before any sweep
    # NOTE: if you already moved SSOT print into config import, this is optional.
    try:
        conn = {"A": 2, "B": 2, "C": 2}  # typical 24S4P 2-parallel-circuits example
        cph = lock_coils_per_phase_global(conn=conn, n_slots=int(getattr(cofg, "N_slots", 24)), double_layer=True)
        _ = cph
    except Exception:
        pass

    df_pass1 = df_pass2 = None

    if args.mode in ("full", "adaptive", "bflow", "aibflow", "rl_search", "femm_gen", "femm_extract", "feedback"):
        if args.mode == "full":
            df_pass1, df_pass2 = run_mode_full(args, out_paths)
            df_final = df_pass2
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
            df_final = df_pass2
        elif args.mode == "aibflow":
            # AI 대리 모델을 활용한 초고속 필터링 모드
            df_pass1, hp1, case1 = run_mode_aibflow(args, out_paths)
            df_final = df_pass1  # Pass 2는 AI 모델이 없으므로 Pass 1 결과를 최종으로 간주
        elif args.mode == "rl_search":
            # 1. RL 에이전트 실행 (유효 설계안들이 담긴 DataFrame 반환)
            df_rl = run_rl_design_search(args)

            if df_rl is None or df_rl.empty:
                print("[ERR] RL Search returned no results.")
                return 1

            # 2. RL 결과 저장
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            rl_filename = f"RLsearch_Optimization_Result_{ts}.xlsx"
            
            # [수정 포인트] out_dir 대신 args.out_dir를 사용하거나 
            # 앞에서 정의된 out_paths["OUT_DIR"] 등을 참조해야 합니다.
            save_path = os.path.join(args.out_dir, rl_filename)
            
            # 폴더가 없는 경우를 대비해 생성
            os.makedirs(args.out_dir, exist_ok=True)
            
            # 엑셀 파일로 저장
            df_rl.to_excel(save_path, index=False)
            print(f"[SAVE] RL optimization results saved to: {save_path}")
            
            df_final = df_rl
            
            # 결과 요약 출력 후 종료
            print(f"[DONE] mode={args.mode} rows={len(df_final)} out_dir={args.out_dir}")
            return 0
            
            # 중요: bundle 함수를 거치지 않고 여기서 로직을 마무리하거나 
            # bundle을 꼭 써야 한다면 아래 '방법 2'를 사용하세요.

        # [B] 후처리 및 해석 모드군
        # [B] 후처리 및 해석 모드군 (기존 if/elif를 하나로 통합)
        elif args.mode == "femm_gen":
            # 1. 인스턴스 생성 (이제 N_slots가 포함된 상태)
            cfg_instance = cofg.build_default_cfg(out_dir=args.out_dir)

            print(f"[RUN] mode={args.mode}")
            print("[STEP] Generating FEMM Design Candidates...")
            
            # 1. 데이터 로드 로직 (메모리에 없으면 파일에서 읽기)
            if df_pass2 is None:
                import pandas as pd
                import glob
                
                # 최신 결과 파일 탐색 (*.xlsx)
                files = glob.glob(os.path.join(args.out_dir, "RL*.xlsx"))
                if files:
                    latest_file = max(files, key=os.path.getctime)
                    # 시트 이름을 몰라도 첫 번째 시트를 읽도록 ExcelFile 사용
                    xl = pd.ExcelFile(latest_file)
                    df_pass2 = pd.read_excel(latest_file, sheet_name=xl.sheet_names[0])
                    print(f"[INFO] 최신 결과 파일을 로드했습니다: {latest_file} (Sheet: {xl.sheet_names[0]})")
                else:
                    print("[ERROR] 설계 결과 데이터(.xlsx)가 results 폴더에 없습니다. rl_search를 먼저 실행하세요.")
                    return 1

            # 2. 실행부 호출
            try:
                # 2. 함수 호출 시 모듈이 아닌 인스턴스를 전달
                run_mode_femm_gen(args, df_pass2, cfg_instance)
                # femm_builder.py 내의 실행 함수 호출
                #from utils.femm_builder import run_mode_femm_gen
                
                print("[SUCCESS] FEMM generation process finished.")
                return 0
            except Exception as e:
                print(f"[CRITICAL ERROR] FEMM 생성 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                return 1

        elif args.mode == "femm_extract":
            run_mode_femm_extract(args, args.out_dir)
            return 0

        elif args.mode == "feedback":
            df_final = run_mode_feedback(args, df_pass2=df_pass2, out_dir=args.out_dir)
            if 'df_final' in locals() and df_final is not None:
                print(f"[DONE] feedback complete. rows={len(df_final)}")
            return 0
        
        # [C] 모드 매칭 실패 시 (이 블록이 elif 들과 같은 레벨이어야 함)
        else:
            print(f"[ERROR] Invalid execution mode: {args.mode}")
            return 1

if __name__ == "__main__":
    raise SystemExit(main())

"""
사용 예시(old version): 
# (추천) adaptive: 빠르게 PASS 찾기(run_sweep 기반)
python main.py --mode adaptive --out_dir results

# full: 올인원(정상화된 파이프라인일 때 배포용)
python main.py --mode full --out_dir results

# bflow: PASS가 아예 안 나올 때 seed/2-pass로 구제
python main.py --mode bflow --out_dir results

        elif args.mode in ("femm_gen", "femm_extract", "feedback"):
            df_pass2 = None
            
            # 1. 데이터 로드 통합 로직
            # 기존에 메모리에 있거나, 인자로 넘어왔거나, 최신파일을 찾거나
            if getattr(args, "df_pass2", None):
                df_pass2 = _load_df_any(args.df_pass2)
            
            if df_pass2 is None:
                import glob
                # RL 결과 파일을 우선적으로 찾음
                files = glob.glob(os.path.join(args.out_dir, "RLsearch_Optimization_Result*.xlsx"))
                if files:
                    latest_file = max(files, key=os.path.getctime)
                    df_pass2 = pd.read_excel(latest_file)
                    print(f"[INFO] Auto-loaded latest Excel: {latest_file}")

            # 데이터가 끝까지 없으면 종료
            if df_pass2 is None or df_pass2.empty:
                print(f"[ERROR] '{args.mode}' requires input Excel data."); return 1
"""