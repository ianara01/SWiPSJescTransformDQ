# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 12:14:24 2026

@author: USER, SANG JIN PARK

core.progress

목표
- import 시점에 부작용(프린트/torch warmup/전역 카운터 초기화)을 없앰
- 진행률/프로파일 집계용 전역 dict(기존 코드 호환) + 명시적 init 1회 호출로 통일
- engine/utils/main 어디서든 가볍게 import 가능 (순환 import 최소화)

사용(권장)
    from core.progress import init_progress
    init_progress()   # main.py에서 1줄

그리고 engine 쪽에서는 필요할 때:
    from core.progress import GEO, PROF, PROG2, progress_add_tiles, progress_tick_tile, progress_update, progress_newline
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


# =========================
# Public globals (compat)
# =========================

# case / geometry / tile 진행 상태
GEO: Dict[str, int] = {
    "case_idx": 0,
    "case_total": 0,
    "geo_steps": 0,   # geometry 조합 수 (stack×slot_pitch×end_factor×span×MLT_scale ...)
    "tile_hits": 0,   # 실제 타일 루프에 진입한 횟수
}

# profiling 집계
PROF: Dict[str, Any] = {
    "combos_evaluated": 0,    # 평가한 격자 원소 수 (rows*cols)
    "gpu_ms_mask": 0.0,       # grid+mask(핵심 계산) GPU 시간(ms)
    "gpu_ms_collect": 0.0,    # 결과 수집/전송 GPU 시간(ms)
    "start_wall": None,       # 전체 wall-clock 시작
    "combos_planned": 0,
}

# 타일 진행 카운터
PROG2: Dict[str, int] = {"tiles_total": 0, "tiles_done": 0}


# =========================
# Internal init state
# =========================

@dataclass
class _ProgressInitState:
    initialized: bool = False
    ENABLE_PROFILING: bool = False   # backward compat
    enable_profiling: bool = False   # new/alias (accept enable_profiling=...)
    live_progress: bool = True
    progress_every_sec: float = 3.0
    device: Any = None  # torch.device or None
    _last_print: float = 0.0


_STATE = _ProgressInitState()


# ==================================
# Small helpers (no torch import)
# ==================================

def progress_add_tiles(n: int) -> None:
    """planned 타일 수 누적"""
    try:
        PROG2["tiles_total"] += int(n)
    except Exception:
        pass


def progress_tick_tile(n: int = 1) -> None:
    """완료 타일 수 누적"""
    try:
        PROG2["tiles_done"] += int(n)
    except Exception:
        pass


def progress_newline() -> None:
    """LIVE 출력 줄바꿈"""
    try:
        print("", flush=True)
    except Exception:
        pass

def progress_update(
    *,
    funnel: Optional[Dict[str, Any]] = None,
    tag: Optional[str] = None,
    prefix: str = "[PROG]",
    force: bool = False,
) -> None:
    """
    간단 진행률 출력.
    - LIVE_PROGRESS가 False면 아무 것도 출력하지 않습니다.
    - engine 루프에서 자주 호출되어도 PROGRESS_EVERY_SEC 간격으로만 출력합니다.
    - pass 카운트는 `funnel` dict를 인자로 받아 표시합니다.
    """
    if not _STATE.live_progress:
        return

    now = time.perf_counter()
    if (not force) and (now - _STATE._last_print) < float(_STATE.progress_every_sec):
        return
    _STATE._last_print = now

    # ---------------------------
    # GEO
    # ---------------------------
    ci = int(GEO.get("case_idx", 0) or 0)
    ct = int(GEO.get("case_total", 0) or 0)

    # case_total이 0이면 표시용 보정
    if ct <= 0:
        case_str = "0/0"
    else:
        case_str = f"{ci+1}/{ct}"

    gs = int(GEO.get("geo_steps", 0) or 0)
    th = int(GEO.get("tile_hits", 0) or 0)

    # ---------------------------
    # Tiles
    # ---------------------------
    td = int(PROG2.get("tiles_done", 0) or 0)
    tt = int(PROG2.get("tiles_total", 0) or 0)

    if tt <= 0:
        tile_str = "0/0"
    else:
        tile_str = f"{td:,}/{tt:,}"

    # ---------------------------
    # wall time
    # ---------------------------
    t0 = PROF.get("start_wall", None)
    if isinstance(t0, (int, float)):
        dt = now - float(t0)
        dt_s = f"{dt:,.1f}s"
    else:
        dt_s = "-"

    # ---------------------------
    # pass count
    # ---------------------------
    passes = 0
    if isinstance(funnel, dict):
        try:
            passes = int(funnel.get("pass_all", 0) or 0)
        except Exception:
            passes = 0

    # ---------------------------
    # 메시지 구성
    # ---------------------------
    msg = (
        f"{prefix} case {case_str} "
        f"| geo={gs:,} "
        f"| tiles {tile_str} "
        f"| hits={th:,} "
        f"| pass={passes:,} "
        f"| wall={dt_s}"
    )

    if tag:
        msg += f" | {tag}"

    print(msg, flush=True)


# =====================================
# Profiling warmup (lazy torch import)
# =====================================

def _torch_warmup(device: Any) -> None:
    """프로파일링 ON일 때만: matmul warm-up + GPU 이벤트 타이밍 누적."""
    # torch는 heavy import라 여기서만 불러옵니다.
    import torch  # type: ignore

    from utils.utils import print_gpu_banner  # local import to avoid cycles

    print_gpu_banner()
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    w0 = time.perf_counter()
    a = torch.rand(1024, 1024, device=device)
    b = torch.rand(1024, 1024, device=device)

    if getattr(device, "type", "cpu") == "cuda":
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()

    _ = a @ b

    if getattr(device, "type", "cpu") == "cuda":
        e1.record()
        torch.cuda.synchronize()
        PROF["gpu_ms_mask"] += float(e0.elapsed_time(e1))
        print(f"[GPU] Warm-up: {e0.elapsed_time(e1)/1000.0:.3f}s", flush=True)
    else:
        print(f"[CPU] Warm-up: {time.perf_counter()-w0:.3f}s", flush=True)


# =====================================
# Public init (main.py one-liner)
# =====================================

def init_progress(
    *,
    enable_profiling: Optional[bool] = None,
    ENABLE_PROFILING: Optional[bool] = None,  # backward compat
    live_progress: Optional[bool] = None,
    progress_every_sec: Optional[float] = None,
    device: Any = None,
    reset_counters: bool = False,
) -> None:
    """진행/프로파일 전역을 초기화.

    - 기본값은 configs.config(있으면)에서 읽고, 없으면 보수적 default를 씁니다.
    - import 시 실행하지 않고, main.py에서 '한 번만' 호출하도록 설계했습니다.

    Parameters
    ----------
    ENABLE_PROFILING : bool | None
        None이면 configs.config.ENABLE_PROFILING(있으면)을 사용
    live_progress : bool | None
        None이면 configs.config.LIVE_PROGRESS(있으면)을 사용
    progress_every_sec : float | None
        None이면 configs.config.PROGRESS_EVERY_SEC(있으면)을 사용
    device : Any
        torch.device. None이면 configs.config.DEVICE(있으면)을 사용
    reset_counters : bool
        True면 GEO/PROF/PROG2 카운터를 0으로 초기화
    """
    if _STATE.initialized and not reset_counters and enable_profiling is None and ENABLE_PROFILING is None and live_progress is None and progress_every_sec is None and device is None:
        # 이미 init 되었고, 아무 override도 없으면 재호출은 no-op
        return

    # ---- defaults from configs.config (optional) ----
    try:
        import configs.config as C  # type: ignore
    except Exception:
        C = None  # type: ignore

    # accept both enable_profiling=... and ENABLE_PROFILING=...
    if ENABLE_PROFILING is None:
        ENABLE_PROFILING = enable_profiling
    if ENABLE_PROFILING is None:
        ENABLE_PROFILING = bool(getattr(C, "ENABLE_PROFILING", False)) if C else False
    if live_progress is None:
        live_progress = bool(getattr(C, "LIVE_PROGRESS", True)) if C else True
    if progress_every_sec is None:
        progress_every_sec = float(getattr(C, "PROGRESS_EVERY_SEC", 3.0)) if C else 3.0
    if device is None and C is not None:
        device = getattr(C, "DEVICE", None)

    _STATE.ENABLE_PROFILING = bool(ENABLE_PROFILING)     # legacy
    _STATE.enable_profiling = bool(ENABLE_PROFILING)     # canonical
    _STATE.live_progress = bool(live_progress)
    _STATE.progress_every_sec = float(progress_every_sec) if progress_every_sec is not None else 3.0
    _STATE.device = device

    if reset_counters:
        GEO.update({"case_idx": 0, "case_total": 0, "geo_steps": 0, "tile_hits": 0})
        PROF.update({"combos_evaluated": 0, "gpu_ms_mask": 0.0, "gpu_ms_collect": 0.0, "start_wall": None, "combos_planned": 0})
        PROG2.update({"tiles_total": 0, "tiles_done": 0})

    # wall start
    PROF["start_wall"] = time.perf_counter()
    _STATE._last_print = 0.0
    _STATE.initialized = True

    # warmup only if profiling
    if _STATE.enable_profiling:
        if _STATE.device is None:
            # torch.device가 없더라도 CPU warmup는 가능하니 torch.device("cpu")로
            try:
                import torch  # type: ignore
                _STATE.device = torch.device("cpu")
            except Exception:
                _STATE.device = None
        if _STATE.device is not None:
            try:
                _torch_warmup(_STATE.device)
            except Exception as e:
                print(f"[PROG][WARN] torch warmup failed: {e}", flush=True)
