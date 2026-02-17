# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 10:24:51 2026

@author: USER, SANG JIN PARK
"""
from dataclasses import dataclass
from typing import Dict, Optional

# ==================== WindingConnSpec: 결선 스펙 ====================
@dataclass(frozen=True)
class WindingConnSpec:
    """
    결선 스펙 (coils_per_phase를 '상당 직렬 코일 수'로 쓸 때 필요한 최소 정보)

    n_parallel_circuits_per_phase:
      - 상(A/B/C) 각각의 병렬 회로(경로) 개수
      - 예) 24S-4P에서 A상이 2병렬 경로면 A:2
    """
    n_parallel_circuits_per_phase: Dict[str, int]  # {"A":2, "B":2, "C":2}
    
def infer_coils_per_phase_24S4P(
    *,
    n_slots: int = 24,
    n_phases: int = 3,
    double_layer: bool = True,
    conn: Optional[WindingConnSpec] = None,
    # 기대하는 형태로 강제 고정(선호값)
    prefer_equal_parallel: bool = True,
) -> Dict[str, int]:
    """
    상별 coils_per_phase(= 상당 직렬 코일 수, '한 상의 한 병렬 경로 기준')를 산정/검증.
    반환 예: {"A":4, "B":4, "C":4} 또는 {"A":8,"B":8,"C":8}

    주의:
      - 여기서 coils_per_phase는 '물리 코일 수'가 아니라
        Nphase = Turns_per_slot_side * coils_per_phase 에서 쓰는 '직렬 누적 개수'로 정의.
      - 결선(병렬 경로 수)이 들어오면 그에 맞춰 자동 산정.
    """
    if n_slots <= 0 or n_phases <= 0:
        raise ValueError("n_slots and n_phases must be positive.")

    # double-layer의 전형적 분포권에서 코일 수는 슬롯수로 보는 경우가 많음.
    # single-layer/특수권선이면 달라질 수 있으니 double_layer False면 보수적으로 처리.
    if double_layer:
        n_coils_total = n_slots
    else:
        # single-layer는 슬롯당 coil-side 1개 → 총 side = S → coil= S/2 가 흔함(권선에 따라 다름)
        if n_slots % 2 != 0:
            raise ValueError("single-layer assumed requires even n_slots for coil pairing.")
        n_coils_total = n_slots // 2

    # 상별 "전체 코일 수" (병렬 경로 고려 전)
    if n_coils_total % n_phases != 0:
        raise ValueError(f"Total coils {n_coils_total} not divisible by phases {n_phases}. "
                         "Winding distribution/definition mismatch.")

    coils_per_phase_total = n_coils_total // n_phases  # 24S double-layer면 24/3 = 8

    # 결선(병렬 경로 수) 반영
    if conn is None:
        # 결선 정보 없으면 '병렬 경로 1'로 가정 → 8 (24S double-layer)
        return {"A": coils_per_phase_total, "B": coils_per_phase_total, "C": coils_per_phase_total}

    # 병렬 경로 수 체크
    for ph in ("A", "B", "C"):
        if ph not in conn.n_parallel_circuits_per_phase:
            raise KeyError(f"Missing parallel circuit count for phase {ph}.")
        k = int(conn.n_parallel_circuits_per_phase[ph])
        if k <= 0:
            raise ValueError(f"Invalid n_parallel_circuits for phase {ph}: {k}")

    if prefer_equal_parallel:
        kA = conn.n_parallel_circuits_per_phase["A"]
        kB = conn.n_parallel_circuits_per_phase["B"]
        kC = conn.n_parallel_circuits_per_phase["C"]
        if not (kA == kB == kC):
            raise ValueError(f"Unequal parallel circuits per phase: A={kA}, B={kB}, C={kC} "
                             "-> global coils_per_phase 단일값 사용 시 흔들릴 수 있음. "
                             "상별로 따로 쓰거나 결선을 동일하게 맞추세요.")

    out = {}
    for ph, k in conn.n_parallel_circuits_per_phase.items():
        if coils_per_phase_total % k != 0:
            raise ValueError(
                f"Phase {ph}: coils_per_phase_total={coils_per_phase_total} not divisible by "
                f"n_parallel_circuits={k}. 결선/코일그룹 정의를 확인하세요."
            )
        out[ph] = coils_per_phase_total // k  # '한 병렬 경로'에 직렬로 들어가는 코일 수

    return out

def lock_coils_per_phase_global(
    *,
    conn: Optional[WindingConnSpec] = None,
    n_slots: int = 24,
    n_phases: int = 3,
    double_layer: bool = True,
) -> int:
    """
    코드가 전역 coils_per_phase(단일 스칼라)를 쓰는 구조라면,
    A/B/C가 동일한 값으로 산정될 때만 고정하고 반환.
    """
    d = infer_coils_per_phase_24S4P(
        n_slots=n_slots,
        n_phases=n_phases,
        double_layer=double_layer,
        conn=conn,
        prefer_equal_parallel=True,
    )
    v = {d["A"], d["B"], d["C"]}
    if len(v) != 1:
        raise ValueError(f"coils_per_phase not unique across phases: {d}")
    return int(next(iter(v)))

# ✅ 데모/테스트 출력은 import 시 실행되지 않도록 별도 함수로 분리
def _demo_lock_coils_per_phase():
    # (병렬 경로 2개/상) 이면 coils_per_phase = 4 로 자동 고정:
    conn = WindingConnSpec(n_parallel_circuits_per_phase={"A":2,"B":2,"C":2})
    cph = lock_coils_per_phase_global(conn=conn, n_slots=24, double_layer=True)
    print("coils_per_phase (A=B=C, 2 parallel circuits):", cph)  # 4

    # (병렬 경로 1개/상) 이면 coils_per_phase = 8:
    conn = WindingConnSpec(n_parallel_circuits_per_phase={"A":1,"B":1,"C":1})
    cph = lock_coils_per_phase_global(conn=conn, n_slots=24, double_layer=True)
    print("coils_per_phase (A=B=C, 1 parallel circuit):", cph)  # 8
