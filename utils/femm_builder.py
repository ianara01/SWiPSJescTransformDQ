# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 22:24:26 2026

@author: user, SANG JIN PARK
"""

# utils/femm_builder.py
import femm
import math
import os

# ========= FEMM 세션 초기화 ===========
def femm_init():
    femm.openfemm()
    femm.newdocument(0)  # 0 = magnetics
    femm.mi_probdef(
        freq=0,
        units="millimeters",
        type="planar",
        precision=1e-8,
        depth=100,
        minangle=30,
    )

#  ========= 재료 정의 ===========
def define_materials():
    # 철심
    femm.mi_addmaterial(
        "M19_Steel",
        mu_x=5000,
        mu_y=5000,
        H_c=0,
        J=0,
        Cduct=0,
        Lam_d=0.95,
        Phi_hmax=0,
        Lam_fill=0.95,
        LamType=0,
        Phi_hx=0,
        Phi_hy=0,
    )

    # 구리
    femm.mi_addmaterial(
        "Copper",
        mu_x=1,
        mu_y=1,
        H_c=0,
        J=0,
        Cduct=58,
        Lam_d=1,
    )

    # 공기
    femm.mi_addmaterial("Air", 1, 1, 0, 0, 0)

# ========== Circuit 정의 (A/B/C 상) =========
def define_circuits():
    femm.mi_addcircprop("PhaseA", 0, 1)
    femm.mi_addcircprop("PhaseB", 0, 1)
    femm.mi_addcircprop("PhaseC", 0, 1)

# ===========슬롯 중심 각 계산 =============
def slot_center_angle(slot_idx, n_slots=24):
    return 2 * math.pi * (slot_idx - 1) / n_slots

# =========== 권선 배치 함수 ==============
def assign_windings_from_table(
    winding_table,
    r_slot_mid,
    n_slots=24,
):
    """
    winding_table: DataFrame
    r_slot_mid: 슬롯 중심 반경(mm)
    """

    for _, row in winding_table.iterrows():
        slot = int(row["Slot"])
        phase = row["Phase"]
        pol = row["Polarity"]
        turns = int(row["Turns_per_slot_side"])

        circ = f"Phase{phase}"
        sign = 1 if pol == "+" else -1

        theta = slot_center_angle(slot, n_slots)
        x = r_slot_mid * math.cos(theta)
        y = r_slot_mid * math.sin(theta)

        femm.mi_addblocklabel(x, y)
        femm.mi_selectlabel(x, y)

        femm.mi_setblockprop(
            "Copper",
            automesh=1,
            meshsize=0,
            incircuit=circ,
            magdir=0,
            group=slot,
            turns=sign * turns,
        )

        femm.mi_clearselected()

# =============== 전체 .fem 생성 래퍼 함수 =============
def build_fem_from_winding(
    winding_table,
    out_fem_path,
    r_slot_mid=60.0,
):
    femm_init()
    define_materials()
    define_circuits()

    # (여기서 stator/slot 형상 draw 함수 호출 가능)
    # draw_stator_geometry(...)

    assign_windings_from_table(
        winding_table=winding_table,
        r_slot_mid=r_slot_mid,
        n_slots=24,
    )

    femm.mi_zoomnatural()
    femm.mi_saveas(out_fem_path)
    femm.closefemm()
    print(f"FEMM 파일이 생성되었습니다: {out_fem_path}")
# ================== EOF ====================