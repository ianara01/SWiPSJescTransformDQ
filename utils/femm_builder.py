# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 22:24:26 2026

@author: user, SANG JIN PARK
"""

# utils/femm_builder.py
import femm
import math
import os
from core.winding_table import build_winding_table_from_row, build_winding_table_24s4p


def draw_stator_geometry():
    import configs.config as C

    ID = C.ID_slot
    OD = C.OD_slot
    N = C.N_slots

    # 외경
    femm.mi_addcircle(0, 0, OD/2)

    # 내경
    femm.mi_addcircle(0, 0, ID/2)

    # 슬롯 구분선
    for i in range(N):
        theta = 2 * math.pi * i / N
        x = (OD/2) * math.cos(theta)
        y = (OD/2) * math.sin(theta)
        femm.mi_addnode(x, y)
        femm.mi_addnode((ID/2)*math.cos(theta), (ID/2)*math.sin(theta))
        femm.mi_addsegment(x, y, (ID/2)*math.cos(theta), (ID/2)*math.sin(theta))
        femm.mi_addarc(
            (OD/2)*math.cos(theta),
            (OD/2)*math.sin(theta),
            (ID/2)*math.cos(theta),
            (ID/2)*math.sin(theta),
            90,
            1,
        )
        femm.mi_addarc(
            (OD/2)*math.cos(theta+2*math.pi/N),
            (OD/2)*math.sin(theta+2*math.pi/N),
            (ID/2)*math.cos(theta+2*math.pi/N),
            (ID/2)*math.sin(theta+2*math.pi/N),
            90,
            1,
        )
        femm.mi_addarc(
            (OD/2)*math.cos(theta),
            (OD/2)*math.sin(theta),
            (OD/2)*math.cos(theta+2*math.pi/N),
            (OD/2)*math.sin(theta+2*math.pi/N),
            90,
            1,
        )

# 슬롯 내부 ========= FEMM 세션 초기화 ===========
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
        "M19_Steel",        # NdFeB N42SH
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
def build_fem_from_winding(winding_table, out_fem_path, r_slot_mid):
    femm.openfemm()
    femm.newdocument(0) # Magnetics
    
    # 1. 기본 회로 추가 (Phase A, B, C)
    for ph in ["A", "B", "C"]:
        femm.mi_addcircuit(f"Phase{ph}", 0, 1) # 0: Series, 1: Amps (가정)

    # 2. 슬롯별 권선 배치 (핵심 보완 부분)
    # 슬롯 당 각도는 24슬롯 기준 15도
    angle_step = 360.0 / 24 
    
    # 2층권 배치를 위한 반경 오프셋 (슬롯 형상에 따라 조절 필요)
    dr = 1.5 

    for _, row in winding_table.iterrows():
        slot_idx = row["Slot"]
        phase = row["Phase"]
        polarity = 1 if row["Polarity"] == "+" else -1
        turns = row["Turns"]
        
        # 슬롯 중앙 각도 (1번 슬롯이 0도라고 가정)
        theta_deg = (slot_idx - 1) * angle_step
        theta_rad = math.radians(theta_deg)
        
        # Layer에 따른 반경 차등 적용 (겹침 방지)
        r_current = r_slot_mid + (dr if row["Layer"] == "U" else -dr)
        
        px = r_current * math.cos(theta_rad)
        py = r_current * math.sin(theta_rad)
        
        # FEMM에 블록 라벨 추가
        femm.mi_addblocklabel(px, py)
        femm.mi_selectlabel(px, py)
        femm.mi_setblockprop("Copper", 0, 0, f"Phase{phase}", 0, 0, turns * polarity)
        femm.mi_clearselected()

    femm.mi_saveas(out_fem_path)
    print(f"FEMM 파일이 생성되었습니다: {out_fem_path}")

def run_femm_generation(df_results, output_dir, r_slot_mid_mm=None):
    """
    df_results에서 최적 설계안을 선별하여 FEMM 파일을 자동 생성합니다.
    """
    import os
    # core.winding_table에서 수정된 함수 임포트
    from core.winding_table import build_winding_table_from_row

    # 1. 저장 폴더 생성 (main.py의 out_dir 하위에 femm 폴더 생성)
    femm_dir = os.path.join(output_dir, "femm_models")
    os.makedirs(femm_dir, exist_ok=True)
    
    # 2. 필터링: FW_margin이 0.05 이상인 후보들만 추출 (상위 n개 대신 필터 권장)
    # 만약 데이터가 너무 많다면 상위 5개 등으로 제한
    candidates = df_results[df_results["FW_margin"] >= 0.05].copy()
    if len(candidates) > 10:
        candidates = candidates.nlargest(10, "FW_margin")
    
    print(f"[FEMM-GEN] Total {len(candidates)} candidates selected for modeling.")

    for i, (idx, row) in enumerate(candidates.iterrows()):
        # A. 권선표 생성 (build_winding_table_from_row 사용 시 인자 자동 매칭)
        # 이 함수 내부에서 _slot_electrical_deg 등을 사용하여 정확한 24S4P 표를 만듭니다.
        wt = build_winding_table_from_row(row)
        
        # B. 파일명 설정 (식별이 가능하도록 구성)
        awg = int(row.get("AWG", 0))
        par = int(row.get("Parallels", 1))
        turns = int(row.get("Turns_per_slot_side", 10))
        file_name = f"24S4P_AWG{awg}_P{par}_N{turns}_idx{idx}.fem"
        file_path = os.path.join(femm_dir, file_name)
        
        # C. 반지름(r_mid) 결정
        # r_slot_mid_mm 인자가 있으면 그것을 쓰고, 없으면 row에서 계산
        if r_slot_mid_mm is not None:
            r_mid = r_slot_mid_mm
        else:
            # 설계 데이터프레임의 실제 컬럼명에 맞춰 수정 필요 (예: D_stator_inner / 2 + offset)
            r_mid = 60.0 # 기본값 예시
        
        # D. 실제 FEMM 파일 생성 실행
        try:
            # build_fem_from_winding은 femm_builder.py 내부에 정의된 함수
            build_fem_from_winding(
                winding_table=wt, 
                out_fem_path=file_path, 
                r_slot_mid=r_mid
            )
            print(f"[{i+1}/{len(candidates)}] Success: {file_name}")
        except Exception as e:
            print(f"[{i+1}/{len(candidates)}] Failed: {file_name} | Error: {e}")

    print(f"[FEMM-GEN] All tasks finished. Files saved in: {femm_dir}")
    
def build_femm_for_top_designs(df, topk=1):
    if df is None or df.empty:
        return

    top = df.head(topk)

    for idx, row in top.iterrows():
        winding_df = build_winding_table_from_row(row)
        fem_name = f"design_{idx}.fem"
        build_fem_from_winding(winding_df, row, fem_name)
# ================== EOF ====================