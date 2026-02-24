# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 22:24:26 2026

@author: user, SANG JIN PARK
"""

# utils/femm_builder.py
import femm
import math
import os
import time
import win32com.client  # <- 여기서 정의되어야 에러가 나지 않습니다.
from core.winding_table import build_winding_table_from_row, build_winding_table_24s4p


# 슬롯 내부 ========= FEMM 세션 초기화 ===========
def femm_init():
    import femm
    # 1. 기존 연결 종료 및 새로 열기
    try:
        femm.closefemm()
    except:
        pass
    
    femm.openfemm()
    # 2. 반드시 자계(Magnetics) 문서를 먼저 생성해야 mi_ 함수들이 활성화됨
    femm.newdocument(0) 

    # 3. 키워드 인자(freq=...) 제거하고 순서대로 입력 (중요!)
    # 순서: freq, units, type, precision, depth, minangle
    import configs.config as C
    depth = getattr(C, "Stack_Rotor", 100) # config의 적층길이 반영
    femm.mi_probdef(0, "millimeters", "planar", 1e-8, depth, 30)
    print(f"[FEMM] Initialized with planar, mm, 0Hz, depth={depth}")

#  ========= 재료 정의 ===========
def define_materials():
    # 철심
    #femm.mi_addmaterial(
    #    "M19_Steel",        # .vs. NdFeB N42SH
    # ---------------------------------------------------------
    # mi_addmaterial(name, mu_x, mu_y, H_c, J, Cduct, Lam_d, 
    #                Phi_hmax, Lam_fill, LamType, Phi_hx, Phi_hy, nstrands, MagDir)
    # ---------------------------------------------------------

    # 1. 철심 (M19 Steel 예시)
    # LamType=1: 전동기용 적층강판(Lamination) 설정
    femm.mi_addmaterial("M19_Steel", 5000, 5000, 0, 0, 0, 0, 0, 0.95, 1, 0, 0)

    # 2. 구리 (Copper)
    # Cduct=58: 전도도 (MS/m)
    femm.mi_addmaterial("Copper", 1, 1, 0, 0, 58, 0, 0, 1, 0, 0, 0)

    # 3. 공기 (Air)
    femm.mi_addmaterial("Air", 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)

    # 4. 영구자석 (필요 시 추가: 예시 N42SH)
    # H_c는 보자력 [A/m]
    # femm.mi_addmaterial("N42SH", 1.05, 1.05, 900000, 0, 0.6, 0, 0, 1, 0, 0, 0)

def draw_stator_geometry():
    import configs.config as C
    
    # 주요 치수 (예시 컬럼명 - 실제 config에 맞춰 수정)
    R_out = C.OD_stator / 2
    R_in  = C.ID_stator / 2
    R_slot_bottom = R_out - 10.0 # 요크 두께를 고려한 슬롯 바닥 반경
    
    # 1. 고정자 외곽선 (Yoke outer boundary)
    femm.mi_addnode(R_out, 0)
    femm.mi_addnode(-R_out, 0)
    femm.mi_addarc(R_out, 0, -R_out, 0, 180, 1)
    femm.mi_addarc(-R_out, 0, R_out, 0, 180, 1)
    
    # 2. 고정자 내경 (Bore boundary)
    femm.mi_addnode(R_in, 0)
    femm.mi_addnode(-R_in, 0)
    femm.mi_addarc(R_in, 0, -R_in, 0, 180, 1)
    femm.mi_addarc(-R_in, 0, R_in, 0, 180, 1)

    # 3. 슬롯 영역을 닫힌 공간으로 그리는 로직 추가 필요
    # (단순 선이 아니라 영역이 분리되어야 Copper 재질 할당이 가능함)

def draw_rotor_geometry():
    import configs.config as C
    import math
    import femm

    # [1] config 치수 로드 (반경으로 변환) - 반경 계산은 config에서 직접 계산하거나, 필요한 경우 추가 인자로 받도록 수정 가능
    r_rotor_out = C.OD_rotor / 2.0  # 회전자 최외경 (자석 포함)
    r_rotor_in  = C.ID_rotor / 2.0  # 회전자 내경 (샤프트 삽입부)
    r_stator_in = C.ID_stator / 2.0 # 고정자 내경 (공극 계산용)

    # [2] 회전자 외곽선 그리기
    femm.mi_addnode(r_rotor_out, 0)
    femm.mi_addnode(-r_rotor_out, 0)
    femm.mi_addarc(r_rotor_out, 0, -r_rotor_out, 0, 180, 1)
    femm.mi_addarc(-r_rotor_out, 0, r_rotor_out, 0, 180, 1)

    # [3] 회전자 내경(샤프트 홀) 그리기
    femm.mi_addnode(r_rotor_in, 0)
    femm.mi_addnode(-r_rotor_in, 0)
    femm.mi_addarc(r_rotor_in, 0, -r_rotor_in, 0, 180, 1)
    femm.mi_addarc(-r_rotor_in, 0, r_rotor_in, 0, 180, 1)

    # [4] 재질 할당을 위한 블록 라벨 배치
    # 회전자 철심 (M19_Steel) - 외경과 내경의 중간 지점
    r_core_mid = (r_rotor_out + r_rotor_in) / 2.0
    femm.mi_addblocklabel(r_core_mid, 0)
    femm.mi_selectlabel(r_core_mid, 0)
    femm.mi_setblockprop("M19_Steel", 1, 0, "<None>", 0, 0, 0)
    femm.mi_clearselected()

    # 공극 영역 (Air) - 고정자 내경과 회전자 외경 사이
    r_gap_mid = (r_stator_in + r_rotor_out) / 2.0
    femm.mi_addblocklabel(r_gap_mid, 0)
    femm.mi_selectlabel(r_gap_mid, 0)
    femm.mi_setblockprop("Air", 1, 0, "<None>", 0, 0, 0)
    femm.mi_clearselected()

    print(f"[GEOMETRY] Rotor (OD:{C.OD_Rotor}, ID:{C.ID_Rotor}) generated.")

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

# ===================== 전체 .fem 생성 래퍼 함수 ===========================
def build_fem_from_winding(winding_table, out_fem_path, r_slot_mid):
    """
    FEMM 모델을 생성하는 메인 함수 - 타이밍 이슈 완벽 차단 버전
    - femm_init()에서 모든 초기화 작업을 수행하여 안정적인 세션 확보
    mi_ 함수 인식을 우회하여 직접 명령어를 전달하는 안정화 버전
    ActiveX를 직접 호출하여 FEMM을 제어하는 가장 안정적인 버전
    """
    try:
        # 1. FEMM 프로세스 연결
        try:
            # 이미 열려있는 FEMM이 있다면 연결, 없으면 새로 실행
            femm_app = win32com.client.Dispatch("femm.activefemm")
        except:
            os.startfile("C:\\femm42\\bin\\femm.exe") # FEMM 설치 경로 확인 필요
            time.sleep(2)
            femm_app = win32com.client.Dispatch("femm.activefemm")

        # 2. 문서 초기화 및 Magnetics 설정
        femm_app.call2femm('newdocument(0)')
        
        import configs.config as C
        depth = getattr(C, "Stack_rotor", 55.0)
        
        # 모든 명령을 문자열로 직접 전달 (mi_ 함수 에러 방지)
        femm_app.call2femm(f'mi_probdef(0, "millimeters", "planar", 1e-8, {depth}, 30)')
        
        # 3. 재료 정의
        femm_app.call2femm('mi_addmaterial("M19_Steel", 5000, 5000, 0, 0, 0, 0, 0, 0.95, 1, 0, 0)')
        femm_app.call2femm('mi_addmaterial("Copper", 1, 1, 0, 0, 58, 0, 0, 1, 0, 0, 0)')
        femm_app.call2femm('mi_addmaterial("Air", 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)')

        # 4. 회로 정의
        for ph in ["A", "B", "C"]:
            femm_app.call2femm(f'mi_addcircuit("Phase{ph}", 0, 1)')

        # 5. 형상 그리기 (기존 함수가 mi_를 사용한다면 여기서도 call2femm으로 감싸야 함)
        # 예시: 직접 그리기 로직
        r_stator_in = C.ID_stator / 2
        r_stator_out = C.OD_stator / 2
        femm_app.call2femm(f'mi_addcircle(0, 0, {r_stator_in})')
        femm_app.call2femm(f'mi_addcircle(0, 0, {r_stator_out})')

        # 6. 권선 할당
        angle_step = 360.0 / 24 
        dr = 1.5 
        for _, row in winding_table.iterrows():
            slot_idx = int(row["Slot"])
            phase = str(row["Phase"])
            pol = 1 if str(row["Polarity"]) == "+" else -1
            turns = row.get("Turns", 10)
            
            theta = math.radians((slot_idx - 1) * angle_step)
            r_curr = r_slot_mid + (dr if str(row["Layer"]).upper() == "U" else -dr)
            
            px, py = r_curr * math.cos(theta), r_curr * math.sin(theta)
            
            femm_app.call2femm(f'mi_addblocklabel({px}, {py})')
            femm_app.call2femm(f'mi_selectlabel({px}, {py})')
            femm_app.call2femm(f'mi_setblockprop("Copper", 0, 0, "Phase{phase}", 0, 0, {turns * pol})')
            femm_app.call2femm('mi_clearselected()')

        # 7. 저장
        # 역슬래시 문제를 피하기 위해 경로를 슬래시로 변환
        safe_path = out_fem_path.replace("\\", "/")
        femm_app.call2femm(f'mi_saveas("{safe_path}")')
        
        # 8. 종료
        femm_app.call2femm('quit()')

    except Exception as e:
        print(f"      [ERR] win32com logic error: {e}")
        raise e
    
def run_femm_generation(df_results, output_dir, r_slot_mid_mm=None):
    """
    df_results에서 최적 설계안을 선별하여 FEMM 파일을 자동 생성합니다.
    r_slot_mid_mm은 기본적으로 C.D_use / 2.0 을 사용합니다.
    """
    import os
    import configs.config as C
    # core.winding_table에서 수정된 함수 임포트
    from core.winding_table import build_winding_table_from_row

    # 1. 절대 경로로 지정하여 유실 방지
    base_dir = os.getcwd() 
    target_dir = os.path.join(base_dir, "results", "femm_models")
    os.makedirs(target_dir, exist_ok=True)
    
    # 2. output_dir이 None이면 target_dir을 사용
    if output_dir is None:
        output_dir = target_dir

    # 반지름 설정: C.D_use/2 사용
    r_mid = r_slot_mid_mm if r_slot_mid_mm is not None else (C.D_use / 2.0)

    # 후보 선정 (상위 12개)
    candidates = df_results.copy()

    # 2. 고효율 후보 필터링: FW_margin이 0.05 이상인 후보들만 추출 (상위 n개 대신 필터 권장)
    # 만약 데이터가 너무 많다면 상위 5개 등으로 제한
    # 'FW_margin'이 없으면 'Margin' 열을 찾아보고, 둘 다 없으면 필터링 없이 진행합니다.
    if "FW_margin" in df_results.columns:
        candidates = df_results[df_results["FW_margin"] >= 0.05].copy()
    elif "V_margin_pct" in df_results.columns:
        candidates = df_results[df_results["V_margin_pct"] >= 0.05].copy()
    else:
        print("[WARN] 'FW_margin' or 'V_margin_pct' column not found. Using all rows.")
        candidates = df_results.copy()

    if len(candidates) > 12:
        candidates = candidates.nlargest(12, "V_margin_pct")
    
    print(f"[FEMM-GEN] Total {len(candidates)} candidates selected for modeling.")
    print(f"[FEMM-GEN] Slot Mid-Radius: {r_mid:.2f} mm (from D_use/2)")

    for i, (idx, row) in enumerate(candidates.iterrows()):
        # A. 권선표 생성 (build_winding_table_from_row 사용 시 인자 자동 매칭)
        # 이 함수 내부에서 _slot_electrical_deg 등을 사용하여 정확한 24S4P 표를 만듭니다.
        wt = build_winding_table_from_row(row)
        
        # B. 파일명 설정 (식별이 가능하도록 구성)
        awg = int(row.get("AWG", 0))
        par = int(row.get("Parallels", 1))
        turns = int(row.get("Turns_per_slot_side", 15))  # 기본값 15로 설정 (실제 컬럼명에 맞춰 수정 필요)
        file_name = f"24S4P_AWG{awg}_P{par}_N{turns}_idx{idx}.fem"
        file_path = os.path.join(target_dir, file_name)
        
        # C. 반지름(r_mid) 결정
        # 2. 반지름 설정: 입력값이 없으면 C.D_use의 절반(반지름)을 사용
        if r_slot_mid_mm is None:
            r_mid = C.D_use / 2.0
        else:
            # 설계 데이터프레임의 실제 컬럼명에 맞춰 수정 필요 (예: D_stator_inner / 2 + offset)
            r_mid = r_slot_mid_mm    # 기본값 예시
        
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
            print(f"[{i+1}/{len(candidates)}] Failed to generate model: {file_name} | Error: {e}")

    print(f"[FEMM-GEN] All tasks finished. Files saved in: {target_dir}")
    print(f"[DONE] Next Step: Perform FEMM Batch Analysis (Ld/Lq Extraction)")
    
def build_femm_for_top_designs(df, topk=1):
    if df is None or df.empty:
        return

    top = df.head(topk)

    for idx, row in top.iterrows():
        winding_df = build_winding_table_from_row(row)
        fem_name = f"design_{idx}.fem"
        build_fem_from_winding(winding_df, row, fem_name)
# ================== EOF ====================

"""
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
        femm.mi_addarc(
            (ID/2)*math.cos(theta),
            (ID/2)*math.sin(theta),
            (ID/2)*math.cos(theta+2*math.pi/N),
            (ID/2)*math.sin(theta+2*math.pi/N),
            90,
            1,
        )
"""