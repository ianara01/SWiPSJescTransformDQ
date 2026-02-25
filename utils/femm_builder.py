# -*- coding: utf-8 -*-
import os
import math
import time
import femm
import win32com.client
import configs.config as C
from core.winding_table import build_winding_table_from_row


def draw_stator_geometry(femm_app):
    # FEMM 실행 및 새 문서 생성
    femm.openfemm()
    femm.newdocument(0)  # 0 = Planar

    """지름 기반 좌표 정밀도를 높여 Arc와 외경을 완벽히 구현합니다."""
    # 1. 치수 로드 (지름 -> 반지름)
    cx, cy = 0.0, 0.0
    r_so = C.OD_stator / 2.0    # 고정자 외경
    r_si = C.ID_stator / 2.0    # 고정자 내경 (77.0 / 2 = 38.5)
    r_sb = C.OD_slot / 2.0      # 슬롯 바닥 (122.5 / 2 = 61.25)
    w_s  = getattr(C, "width_slot", 6.0)
    n    = C.N_slots

    def call(cmd):
        femm_app.call2femm(cmd)

    # [해결 1] 고정자 외경 원 구현 (반드시 루프 밖에서 실행)
    #call(f'mi_addcircle(0, 0, {r_so})')
    # [수정 권고안] 4개의 노드와 90도 호(Arc)를 이용한 외경 구현
    # 1단계: 4개의 기준 노드 생성 (정밀도를 위해 변수화)
    nodes = [
        [cx + r_so, cy],        # 0도 (우)
        [cx, cy + r_so],        # 90도 (상)
        [cx - r_so, cy],        # 180도 (좌)
        [cx, cy - r_so]         # 270도 (하)
    ]

    for p in nodes:
        call(f'mi_addnode({p[0]}, {p[1]})')

    # 2단계: 4개의 호 연결 (반시계 방향)
    # mi_addarc(x1, y1, x2, y2, angle, maxseg)
    call(f'mi_addarc({nodes[0][0]}, {nodes[0][1]}, {nodes[1][0]}, {nodes[1][1]}, 90, 1)')
    call(f'mi_addarc({nodes[1][0]}, {nodes[1][1]}, {nodes[2][0]}, {nodes[2][1]}, 90, 1)')
    call(f'mi_addarc({nodes[2][0]}, {nodes[2][1]}, {nodes[3][0]}, {nodes[3][1]}, 90, 1)')
    call(f'mi_addarc({nodes[3][0]}, {nodes[3][1]}, {nodes[0][0]}, {nodes[0][1]}, 90, 1)')

    # 2. 슬롯 그리기 (정밀한 Arc 및 Segment 사용)
    angle_step = 2 * math.pi / n

    for i in range(n):
        theta = i * angle_step
        
        # 정밀한 각도 계산
        alpha_st = math.asin((w_s / 2.0) / r_si)
        alpha_sb = math.asin((w_s / 2.0) / r_sb)

        # 슬롯 네 모서리 좌표 (Round 처리로 정밀도 통일)
        p1 = [round(r_si * math.cos(theta - alpha_st), 8), round(r_si * math.sin(theta - alpha_st), 8)]
        p2 = [round(r_si * math.cos(theta + alpha_st), 8), round(r_si * math.sin(theta + alpha_st), 8)]
        p3 = [round(r_sb * math.cos(theta + alpha_sb), 8), round(r_sb * math.sin(theta + alpha_sb), 8)]
        p4 = [round(r_sb * math.cos(theta - alpha_sb), 8), round(r_sb * math.sin(theta - alpha_sb), 8)]

        # 노드 및 직선(Segment) 생성
        for p in [p1, p2, p3, p4]: call(f'mi_addnode({p[0]}, {p[1]})')
        call(f'mi_addsegment({p1[0]}, {p1[1]}, {p4[0]}, {p4[1]})')
        call(f'mi_addsegment({p2[0]}, {p2[1]}, {p3[0]}, {p3[1]})')
        
        # [해결 2] 슬롯 바닥 Arc (p3 -> p4)
        # 각도는 양수(+)로 입력하여 반시계 방향 보장
        arc_angle_base = math.degrees(alpha_sb * 2)
        call(f'mi_addarc({p3[0]}, {p3[1]}, {p4[0]}, {p4[1]}, {arc_angle_base}, 1)')

        # [해결 3] 이빨 끝(내경) Arc (p2 -> 다음 슬롯의 p1)
        next_theta = (i + 1) * angle_step
        p_next1 = [round(r_si * math.cos(next_theta - alpha_st), 8), round(r_si * math.sin(next_theta - alpha_st), 8)]
        
        call(f'mi_addnode({p_next1[0]}, {p_next1[1]})')
        arc_angle_tooth = math.degrees(angle_step - (alpha_st * 2))
        call(f'mi_addarc({p2[0]}, {p2[1]}, {p_next1[0]}, {p_next1[1]}, {arc_angle_tooth}, 1)')

    # [해결 4] 재질 라벨 위치 (이빨 사이 요크 영역)
    # 0도 방향은 슬롯이 있으므로, 절반 각도 지점에 배치하여 충돌 방지
    label_r = (r_so + r_sb) / 2.0       # 약 66.8mm 지점
    lx = label_r * math.cos(angle_step / 2)
    ly = label_r * math.sin(angle_step / 2)
    call(f'mi_addblocklabel({lx}, {ly})')
    call(f'mi_selectlabel({lx}, {ly})')
    call('mi_setblockprop("M19_Steel", 1, 0, "<None>", 0, 0, 0)')
    call('mi_clearselected()')
    call('mi_drawhole(0, 0, 10, 10)')

    # 화면에 꽉 차게 보기
    call('mi_zoomnatural()')
    call('mi_refreshview()')

def draw_rotor_geometry(femm_app):
    """
    회전자 내/외경을 4분할 Arc 방식으로 그려서 형상을 정밀하게 구현합니다.
    지름(Diameter) 데이터를 바탕으로 회전자 내/외경 및 재질 라벨을 생성합니다.
    """
    # 1. 치수 로드
    r_ro = C.OD_rotor / 2.0   # 회전자 외경 반지름
    r_ri = C.ID_rotor / 2.0   # 회전자 내경 반지름
    r_si = C.ID_stator / 2.0  # 고정자 내경 반지름 (공극 라벨용)
    cx, cy = 0.0, 0.0

    def call(cmd):
        femm_app.call2femm(cmd)

    # --- 2. 회전자 외경(OD) 그리기 (4분할 Arc) ---
    ro_nodes = [
        [cx + r_ro, cy], [cx, cy + r_ro], [cx - r_ro, cy], [cx, cy - r_ro]
    ]
    for p in ro_nodes:
        call(f'mi_addnode({p[0]}, {p[1]})')
    
    call(f'mi_addarc({ro_nodes[0][0]}, {ro_nodes[0][1]}, {ro_nodes[1][0]}, {ro_nodes[1][1]}, 90, 1)')
    call(f'mi_addarc({ro_nodes[1][0]}, {ro_nodes[1][1]}, {ro_nodes[2][0]}, {ro_nodes[2][1]}, 90, 1)')
    call(f'mi_addarc({ro_nodes[2][0]}, {ro_nodes[2][1]}, {ro_nodes[3][0]}, {ro_nodes[3][1]}, 90, 1)')
    call(f'mi_addarc({ro_nodes[3][0]}, {ro_nodes[3][1]}, {ro_nodes[0][0]}, {ro_nodes[0][1]}, 90, 1)')

    # --- 3. 회전자 내경(ID) 그리기 (4분할 Arc) ---
    ri_nodes = [
        [cx + r_ri, cy], [cx, cy + r_ri], [cx - r_ri, cy], [cx, cy - r_ri]
    ]
    for p in ri_nodes:
        call(f'mi_addnode({p[0]}, {p[1]})')
    
    call(f'mi_addarc({ri_nodes[0][0]}, {ri_nodes[0][1]}, {ri_nodes[1][0]}, {ri_nodes[1][1]}, 90, 1)')
    call(f'mi_addarc({ri_nodes[1][0]}, {ri_nodes[1][1]}, {ri_nodes[2][0]}, {ri_nodes[2][1]}, 90, 1)')
    call(f'mi_addarc({ri_nodes[2][0]}, {ri_nodes[2][1]}, {ri_nodes[3][0]}, {ri_nodes[3][1]}, 90, 1)')
    call(f'mi_addarc({ri_nodes[3][0]}, {ri_nodes[3][1]}, {ri_nodes[0][0]}, {ri_nodes[0][1]}, 90, 1)')

    # --- 4. 재질 라벨 배치 ---
    # (1) 회전자 철심 라벨 (내경과 외경 사이)
    r_rot_iron = (r_ro + r_ri) / 2.0
    call(f'mi_addblocklabel({r_rot_iron}, 0)')
    call(f'mi_selectlabel({r_rot_iron}, 0)')
    call('mi_setblockprop("M19_Steel", 1, 0, "<None>", 0, 0, 0)')
    call('mi_clearselected()')

    # (2) 공극(Air) 라벨 (고정자 내경과 회전자 외경 사이)
    r_gap = (r_si + r_ro) / 2.0
    call(f'mi_addblocklabel({r_gap}, 0)')
    call(f'mi_selectlabel({r_gap}, 0)')
    call('mi_setblockprop("Air", 1, 0, "<None>", 0, 0, 0)')
    call('mi_clearselected()')
    call('mi_drawhole(0, 0, 10, 10)')

    # 화면에 꽉 차게 보기
    call('mi_zoomnatural()')
    call('mi_refreshview()')

def build_fem_from_winding(winding_table, out_fem_path, r_slot_mid):
    """형상 함수들을 통합 호출하여 완전한 모델을 생성합니다."""
    try:
        # FEMM 연결 로직
        try:
            femm_app = win32com.client.Dispatch("femm.activefemm")
        except:
            os.startfile("C:\\femm42\\bin\\femm.exe")
            time.sleep(3)
            femm_app = win32com.client.Dispatch("femm.activefemm")

        femm_app.call2femm('newdocument(0)')
        depth = getattr(C, "Stack_rotor", 55.0)
        femm_app.call2femm(f'mi_probdef(0, "millimeters", "planar", 1e-8, {depth}, 30)')
        
        # 재료 정의 (필수)
        femm_app.call2femm('mi_addmaterial("M19_Steel", 5000, 5000, 0, 0, 0, 0, 0, 0.95, 1, 0, 0)')
        femm_app.call2femm('mi_addmaterial("Copper", 1, 1, 0, 0, 58, 0, 0, 1, 0, 0, 0)')
        femm_app.call2femm('mi_addmaterial("Air", 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)')
        for ph in ["A", "B", "C"]:
            femm_app.call2femm(f'mi_addcircuit("Phase{ph}", 0, 1)')

        # [수정 핵심] 형상 그리기 호출 (이 순서가 중요합니다)
        draw_stator_geometry(femm_app)
        draw_rotor_geometry(femm_app)

        # 권선 배치
        angle_step_deg = 360.0 / C.N_slots 
        dr = 2.0 # 층 간격
        for _, row in winding_table.iterrows():
            slot_idx = int(row["Slot"])
            phase = str(row["Phase"])
            pol = 1 if str(row["Polarity"]) == "+" else -1
            turns = row.get("Turns_per_slot_side", 10)
            
            theta_rad = math.radians((slot_idx - 1) * angle_step_deg)
            r_curr = r_slot_mid + (dr if str(row.get("Layer", "U")) == "U" else -dr)
            px, py = r_curr * math.cos(theta_rad), r_curr * math.sin(theta_rad)
            
            femm_app.call2femm(f'mi_addblocklabel({px}, {py})')
            femm_app.call2femm(f'mi_selectlabel({px}, {py})')
            femm_app.call2femm(f'mi_setblockprop("Copper", 1, 0, f"Phase{phase}", 0, 0, {turns * pol})')
            femm_app.call2femm('mi_clearselected()')

        safe_path = os.path.abspath(out_fem_path).replace("\\", "/")
        femm_app.call2femm(f'mi_saveas("{safe_path}")')

    except Exception as e:
        print(f"      [ERR] build_fem error: {e}")

def run_femm_generation(df_results, output_dir, r_slot_mid_mm=None):
    """결과 데이터프레임에서 후보를 뽑아 FEMM 배치를 실행합니다."""
    target_dir = os.path.join(os.getcwd(), "results", "femm_models")
    os.makedirs(target_dir, exist_ok=True)
    
    r_mid = r_slot_mid_mm if r_slot_mid_mm is not None else (C.D_use / 2.0)

    # 상위 후보 선정 (V_margin_pct 기준)
    candidates = df_results.copy()
    if "V_margin_pct" in candidates.columns:
        candidates = candidates.nlargest(12, "V_margin_pct")
    
    print(f"[FEMM-GEN] Total {len(candidates)} candidates selected.")

    for i, (idx, row) in enumerate(candidates.iterrows()):
        wt = build_winding_table_from_row(row)
        
        awg = int(row.get("AWG", 0))
        par = int(row.get("Parallels", 1))
        file_name = f"24S4P_AWG{awg:02d}_P{par}_idx{idx}.fem"
        file_path = os.path.join(target_dir, file_name)
        
        try:
            build_fem_from_winding(wt, file_path, r_mid)
            print(f"[{i+1}/{len(candidates)}] Success: {file_name}")
        except Exception as e:
            print(f"[{i+1}/{len(candidates)}] Failed: {file_name} | {e}")

    print(f"[FEMM-GEN] All tasks finished. Saved in: {target_dir}")
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