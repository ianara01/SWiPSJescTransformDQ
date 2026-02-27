# -*- coding: utf-8 -*-
import os
import math
import time
import femm
import numpy as np
import pandas as pd
import win32com.client
import configs.config as cfg

from core.winding_table import build_winding_table_from_row


def draw_stator_geometry(femm_app):
    # FEMM 실행 및 새 문서 생성
    femm.openfemm()
    femm.newdocument(0)  # 0 = Planar

    """지름 기반 좌표 정밀도를 높여 Arc와 외경을 완벽히 구현합니다."""
    # 1. 치수 로드 (지름 -> 반지름)
    cx, cy = 0.0, 0.0
    r_so = cfg.OD_stator / 2.0    # 고정자 외경
    r_si = cfg.ID_stator / 2.0    # 고정자 내경 (77.0 / 2 = 38.5)
    r_sb = cfg.OD_slot / 2.0      # 슬롯 바닥 (122.5 / 2 = 61.25)
    w_s  = getattr(cfg, "width_slot", 6.0)
    n    = cfg.N_slots

    def call(cmd):
        femm_app.call2femm(cmd)

    # [수정] 모든 그리기 시작 전 선택 해제 및 초기화
    call('mi_clearselected()')

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

    # 화면에 꽉 차게 보기
    call('mi_zoomnatural()')
    call('mi_refreshview()')

def draw_rotor_geometry(femm_app):
    # FEMM 실행 및 새 문서 생성
    femm.openfemm()
    femm.newdocument(0)  # 0 = Planar
    """
    회전자 내/외경을 4분할 Arc 방식으로 그려서 형상을 정밀하게 구현합니다.
    지름(Diameter) 데이터를 바탕으로 회전자 내/외경 및 재질 라벨을 생성합니다.
    """

    # 1. 치수 로드
    r_ro = cfg.OD_rotor / 2.0   # 회전자 외경 반지름
    r_ri = cfg.ID_rotor / 2.0   # 회전자 내경 반지름
    r_si = cfg.ID_stator / 2.0  # 고정자 내경 반지름 (공극 라벨용)
    cx, cy = 0.0, 0.0

    def call(cmd):
        femm_app.call2femm(cmd)

    # [수정] 모든 그리기 시작 전 선택 해제 및 초기화
    call('mi_clearselected()')

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

    # 화면에 꽉 차게 보기
    call('mi_zoomnatural()')
    call('mi_refreshview()')

# Step 1. IPM Rotor 포함 자동 생성 (build_fem_from_winding()에서 호출) - (기존 함수): 형상 설계 및 재료 할당
def build_fem_from_winding(winding_table, file_path, r_slot_mid):
    import configs.config as cfg
    import os, time, math
    import shutil
    import tempfile
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    result_data = None
    
    # --- [경로 최적화 로직] ---
    abs_file_path = os.path.abspath(file_path).replace("\\", "/")
    
    # 'results' 기준 경로 파싱
    if "results" in abs_file_path:
        base_results_path = abs_file_path.split("/results")[0] + "/results"
    else:
        base_results_path = os.path.join(os.getcwd(), "results").replace("\\", "/")

    # 폴더 경로 확정
    ans_dir = os.path.join(base_results_path, "ans").replace("\\", "/")
    models_dir = os.path.join(base_results_path, "femm_models").replace("\\", "/")
    
    os.makedirs(ans_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 파일명 정의
    base_name = os.path.basename(file_path) # xxx.fem
    ans_filename = base_name.replace(".fem", ".ans")
    
    # 최종 저장 경로
    final_fem_path = os.path.join(models_dir, base_name).replace("\\", "/")
    final_ans_path = os.path.join(ans_dir, ans_filename).replace("\\", "/")

    print(f"[PATH CHECK] Saving FEM to: {final_fem_path}")
    print(f"[PATH CHECK] Saving ANS to: {final_ans_path}")

    original_cwd = os.getcwd() # 원래 실행 경로 저장

    try:
        # FEMM 연결 로직
        try:
            femm_app = win32com.client.Dispatch("femm.activefemm")
        except:
            os.startfile("C:\\femm42\\bin\\femm.exe")
            time.sleep(3)
            femm_app = win32com.client.Dispatch("femm.activefemm")

        femm_app.call2femm('newdocument(0)')

        # 1. 해석 설정 최적화 (해석 정밀도 조정)
        depth = getattr(cfg, "Stack_rotor", 55.0)
        # 1e-8 정밀도가 너무 높으면 시간이 오래 걸릴 수 있으므로 1e-5 정도로 조정 권장
        femm_app.call2femm(f'mi_probdef(0, "millimeters", "planar", 1e-6, {depth}, 30)')
        
        # 재료 정의 (필수)
        femm_app.call2femm('mi_addmaterial("M19_Steel", 5000, 5000, 0, 0, 0, 0, 0, 0.95, 1, 0, 0)')
        femm_app.call2femm('mi_addmaterial("Copper", 1, 1, 0, 0, 58, 0, 0, 1, 0, 0, 0)')
        femm_app.call2femm('mi_addmaterial("Air", 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)')
        for phase_char in ["A", "B", "C"]:
            femm_app.call2femm(f'mi_addcircuit("Phase{phase_char}", 0, 1)')

        # [수정 핵심] 형상 그리기 호출 (이 순서가 중요합니다)
        draw_stator_geometry(femm_app)
        draw_rotor_geometry(femm_app)

        # 권선 배치
        angle_step_deg = 360.0 / cfg.N_slots 
        dr = 2.0 # 층 간격
        for _, row in winding_table.iterrows():
            slot_idx = int(row["Slot"])
            phase = str(row["Phase"])
            pol = 1 if str(row["Polarity"]) == "+" else -1
            turns = row.get("Turns_per_slot_side", 10)
            
            theta_rad = math.radians((slot_idx - 1) * angle_step_deg)
            r_curr = r_slot_mid + (dr if str(row.get("Layer", "U")) == "U" else -dr)
            px, py = r_curr * math.cos(theta_rad), r_curr * math.sin(theta_rad)
            
            # 1. 기존 라벨이 있을지 모르니 해당 위치 근처를 한 번 비웁니다.
            femm_app.call2femm(f'mi_clearselected()')
            femm_app.call2femm(f'mi_addblocklabel({px}, {py})')
            femm_app.call2femm(f'mi_selectlabel({px}, {py})')
            
            # 2. 재료 할당 (Copper) - 대소문자 주의 ("Copper")
            # 'Copper' 재료가 mi_addmaterial로 먼저 등록되어 있어야 합니다.
            femm_app.call2femm(f'mi_setblockprop("Copper", 1, 0, "Phase{phase}", 0, 0, {turns * pol})')
            femm_app.call2femm('mi_clearselected()')

        # [중요 추가] 스테이터 코어와 로터 코어에도 재료 할당 확인
        # 아래 좌표(예: r_stator_yoke 근처: OD_stator/2)는 사용자님의 모델 치수에 맞춰 조정이 필요합니다.
        # 스테이터 요크 부분 (예시 좌표)
        femm_app.call2femm(f'mi_addblocklabel(0, {cfg.OD_stator /  - 2})') 
        femm_app.call2femm(f'mi_selectlabel(0, {cfg.OD_stator /2 - 2})')
        femm_app.call2femm('mi_setblockprop("M19_Steel", 1, 0, "<None>", 0, 0, 0)')
        femm_app.call2femm('mi_clearselected()')

        # 에어갭/공기 부분 (예시 좌표)
        femm_app.call2femm('mi_addblocklabel(0, 0)') 
        femm_app.call2femm('mi_selectlabel(0, 0)')
        femm_app.call2femm('mi_setblockprop("Air", 1, 0, "<None>", 0, 0, 0)')
        femm_app.call2femm('mi_clearselected()')

        # [Step 1] 원본 모델을 femm_models 폴더에 확정 저장 - .fem 파일 저장 (명시적 절대 경로 사용)
        final_fem_path = os.path.abspath(final_fem_path).replace("\\", "/")
        femm_app.call2femm(f'mi_saveas("{final_fem_path}")') 
        print(f"[DEBUG] Permanent FEM saved at: {final_fem_path}")

        # [Step 2] [강제 해결책] 시스템 임시 디렉토리에 해석 공간 마련
        # 윈도우 보안의 영향을 받지 않는 곳에서 해석을 돌립니다.
        with tempfile.TemporaryDirectory() as tmpdirname:
            # 원본 .fem 파일을 임시 폴더로 복사 (이름 유지)
            temp_fem = os.path.join(tmpdirname, base_name).replace("\\", "/")
            import shutil
            shutil.copy2(final_fem_path, temp_fem) # 원본은 그대로 두고 복사본 생성

            temp_ans = temp_fem.replace(".fem", ".ans")

            # 임시 폴더에 있는 복사본을 엽니다.
            #femm_app.call2femm(f'open("{temp_fem}")')

            # (1) 해석 실행 전 임시 폴더에 모델 저장
            femm_app.call2femm(f'mi_saveas("{temp_fem}")')
            
            # (2) 메쉬 및 해석 실행 (0 모드 유지)
            print(f"[FEMM] Analyzing in safe temp zone: {tmpdirname}")
            # FEMM 창을 강제로 보여줍니다.
            femm_app.call2femm('showwindow()') 
            
            # 메쉬 생성 시도
            print(f"[FEMM] Unit Step 1-1: Creating Mesh... in directory: {tmpdirname}")
            res_mesh = femm_app.call2femm('mi_createmesh()')
            print(f"[DEBUG] Mesh Result: {res_mesh}")
            
            # 해석 실행 (창이 뜰 것입니다. 에러 메시지가 나오는지 보세요)
            femm_app.call2femm('mi_analyze(0)')
            time.sleep(2.0) # 충분한 해석 시간 확보

            # (3) 해석 결과가 나왔는지 확인
            # mi_analyze가 성공하면 temp_ans 파일이 물리적으로 생겨야 합니다.
            if not os.path.exists(temp_ans):
                # 만약 안 생겼다면 FEMM에게 명시적으로 로드를 시킵니다.
                femm_app.call2femm('mi_loadsolution()')
                time.sleep(1.0)
                femm_app.call2femm(f'mo_saveanalysis("{temp_ans}")')

            # (4) 성공적으로 생성된 .ans 파일을 목적지로 복사
            if os.path.exists(temp_ans):
                shutil.copy2(temp_ans, final_ans_path)
                print(f"[SUCCESS] Finally saved ANS results to: {final_ans_path}")
            else:
                print(f"[CRITICAL] FEMM failed to produce .ans even in temp zone.")

        # 3. 데이터 추출 (추출은 final_ans_path에서 수행)
        femm_app.call2femm(f'mo_open("{final_ans_path}")')

        # [Step 6] 데이터 추출 루프 (파일이 하드디스크에 기록된 후 추출)
        results = {}
        for p in ["A", "B", "C"]:
            p_name = f"Phase{p}"
            res = femm_app.call2femm(f'mo_getcircuitproperties("{p_name}")')
            if isinstance(res, (list, tuple)) and len(res) >= 3:
                results[p] = [float(res[0]), float(res[1]), float(res[2])]

        if "A" in results:
            flux, curr, volt = results["A"]
            L = flux / curr if curr != 0 else 0
            result_data = {
                "flux": flux, "current": curr, "voltage": volt, 
                "inductance": L, "all_phases": results,
                "ans_file": final_ans_path # 저장된 경로 전달
            }
            print(f"--- [SUCCESS] Analysis Completed & Saved at {ans_dir} ---")

        # [Step 7] 리소스 정리
        femm_app.call2femm('mo_close()')
        femm_app.call2femm('mi_close()')

    finally:
            # 에러가 나더라도 원래 작업 디렉토리로 복구 (매우 중요)
            os.chdir(original_cwd)
            print(f"[SYSTEM] Restored working directory to: {os.getcwd()}")
    
    return result_data


        # [Step 4] 해석이 끝난 후, 생성된 .ans 파일을 우리가 원하는 results/ans 폴더로 강제 이동/복사합니다.
        #import shutil
        #try:
            # 해석 직후 생성된 파일을 전용 폴더(ans_dir)로 옮깁니다.
        #    if os.path.exists(default_ans_path):
        #        shutil.move(default_ans_path, final_ans_path)
        #        print(f"[MOVE] Result moved to: {final_ans_path}")
        #except Exception as move_err:
        #    print(f"[ERR] File move failed: {move_err}")

        # [Step 5] 중요: mi_loadsolution()을 사용하여 ans 폴더 보고, 이동시킨 경로를 mo_open으로 직접 엽니다!
        # 이렇게 하면 "데이터를 찾을 수 없습니다"라는 에러창 자체가 뜨지 않습니다.

# Step 2. FEMM 자동 Solve & Flux 계산 - 물리적 해석 실행 및 기초 데이터(Flux) 수집
def extract_results_batch():
    """
    results/ans 폴더 내의 모든 .ans 파일을 읽어 
    Phase A, B, C의 전기적 특성을 추출하고 엑셀 리포트를 생성합니다.
    """
    # 1. 경로 설정
    base_path = os.path.join(os.getcwd(), "results")
    ans_dir = os.path.join(base_path, "ans")
    report_path = os.path.join(base_path, "Inductance_Report.xlsx")
    
    if not os.path.exists(ans_dir):
        print(f"[ERR] 폴더를 찾을 수 없습니다: {ans_dir}")
        return

    # .ans 파일 목록 가져오기
    ans_files = [f for f in os.listdir(ans_dir) if f.endswith(".ans")]
    if not ans_files:
        print("[WARN] 분석할 .ans 파일이 없습니다.")
        return

    print(f"[BATCH] 총 {len(ans_files)}개의 결과 파일을 분석합니다...")

    # 2. FEMM 연결
    try:
        femm_app = win32com.client.Dispatch("femm.activefemm")
    except:
        print("[ERR] FEMM이 실행 중이지 않습니다.")
        return

    all_results = []

    # 3. 파일별 데이터 추출 루프
    for idx, filename in enumerate(ans_files):
        full_path = os.path.join(ans_dir, filename).replace("\\", "/")
        print(f"[{idx+1}/{len(ans_files)}] 분석 중: {filename}")
        
        try:
            # 파일 열기
            femm_app.call2femm(f'mo_open("{full_path}")')
            
            file_data = {"FileName": filename}
            
            # 각 Phase(A, B, C) 데이터 추출
            for p in ["A", "B", "C"]:
                p_name = f"Phase{p}"
                # mo_getcircuitproperties는 [Flux, Current, Voltage] 반환
                res = femm_app.call2femm(f'mo_getcircuitproperties("{p_name}")')
                
                if isinstance(res, (list, tuple)) and len(res) >= 3:
                    flux, curr, volt = float(res[0]), float(res[1]), float(res[2])
                    inductance = flux / curr if curr != 0 else 0
                    
                    # 데이터 저장
                    file_data[f"Flux_{p}"] = flux
                    file_data[f"Current_{p}"] = curr
                    file_data[f"Inductance_{p}(H)"] = inductance
                    file_data[f"Voltage_{p}"] = volt
            
            all_results.append(file_data)
            femm_app.call2femm('mo_close()') # 파일 닫기
            
        except Exception as e:
            print(f"  [ERR] {filename} 처리 중 오류: {e}")

    # 4. 데이터프레임 생성 및 엑셀 저장
    if all_results:
        df = pd.DataFrame(all_results)
        
        # 가독성을 위해 컬럼 순서 정렬 (FileName 우선)
        cols = ['FileName'] + [c for c in df.columns if c != 'FileName']
        df = df[cols]
        
        # 엑셀 파일 저장
        df.to_excel(report_path, index=False)
        print("\n" + "="*50)
        print(f"[SUCCESS] 리포트 생성 완료!")
        print(f"경로: {report_path}")
        print("="*50)
        
        return df
    else:
        print("[FAIL] 추출된 데이터가 없습니다.")
        return None

# 실행 예시
# extract_results_batch()

# Step 3. DQ 변환 및 최적 설계 후보 도출 - Park Transform(DQ 변환) 적용하여 Ld, Lq 계산 및 최적 모델 선정 (extract_results_with_dq_transform())
def extract_results_with_dq_transform():
    """
    12개 결과 분석 + Park Transform(DQ 변환) + 엑셀 저장
    """
    base_path = os.path.join(os.getcwd(), "results")
    ans_dir = os.path.join(base_path, "ans")
    report_path = os.path.join(base_path, "Ld_Lq_Analysis_Report.xlsx")
    
    ans_files = [f for f in os.listdir(ans_dir) if f.endswith(".ans")]
    if not ans_files:
        print("[ERR] 분석할 .ans 파일이 없습니다.")
        return

    try:
        femm_app = win32com.client.Dispatch("femm.activefemm")
    except:
        print("[ERR] FEMM 연결 실패")
        return

    all_results = []
    
    # Park Transform을 위한 상수 (2/3 배율)
    SQRT3 = math.sqrt(3.0)

    for filename in ans_files:
        full_path = os.path.join(ans_dir, filename).replace("\\", "/")
        try:
            femm_app.call2femm(f'mo_open("{full_path}")')
            
            data = {"FileName": filename}
            fluxes = []
            currents = []

            for p in ["A", "B", "C"]:
                res = femm_app.call2femm(f'mo_getcircuitproperties("Phase{p}")')
                f_val, i_val, _ = float(res[0]), float(res[1]), float(res[2])
                data[f"Flux_{p}"] = f_val
                data[f"Curr_{p}"] = i_val
                fluxes.append(f_val)
                currents.append(i_val)

            # --- [Park Transform 로직] ---
            # 각 상의 위상차 120도(2/3*pi) 가정
            # 여기서는 회전자 위치(theta)가 0도인 시점의 정적 해석 결과를 가정함
            lambda_a, lambda_b, lambda_c = fluxes
            ia, ib, ic = currents

            # Clark Transform (ABC -> Alpha/Beta)
            lambda_alpha = (2/3) * (lambda_a - 0.5 * lambda_b - 0.5 * lambda_c)
            lambda_beta = (2/3) * (SQRT3/2 * lambda_b - SQRT3/2 * lambda_c)
            
            i_alpha = (2/3) * (ia - 0.5 * ib - 0.5 * ic)
            i_beta = (2/3) * (SQRT3/2 * ib - SQRT3/2 * ic)

            # Park Transform (Alpha/Beta -> DQ) 
            # 해석 시 설정한 Rotor Angle(theta)이 0이라고 가정할 때:
            theta = 0 
            lambda_d = lambda_alpha * math.cos(theta) + lambda_beta * math.sin(theta)
            lambda_q = -lambda_alpha * math.sin(theta) + lambda_beta * math.cos(theta)
            
            id_val = i_alpha * math.cos(theta) + i_beta * math.sin(theta)
            iq_val = -i_alpha * math.sin(theta) + i_beta * math.cos(theta)

            # Ld, Lq 계산 (전류가 0이 아닐 때)
            data["Ld(H)"] = lambda_d / id_val if abs(id_val) > 1e-6 else 0
            data["Lq(H)"] = lambda_q / iq_val if abs(iq_val) > 1e-6 else 0
            data["Salient_Ratio(Lq/Ld)"] = data["Lq(H)"] / data["Ld(H)"] if data["Ld(H)"] != 0 else 0

            all_results.append(data)
            femm_app.call2femm('mo_close()')
            
        except Exception as e:
            print(f"[ERR] {filename} 분석 실패: {e}")

    # 엑셀 저장 및 최적 결과 도출
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_excel(report_path, index=False)
        
        # Lq/Ld 돌극비가 가장 큰 모델(최적 설계 후보) 찾기
        best_model = df.loc[df['Salient_Ratio(Lq/Ld)'].idxmax()]
        
        print("\n" + "="*60)
        print(f"▶ 전체 리포트 저장 완료: {report_path}")
        print(f"▶ 최적 설계 모델(돌극비 기준): {best_model['FileName']}")
        print(f"   - Ld: {best_model['Ld(H)']: .6f} H")
        print(f"   - Lq: {best_model['Lq(H)']: .6f} H")
        print(f"   - 돌극비(Lq/Ld): {best_model['Salient_Ratio(Lq/Ld)']: .4f}")
        print("="*60)
        
        return best_model
    return None

# 실행 예시
#best_design = extract_results_with_dq_transform()

def get_femm_results(fem_path):
    import time
    """FEMM 결과 파일에서 인덕턴스 정보를 추출합니다."""
    try:
        # 1. FEMM 인스턴스 연결
        femm_app = win32com.client.Dispatch("femm.activefemm")
        
        # 2. 파일 열기 및 솔루션 로드
        femm_app.call2femm(f'opendocument("{fem_path}")') # mi_open보다 확실함
        femm_app.call2femm('mi_loadsolution()')
        
        # [중요] 후처리 창(Post-processor) 활성화를 위한 짧은 대기
        time.sleep(0.5) 
        femm_app.call2femm('mo_showwindow()')
        
        # 3. 데이터 추출 및 타입 캐스팅
        vals = femm_app.call2femm('mo_getcircuitproperties("PhaseA")')
        
        # 리스트 형태인지, 문자열 에러가 아닌지 확인
        if isinstance(vals, (list, tuple)) and len(vals) >= 3:
            flux = float(vals[0])
            current = float(vals[1])
            voltage = float(vals[2])
            
            # 인덕턴스 계산 시 0으로 나누기 방지
            inductance = flux / current if abs(current) > 1e-6 else 0
            
            # [중요] 메모리 관리: 결과 창 닫기 (이걸 안 하면 창이 12개 뜹니다!)
            femm_app.call2femm('mo_close()')
            
            return {
                "flux": flux,
                "current": current,
                "voltage": voltage,
                "inductance": inductance
            }
        else:
            print(f" [WARN] No valid circuit data in {fem_path}")
            return None
            
    except Exception as e:
        print(f" [ERR] get_femm_results error in {fem_path}: {e}")
        return None

def run_femm_generation(df_results, target_dir, r_slot_mid_mm=None):
    """결과 데이터프레임에서 후보를 뽑아 FEMM 배치를 실행합니다."""
    target_dir = os.path.join(os.getcwd(), "results", "femm_models")
    os.makedirs(target_dir, exist_ok=True)
    
    r_mid = r_slot_mid_mm if r_slot_mid_mm is not None else (cfg.D_use / 2.0)

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


def generate_design_candidates():
    """
    설계 후보군 12개를 생성하거나, 기존에 생성된 .fem 파일 경로를 수집합니다.
    이 데이터는 main.py의 해석 루프에서 입력값으로 사용됩니다.
    """
    candidates = []
    base_results_path = os.path.join(os.getcwd(), "results")
    models_dir = os.path.join(base_results_path, "femm_models")
    ans_dir = os.path.join(base_results_path, "ans")

    # 1. 12개의 후보군 정의 (예: 권선 인덱스 229~240)
    # 실제로는 AI 엔진이나 Winding Manager에서 생성한 리스트를 가져옵니다.
    target_indices = range(229, 241) 

    for idx in target_indices:
        # 파일명 규칙 설정 (예: 24S4P_AWG00_P0_idx240.fem)
        file_name = f"{cfg.N_slots}S{cfg.N_poles}P_AWG{cfg.AWG}_{cfg.Pattern}_idx{idx}.fem"
        fem_path = os.path.join(models_dir, file_name).replace("\\", "/")
        ans_path = os.path.join(ans_dir, file_name.replace(".fem", ".ans")).replace("\\", "/")
        
        # 2. 해당 후보에 대한 권선표(Winding Table) 매칭
        # 실제 환경에서는 DB나 CSV에서 해당 idx의 권선 데이터를 로드합니다.
        # 여기서는 예시로 로직만 구성합니다.
        try:
            # winding_manager가 있다면: winding_table = wm.get_table(idx)
            # 여기선 가상의 데이터 구조를 할당
            winding_table = None # 실제 실행 시엔 각 idx에 맞는 DataFrame 전달
        except:
            winding_table = None

        # 3. 후보군 객체 생성
        design_info = {
            "index": idx,
            "name": file_name,
            "fem_path": fem_path,
            "ans_path": ans_path,
            "winding_table": winding_table,  # build_fem_from_winding의 입력값
            "status": "ready" if os.path.exists(fem_path) else "need_generation"
        }
        
        candidates.append(design_info)

    print(f"[CANDIDATE] 총 {len(candidates)}개의 설계 후보군이 준비되었습니다.")
    return candidates

def build_femm_for_top_designs(df, topk=1):
    if df is None or df.empty:
        return

    top = df.head(topk)

    for idx, row in top.iterrows():
        winding_df = build_winding_table_from_row(row)
        fem_name = f"design_{idx}.fem"
        file_path = os.path.join(os.getcwd(), "results", "femm_models", fem_name)
        build_fem_from_winding(winding_df, file_path, r_mid)
# =============================== EOF ==================================
"""
def get_femm_results(fem_path):
    FEMM 결과 파일에서 인덕턴스 정보를 안전하게 추출합니다.
    try:
        # 경로의 백슬래시 문제를 방지하기 위해 정규화
        safe_path = os.path.abspath(fem_path).replace("\\", "/")
        
        femm_app = FEMMApp() # 기존 연결된 Dispatch 객체 사용 권장
        
        # 1. 파일 열기 및 결과 로드
        femm_app.call2femm(f'open("{safe_path}")') # mi_open보다 open이 범용적임
        femm_app.call2femm('mi_loadsolution()')
        
        # [핵심] 결과 창이 완전히 뜰 때까지 잠시 대기 및 포커스 강제 이동
        import time
        time.sleep(0.2)
        femm_app.call2femm('mo_showwindow()') 
        
        # 2. 데이터 추출
        vals = femm_app.call2femm('mo_getcircuitproperties("PhaseA")')
        
        # 3. 데이터 유효성 검사 (문자열 'e', 'r' 방지)
        if isinstance(vals, (list, tuple)) and len(vals) >= 3:
            try:
                # 데이터를 실수형(float)으로 강제 변환
                flux = float(vals[0])
                curr = float(vals[1])
                volt = float(vals[2])
                
                L = flux / curr if curr != 0 else 0
                
                # [추가] 사용이 끝난 결과 창과 입력 창을 닫아 메모리 누수 방지
                femm_app.call2femm('mo_close()')
                femm_app.call2femm('mi_close()')
                
                return {
                    "flux": flux,
                    "current": curr,
                    "voltage": volt,
                    "inductance": L
                }
            except (ValueError, TypeError):
                print(f"      [ERR] Invalid data format from FEMM: {vals}")
                return None
        else:
            print(f"      [ERR] Could not get circuit properties. Raw: {vals}")
            return None
            
    except Exception as e:
        print(f"      [ERR] get_femm_results error: {e}")
        return None
"""