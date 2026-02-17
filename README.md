# SWiPSJescTransformDQ
AWG type and length decision for ESC operation

Same Project but different Search Strategy for passes using combos

Choose No. 1 and input full for run_mode_full

Choose No. 2 and input adaptive for run_mode_adaptive

Choose No. 3 and input bflow for run_mode_bflow

Choose No. 4 and input femm_gen for FEMM generation of FW-safe winding

Choose No. 5 and input femm_extract for LD/LQ Extraction using FEMM

Choose No. 6 and input feedback for Application of LD/LQ Feedback

                    ┌──────────────┐
                    │   GUI UI     │
                    └──────┬───────┘
                           ↓
                ┌────────────────────┐
                │ Search Controller  │
                └──────┬─────────────┘
                       ↓
     ┌───────────────────────────────────┐
     │  Self-Learning BFlow Engine       │
     │  + Bayesian Narrowing             │
     │  + PASS Clustering                │
     │  + RL Exploration                 │
     │  + MCMC Fine Tuning               │
     └───────────────────────────────────┘
                       ↓
              GPU Vectorized Physics
                       ↓
               Result Memory DB


SWiPSJescTransformDQ/
├─ pyproject.toml                
       # (권장) 패키징/의존성 관리 (poetry/uv/pip)
├─ requirements.txt              
       # (대안) pip requirements
├─ README.md
├─ .env.example                  
       # 환경변수 예시 (DB 경로, GPU, 로그레벨 등)
├─ configs/
│  ├─ config.py                  
       # SSOT: 전역 기본값(기존 유지)
│  ├─ presets/                   
       # 실행 프리셋(현업에서 매우 유용)
│  │  ├─ full_fast.yaml
│  │  ├─ bflow_safe.yaml
│  │  ├─ adaptive_prod.yaml
│  └─ schema.yaml                
       # (선택) hp/case 검증 스키마
├─ core/
│  ├─ engine.py                  
       # run_sweep + 모드 엔진 핵심 (기존)
│  ├─ physics.py                 
       # 물리 계산(기존)
│  ├─ progress.py                
       # 진행률/프로파일 (기존)
│  ├─ femm/                       
       # FEMM 파이프라인을 core 하위로 분리(권장)
│  │  ├─ builder.py              
       # femm_builder.py 이동
│  │  ├─ ldlq.py                 
       # femm_ldlq.py 이동
│  │  ├─ pipeline.py             
       # femm_pipeline.py 이동
│  │  └─ __init__.py
│  ├─ search/
│  │  ├─ bflow.py                
       # run_bflow_full_two_pass/bflow_sweep_once_with_hp 이동
│  │  ├─ adaptive.py             
       # setup_rpm_adaptive_envelope_and_run 이동
│  │  ├─ full.py                 
       # run_full_pipeline 래핑/정리
│  │  ├─ narrowing.py            
       # (추가) Bayesian/MCMC/클러스터링 narrowing 로직
│  │  └─ __init__.py
│  ├─ io/
│  │  ├─ outputs.py              
       # build_output_paths, save_bundle, report 통합
│  │  ├─ artifacts.py            
       # excel/parquet/csvgz + heatmap/fixes 등
│  │  └─ __init__.py
│  ├─ db/
│  │  ├─ repo.py                 
       # DB 저장/조회(학습 메모리, 런 기록)
│  │  ├─ schema.py               
       # ORM/DDL 정의
│  │  └─ __init__.py
│  └─ __init__.py
├─ utils/
│  ├─ utils.py                   
       # T(), _row_get 등 (기존)
│  ├─ logging.py                 
       # 로거 설정(추가)
│  ├─ validate.py                
       # hp/case 검증, 방탄 로직(추가)
│  └─ __init__.py
├─ cli/
│  ├─ main.py                    
       # 엔트리포인트(현재 main.py를 이쪽으로 이동 권장)
│  ├─ args.py                    
       # argparse 정의/interactive fallback
│  └─ __init__.py
├─ ui/                           
       # (선택) GUI/대시보드
│  ├─ streamlit_app.py
│  └─ components/
├─ results/                      
       # 기본 출력 폴더(실행 산출물)
├─ data/
│  ├─ memory.db                  
       # SQLite(기본) / 로컬 개발용
│  └─ cache/                     
       # 캐시(예: ldlq json, tensor cache)
└─ tests/
   ├─ test_engine_smoke.py
   ├─ test_bflow_guards.py
   └─ test_db_repo.py
