# KAMP injection molding: CN7 and RG3

KAMP 사출성형 데이터의 CN7 품질 위험 예측과 RG3 불량 위험 순위화를 위한 프로젝트입니다. `cn7/`과 `rg3/`은 각각 독립적인 실행 파이프라인이며, 두 제품에 공통인 원본 CSV만 저장소 루트 `data/raw/`에서 공유합니다.

> 이 모델은 자동 합격·불합격 판정용이 아닙니다. 결과는 검사 우선순위와 모델 신뢰도 판단을 보조합니다. README에 적힌 성능은 개발 데이터 검증 결과이며 미래 생산 성능을 보장하지 않습니다.

## 구조

```text
.
├── data/raw/                 # 공통 원본 CSV, Git 제외
├── cn7/
│   ├── data/raw/             # CN7 전용 원본 CSV, Git 제외
│   ├── data/processed/       # 전처리 결과, Git 제외
│   ├── notebooks/            # 최종 분석 및 research 이력
│   ├── src/                  # 전처리, 학습, 검증, 추론
│   ├── models/               # 소형 최종 모델 아티팩트
│   ├── outputs/              # 실행 결과; 소형 요약만 Git 포함
│   ├── run_cn7.py
│   ├── run_cn7_test.py
│   └── run_cn7_temporal_ood.py
├── rg3/                      # 같은 data/notebooks/src/models/outputs 구분
│   └── run_rg3.py
├── notebooks/exploration/    # 초기 공통 EDA·전처리 연습
├── docs/reference/           # 대회 자료
├── docs/research/            # 분석 로그와 기획
├── docs/reports/             # 결과 보고서
├── requirements.txt
└── .gitignore
```

## 데이터 준비

대용량 데이터와 재생성 가능한 CSV는 Git에 올리지 않습니다. 기존 바탕화면 `kamp` 또는 `final` 폴더에서 아래 파일을 **복사**하여 배치합니다. 두 원본 폴더는 이 저장소 정리 과정에서 수정하지 않았습니다.

| 경로 | 파일 |
|---|---|
| `data/raw/` | `labeled_data.csv`, `unlabeled_data.csv` |
| `cn7/data/raw/` | `moldset_labeled_cn7.csv`, `moldset_unlabeled_cn7.csv` |
| `rg3/data/raw/` | `moldset_labeled_rg3.csv`, `moldset_unlabeled_rg3.csv` |

원본 파일은 `final/CN7/data/raw/`, `final/RG3/data/raw/`에 있습니다. `labeled_data.csv`와 `unlabeled_data.csv`는 두 폴더에 중복되어 있으므로 각각 한 번만 복사합니다. 데이터 사용 및 공개 범위는 KAMP 제공 조건을 확인하세요.

과거 분석 노트북은 별도의 복원 CSV를 사용합니다. 필요한 경우 `kamp/1. 사출성형기 AI 데이터셋/1. 사출성형기 AI 데이터셋/`의 `moldset_labeled_cn7_되돌림.csv`, `moldset_labeled_rg3_다시되돌림.csv`, `supervised_label_cn7.csv`를 각각 `cn7/data/raw/moldset_labeled_cn7_restored.csv`, `rg3/data/raw/moldset_labeled_rg3_restored.csv`, `cn7/data/raw/supervised_label_cn7.csv`로 복사하세요. 이 노트북들은 연구 기록이며 일부 중간 출력에 의존합니다.

## 실행

Python 3.12 기준입니다. 저장소 루트에서 의존성을 설치한 뒤 제품 디렉터리에서 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd cn7
python run_cn7.py
python run_cn7_temporal_ood.py
python run_cn7_test.py
cd ..\rg3
python run_rg3.py
```

CN7은 LightGBM 앙상블과 PCA 기반 OOD 안전 계층을 사용합니다. RG3은 RH 데이터에서 XGBoost와 시간 흐름 특징을 사용합니다. 자세한 전처리 가정, 검증 방법, 결과는 [CN7 설명](cn7/README.md)과 [RG3 설명](rg3/README.md)을 참조하세요.

`notebooks/*/research/`와 `notebooks/exploration/`은 연구 이력입니다. 일부 노트북은 당시의 작업 경로와 중간 산출물을 참조하므로, 재현 실행은 위 파이프라인을 기준으로 합니다.
노트북의 실행 결과 셀은 저장소 용량을 줄이기 위해 비웠습니다. 연구 노트북은 해당 노트북 디렉터리를 작업 디렉터리로 열어야 상대경로가 맞으며, 별도의 중간 CSV가 필요한 단계도 있습니다.

## 정리 기준

- `final`의 CN7/RG3 구현을 최종 코드로 채택했습니다.
- `kamp` 내부의 `final` 복사본, 중복 CSV, 이전 실험 출력, 가상환경, CatBoost 캐시, 압축 데이터셋은 가져오지 않았습니다.
- 중요한 연구 단계 노트북과 보고서는 영문 경로로 보존했습니다. 문서와 노트북 내부의 한국어 설명은 원문을 유지했습니다.
- `models/`에는 재현과 추론에 필요한 소형 모델을 포함했습니다. 데이터, 큰 예측표, 임시 모델은 `.gitignore`로 제외합니다.
