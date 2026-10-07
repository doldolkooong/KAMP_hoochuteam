# CN7 제조 품질위험 예측 + OOD 안전계층

초기 모델·특징·불균형 처리 실험과 재분석 과정은 [`notebooks/experiments/`](notebooks/experiments/) 및 [`notebooks/research/`](notebooks/research/)에 원본 출력과 함께 보존했습니다. 원래 파일명과 위치는 [전체 노트북 목록](../docs/notebook_inventory.md)을 참고하세요.

KAMP 사출성형 제조데이터의 **CN7 공정 품질위험을 예측하고 검사 우선순위를 지원**하기 위한 프로젝트입니다.

최종 구조는 단일 분류기만 사용하지 않습니다.

```text
기존 Failure Regime
    → LightGBM Risk Model

모델 간 판단 불일치
    → Ensemble Uncertainty

학습분포 밖의 신규 공정상태
    → PCA Reconstruction Error OOD Safety Layer

센서상 정상과 중첩되는 Hard Defect
    → 기존 품질검사 유지 / 추가 센서·비전 데이터 필요
```

> 이 모델은 자동 합격/불합격 판정기가 아닙니다. 낮은 Risk를 검사면제 근거로 사용하지 않으며, **검사 우선순위와 모델 신뢰도 판단을 지원**합니다.

---

# 1. 확정 모델 사양

| 항목 | 설정 |
|---|---|
| 학습 데이터 | 1,211 rows |
| 정상 / 불량 | 1,194 / 17 |
| Supervised Model | LightGBM |
| Validation | 4-Fold StratifiedGroupKFold |
| Group | `TimeStamp` |
| CV Seed | 42, 100, 2026, 777, 1234 |
| Feature | 23 Original + `Side_RH` + Roll5 Mean 23 |
| Feature 수 | 47 |
| Rolling Window | 직전 5 Cycle |
| Class Weight | `balanced` |
| Risk Threshold | `Risk_Probability >= 0.060` |
| OOD Model | RobustScaler + PCA Reconstruction Error |
| PCA retained variance | 95% |
| OOD Calibration | 과거 정상 Cycle의 시간순 후반 20% |
| OOD Threshold | Calibration score 99 percentile |

현재 `run_cn7.py`가 생성한 `outputs/cn7_metrics.csv`의 반복 Group-OOF 결과:

```text
PR-AUC = 0.8164 ± 0.0181
```

현재 파이프라인의 개발 OOF 정책 결과:

| 정책 | Precision | Recall | FN | 검사율 |
|---|---:|---:|---:|---:|
| Risk Only | 66.67% | 82.35% | 3 | 1.73% |
| Risk + Uncertainty | 60.00% | 88.24% | 2 | 2.06% |

**위 수치는 개발 데이터 내부 OOF 결과이며 미래 생산성능이 아닙니다.**

---

# 2. Risk Threshold 단위 주의

프로젝트에는 두 개의 Risk 표현이 있습니다.

```text
Risk_Probability : 0 ~ 1 확률
Risk_Score       : Risk_Probability × 100, 표시용 0 ~ 100 점수
```

운영 Threshold `0.060`은 **`Risk_Probability` 기준**입니다.

따라서 동일한 기준을 `Risk_Score`에 적용하면:

```text
Risk_Probability >= 0.060
≡
Risk_Score >= 6.0
```

`Risk_Score >= 0.060`으로 비교하면 threshold를 100배 낮게 적용하는 것이므로 잘못된 계산입니다.

---

# 3. 미래 시간검증 결과

`run_cn7_temporal_ood.py`는 `labeled_data.csv`에서 CN7 전체 기간을 복원한 뒤 시간순으로 검증합니다.

## E4 — 2020-10-27

Supervised model:

```text
Train : 2020-10-16 04:57:47 ~ 2020-10-20 06:20:14
Test  : 2020-10-27 00:10:40 ~ 2020-10-27 05:48:36
PR-AUC: 0.1156
```

Risk Threshold `0.060`과 Ensemble Uncertainty는 E4 불량을 탐지하지 못했습니다.

| 정책 | Precision | Recall | TP | FP | FN | 검사율 |
|---|---:|---:|---:|---:|---:|---:|
| Risk Only | 0% | 0% | 0 | 0 | 10 | 0% |
| Risk + Uncertainty | 0% | 0% | 0 | 0 | 10 | 0% |

E4는 기존 Failure Regime과 다른 강한 다변량 공정이탈을 보였기 때문에 PCA OOD Safety Layer를 추가했습니다.

### E4 PCA OOD 결과 — Cycle 기준

```text
Past normal cycles : 592
PCA fit             : 473 cycles
Calibration         : 119 cycles
PCA components      : 11
OOD threshold       : 11.299442
```

| 지표 | 결과 |
|---|---:|
| Test Cycle | 314 |
| 실제 불량 Cycle | 5 |
| OOD Flag | 4 |
| TP / FP / FN | **4 / 0 / 1** |
| Precision | **100%** |
| Recall | **80%** |
| F1 | **88.89%** |
| 검사율 | **1.27%** |

E4 불량 Cycle의 OOD Score:

```text
00:56:21 → 518.983  → OOD
00:59:00 → 157.251  → OOD
01:03:34 → 102.826  → OOD
01:04:41 →  14.225  → OOD
01:05:41 →   0.950  → 미탐
Threshold  →  11.299
```

즉 기존 LightGBM이 모두 낮은 Risk를 출력한 신규 Failure Regime 중 **5개 불량 Cycle 중 4개를 정상분포 이탈로 포착**했습니다.

## E5 — 2020-11-03

```text
PR-AUC: 0.0147
불량 Cycle: 1
```

PCA OOD는 463개 Cycle 중 6개를 Flag했지만 실제 불량 Cycle은 포함하지 못했습니다.

| 지표 | 결과 |
|---|---:|
| TP / FP / FN | 0 / 6 / 1 |
| Recall | 0% |
| OOD 검사율 | 1.30% |

E5는 현재 제공 공정센서 공간에서 정상과 상당히 중첩되는 Hard Defect로 해석합니다. 따라서 OOD를 포함해도 모든 불량을 자동 탐지할 수 있다고 주장하지 않습니다.

> **중요:** OOD 분석은 E4/E5에서 기존 모델의 실패를 확인한 뒤 수행한 사후 안전성 보완실험입니다. 독립적인 최종 성능검증을 대체하지 않습니다.

---

# 4. 최종 현장 운영 원칙

```text
1. In-Distribution
   → LightGBM Risk + Ensemble Uncertainty로 검사 우선순위 지원

2. Individual OOD
   → PCA OOD Flag를 최우선 안전경보로 사용

3. Session-wide Distribution Shift
   → 개별 제품을 전부 불량으로 판단하지 않음
   → Model_Trust_Status를 DEGRADED_BY_DRIFT로 표시
   → 기존 품질검사를 유지하고 모델 신뢰도를 하향

4. 센서상 정상과 중첩되는 Hard Defect
   → AI 단독 판정 금지
   → 기존 품질검사 및 추가 센서/비전검사 필요
```

`OOD_Flag=1`은 **불량 확정**이 아니라 **학습 정상분포와 다른 공정상태**라는 의미입니다.

---

# 5. 프로젝트 구조

`labeled_data.csv`와 `unlabeled_data.csv`는 두 제품이 공유하므로 저장소 루트 `data/raw/`에 둡니다.

```text
cn7/
│
├─ data/
│  ├─ raw/
│  │  ├─ moldset_labeled_cn7.csv
│  │  ├─ moldset_unlabeled_cn7.csv
│  └─ processed/
│
├─ src/
│  ├─ preprocess_cn7.py
│  ├─ feature_engineering_cn7.py
│  ├─ train_cn7.py
│  ├─ evaluate_cn7.py
│  ├─ inference_cn7.py
│  ├─ ood_cn7.py
│  ├─ temporal_ood_cn7.py
│  ├─ test_cn7.py
│  └─ config_cn7.py
│
├─ models/
│  ├─ cn7_lgbm_seed42.pkl
│  ├─ cn7_lgbm_seed100.pkl
│  ├─ cn7_lgbm_seed2026.pkl
│  ├─ cn7_lgbm_seed777.pkl
│  ├─ cn7_lgbm_seed1234.pkl
│  └─ cn7_pca_ood.pkl
│
├─ outputs/
│  ├─ cn7_oof_prediction.csv
│  ├─ cn7_threshold_result.csv
│  ├─ cn7_metrics.csv
│  ├─ cn7_temporal_ood_side_results.csv
│  ├─ cn7_temporal_ood_cycle_results.csv
│  └─ cn7_temporal_ood_predictions.csv
│
├─ notebooks/
│  ├─ cn7_analysis.ipynb
│  └─ cn7_ood_safety_layer.ipynb
│
├─ run_cn7.py
├─ run_cn7_test.py
├─ run_cn7_temporal_ood.py
├─ requirements.txt
└─ README.md
```

---

# 6. 파일별 역할

| 파일 | 역할 |
|---|---|
| `preprocess_cn7.py` | 학습 데이터의 TimeStamp/Side 복원 및 기본 전처리 |
| `feature_engineering_cn7.py` | 47개 최종 Feature 생성 |
| `train_cn7.py` | 5개 Seed LightGBM 학습 및 OOF 생성 |
| `evaluate_cn7.py` | 내부 OOF/Threshold 평가 |
| `inference_cn7.py` | LightGBM Risk + Uncertainty 추론 |
| `ood_cn7.py` | 정상 Cycle 기반 PCA OOD 학습/저장/추론 |
| `temporal_ood_cn7.py` | E4/E5 시간검증과 OOD 재현 |
| `test_cn7.py` | unlabeled 추론 + OOD 적용 가능성/Drift 상태 출력 |
| `run_cn7_temporal_ood.py` | Temporal + OOD 평가 실행 진입점 |

---

# 7. 실행환경

Python 3.12 권장:

```powershell
uv venv --python 3.12
.venv\Scripts\activate
uv pip install -r requirements.txt
```

---

# 8. 학습 Pipeline

```powershell
python run_cn7.py
```

실행 순서:

```text
전처리
→ 47 Feature 생성
→ 5 Seed LightGBM OOF/학습
→ 내부 평가
→ PCA OOD baseline 학습
→ 학습데이터 inference sanity check
```

OOD baseline은 supervised 학습과 같은 2020-10-16/19/20 구간의 **정상 Cycle만 사용**합니다.

생성 모델:

```text
models/cn7_pca_ood.pkl
```

---

# 9. Temporal + OOD 재현

```powershell
python run_cn7_temporal_ood.py
```

생성 파일:

```text
outputs/cn7_temporal_ood_side_results.csv
outputs/cn7_temporal_ood_cycle_results.csv
outputs/cn7_temporal_ood_predictions.csv
```

이 스크립트는 E4/E5 label을 OOD fit이나 threshold calibration에 사용하지 않습니다. 각 Future Fold보다 과거의 정상 Cycle만 사용합니다.

---

# 10. Unlabeled 추론

```powershell
python run_cn7_test.py
```

LightGBM은 `moldset_unlabeled_cn7.csv`의 scaled 센서를 사용하고, PCA OOD는 `unlabeled_data.csv`에서 복원한 raw 센서를 사용합니다.

OOD baseline은 학습 종료 이후 **신규 Cycle에만 적용**합니다. Baseline 이전 과거 데이터는 `OOD_Applicable=0`으로 표시하여 현재 기준과 과거 공정을 비교해 대량 OOD로 오해하지 않도록 했습니다.

주요 출력 컬럼:

```text
Risk_Probability
Risk_Score
Risk_Threshold_Probability
Risk_Threshold_Score
Prediction_Stability
Risk_Only_Flag
Uncertainty_Flag
Risk_Uncertainty_Flag
OOD_Score
OOD_Threshold
OOD_Applicable
OOD_Flag
Session_OOD_Rate
Session_Drift_Flag
Model_Trust_Status
Final_Inspection_Flag
Inspection_Priority
Inspection_Action
```

---

# 11. 해석 시 주의사항

- `0.8157 ± 0.0139`는 **개발 데이터 내부 Group-OOF 성능**입니다.
- Risk + Uncertainty Recall `94.12%`도 **내부 OOF Simulation**입니다.
- E4/E5 시간검증에서는 Supervised Risk와 Uncertainty가 미래 Failure Regime을 일반화하지 못했습니다.
- E4 PCA OOD의 Recall `80%`, Precision `100%`는 **사후 안전성 보완실험** 결과입니다.
- E5는 OOD에서도 미탐되었습니다.
- OOD Flag는 불량 확정값이 아니라 분포 이탈 경보입니다.
- Session 전체가 Drift일 경우 개별 OOD를 자동 불량판정으로 사용하면 안 됩니다.
- 실제 배포 전에는 신규 생산기간을 별도 Holdout으로 확보해 OOD threshold까지 다시 독립 검증해야 합니다.

---

# 12. 핵심 요약

```text
CN7 최종 제안
=
LightGBM Risk Ranking
+
Ensemble Uncertainty
+
PCA OOD Safety Layer
+
Session Drift Guard
+
기존 품질검사 유지
```

핵심 성과는 내부 성능만 제시하는 것이 아니라, **미래 Failure Regime에서 분류모델이 실패하는 조건을 확인하고 정상 공정분포 기반 OOD 안전계층으로 E4의 5개 불량 Cycle 중 4개를 1.27% 검사율로 포착한 것**입니다.
