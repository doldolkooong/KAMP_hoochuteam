from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config_cn7 import (
    BEST_PARAMS,
    CV_SEEDS,
    OOD_TEMPORAL_CYCLE_RESULTS_PATH,
    OOD_TEMPORAL_PREDICTIONS_PATH,
    OOD_TEMPORAL_SIDE_RESULTS_PATH,
    OPERATION_THRESHOLD,
    OUTPUT_DIR,
    RAW_LABELED_PATH,
    ROLL_WINDOW,
    SENSOR_COLS,
    SESSION_GAP_SECONDS,
    TARGET,
    TIME_COL,
)
from .ood_cn7 import (
    build_cycle_frame,
    fit_ood_artifact,
    prepare_cn7_raw_rows,
    score_ood_cycles,
)


TEMPORAL_FOLDS = [
    {
        "Fold": "Future_E4_2020-10-27",
        "Train_End": pd.Timestamp("2020-10-27 00:00:00"),
        "Test_Start": pd.Timestamp("2020-10-27 00:00:00"),
        "Test_End": pd.Timestamp("2020-10-28 00:00:00"),
    },
    {
        "Fold": "Future_E5_2020-11-03",
        "Train_End": pd.Timestamp("2020-11-03 00:00:00"),
        "Test_Start": pd.Timestamp("2020-11-03 00:00:00"),
        "Test_End": pd.Timestamp("2020-11-04 00:00:00"),
    },
]


def _metrics(y_true, pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Flag_Count": int(np.sum(pred)),
        "Flag_Rate_pct": float(np.mean(pred) * 100),
    }


def _build_side_unit(raw_rows: pd.DataFrame) -> pd.DataFrame:
    # 동일 TimeStamp+Side 중복은 센서/Target이 같은 반복행이므로 1행으로 압축한다.
    side = (
        raw_rows.groupby([TIME_COL, "Side"], as_index=False)
        .agg({**{c: "median" for c in SENSOR_COLS}, TARGET: "max"})
        .sort_values([TIME_COL, "Side"])
        .reset_index(drop=True)
    )
    side["Side_RH"] = (side["Side"] == "RH").astype(int)
    return side


def _add_rolling_features(side_unit: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # 같은 TimeStamp에서 Side별 센서가 같은지 확인한다.
    conflict = (
        side_unit[[TIME_COL] + SENSOR_COLS]
        .drop_duplicates()
        .groupby(TIME_COL)
        .size()
        .gt(1)
        .sum()
    )
    if conflict:
        raise ValueError(f"동일 Cycle에서 센서 조합이 다른 TimeStamp가 {conflict}개 있습니다.")

    cycle_sensor = (
        side_unit[[TIME_COL] + SENSOR_COLS]
        .drop_duplicates(subset=[TIME_COL])
        .sort_values(TIME_COL)
        .reset_index(drop=True)
    )
    cycle_sensor["_gap_sec"] = cycle_sensor[TIME_COL].diff().dt.total_seconds()
    cycle_sensor["_session_id"] = (
        cycle_sensor["_gap_sec"].isna()
        | (cycle_sensor["_gap_sec"] > SESSION_GAP_SECONDS)
    ).cumsum()

    roll_cols = []
    for col in SENSOR_COLS:
        new_col = f"{col}_roll5_mean"
        cycle_sensor[new_col] = (
            cycle_sensor.groupby("_session_id")[col]
            .transform(
                lambda s: s.shift(1).rolling(window=ROLL_WINDOW, min_periods=1).mean()
            )
        )
        roll_cols.append(new_col)

    model_df = side_unit.merge(
        cycle_sensor[[TIME_COL] + roll_cols],
        on=TIME_COL,
        how="left",
        validate="many_to_one",
    )
    return model_df, roll_cols


def evaluate_temporal_ood(
    raw_labeled_path=RAW_LABELED_PATH,
    side_results_path=OOD_TEMPORAL_SIDE_RESULTS_PATH,
    cycle_results_path=OOD_TEMPORAL_CYCLE_RESULTS_PATH,
    predictions_path=OOD_TEMPORAL_PREDICTIONS_PATH,
):
    raw = pd.read_csv(raw_labeled_path)
    raw_rows = prepare_cn7_raw_rows(raw)
    side_unit = _build_side_unit(raw_rows)
    model_df, roll_cols = _add_rolling_features(side_unit)
    feature_cols = SENSOR_COLS + ["Side_RH"] + roll_cols

    cycle_all = build_cycle_frame(raw_rows)

    side_results = []
    cycle_results = []
    prediction_parts = []

    for fold in TEMPORAL_FOLDS:
        fold_name = fold["Fold"]
        train_df = model_df[model_df[TIME_COL] < fold["Train_End"]].copy()
        test_df = model_df[
            (model_df[TIME_COL] >= fold["Test_Start"])
            & (model_df[TIME_COL] < fold["Test_End"])
        ].copy()

        if test_df.empty:
            continue

        x_train = train_df[feature_cols]
        y_train = train_df[TARGET].astype(int)
        x_test = test_df[feature_cols]
        y_test = test_df[TARGET].astype(int).to_numpy()

        seed_probs = []
        for seed in CV_SEEDS:
            model = LGBMClassifier(
                **BEST_PARAMS,
                class_weight="balanced",
                random_state=seed,
                feature_fraction_seed=seed,
                bagging_seed=seed,
                data_random_seed=seed,
                verbose=-1,
            )
            model.fit(x_train, y_train)
            seed_probs.append(model.predict_proba(x_test)[:, 1])

        matrix = np.column_stack(seed_probs)
        risk_probability = matrix.mean(axis=1)
        risk_std = matrix.std(axis=1)
        votes = (matrix >= OPERATION_THRESHOLD).sum(axis=1)
        unstable = (votes > 0) & (votes < len(CV_SEEDS))

        # IMPORTANT: OPERATION_THRESHOLD=0.060은 0~1 Risk_Probability 기준이다.
        # Risk_Score_pct(=Probability*100)로 비교할 경우 동등 threshold는 6.0이다.
        risk_only = (risk_probability >= OPERATION_THRESHOLD).astype(int)
        risk_uncertainty = ((risk_probability >= OPERATION_THRESHOLD) | unstable).astype(int)

        pr_auc = (
            average_precision_score(y_test, risk_probability)
            if len(np.unique(y_test)) == 2
            else np.nan
        )

        # OOD는 동일 fold의 미래 라벨을 사용하지 않고, 그 이전 정상 Cycle로만 fit/calibration.
        normal_past = cycle_all[
            (cycle_all[TIME_COL] < fold["Train_End"])
            & (cycle_all[TARGET] == 0)
        ].copy()
        ood_artifact = fit_ood_artifact(normal_past)

        test_cycle = cycle_all[
            (cycle_all[TIME_COL] >= fold["Test_Start"])
            & (cycle_all[TIME_COL] < fold["Test_End"])
        ].copy()
        test_cycle = score_ood_cycles(test_cycle, ood_artifact)

        pred_df = test_df[[TIME_COL, "Side", "Side_RH", TARGET]].copy()
        pred_df["Fold"] = fold_name
        pred_df["Risk_Probability"] = risk_probability
        pred_df["Risk_Score_pct"] = risk_probability * 100
        pred_df["Risk_STD"] = risk_std
        pred_df["Positive_Votes"] = votes
        pred_df["Prediction_Stability"] = np.where(unstable, "Unstable", "Stable")
        pred_df["Risk_Only_Flag"] = risk_only
        pred_df["Risk_Uncertainty_Flag"] = risk_uncertainty

        pred_df = pred_df.merge(
            test_cycle[[TIME_COL, "OOD_Score", "OOD_Threshold", "OOD_Flag"]],
            on=TIME_COL,
            how="left",
            validate="many_to_one",
        )
        pred_df["OOD_Flag"] = pred_df["OOD_Flag"].fillna(0).astype(int)
        pred_df["Risk_OOD_Flag"] = (
            (pred_df["Risk_Only_Flag"] == 1) | (pred_df["OOD_Flag"] == 1)
        ).astype(int)
        pred_df["Risk_Uncertainty_OOD_Flag"] = (
            (pred_df["Risk_Uncertainty_Flag"] == 1) | (pred_df["OOD_Flag"] == 1)
        ).astype(int)

        for name, col in [
            ("Risk Only", "Risk_Only_Flag"),
            ("Risk + Uncertainty", "Risk_Uncertainty_Flag"),
            ("OOD Only", "OOD_Flag"),
            ("Risk + OOD", "Risk_OOD_Flag"),
            ("Risk + Uncertainty + OOD", "Risk_Uncertainty_OOD_Flag"),
        ]:
            row = {
                "Fold": fold_name,
                "Policy": name,
                "Risk_PR_AUC": pr_auc,
                "Risk_Threshold_Probability": OPERATION_THRESHOLD,
                "Risk_Threshold_Score_pct": OPERATION_THRESHOLD * 100,
            }
            row.update(_metrics(y_test, pred_df[col].astype(int).to_numpy()))
            side_results.append(row)

        cycle_eval = (
            pred_df.groupby(TIME_COL, as_index=False)
            .agg(
                Actual_Cycle=(TARGET, "max"),
                Risk_Only_Cycle=("Risk_Only_Flag", "max"),
                Risk_Uncertainty_Cycle=("Risk_Uncertainty_Flag", "max"),
                OOD_Cycle=("OOD_Flag", "max"),
                Risk_OOD_Cycle=("Risk_OOD_Flag", "max"),
                Risk_Uncertainty_OOD_Cycle=("Risk_Uncertainty_OOD_Flag", "max"),
            )
        )
        for name, col in [
            ("Risk Only", "Risk_Only_Cycle"),
            ("Risk + Uncertainty", "Risk_Uncertainty_Cycle"),
            ("OOD Only", "OOD_Cycle"),
            ("Risk + OOD", "Risk_OOD_Cycle"),
            ("Risk + Uncertainty + OOD", "Risk_Uncertainty_OOD_Cycle"),
        ]:
            row = {"Fold": fold_name, "Policy": name}
            row.update(
                _metrics(
                    cycle_eval["Actual_Cycle"].astype(int).to_numpy(),
                    cycle_eval[col].astype(int).to_numpy(),
                )
            )
            cycle_results.append(row)

        prediction_parts.append(pred_df)

        print("\n" + "=" * 110)
        print(fold_name)
        print("=" * 110)
        print(f"Train: {train_df[TIME_COL].min()} ~ {train_df[TIME_COL].max()}")
        print(f"Test : {test_df[TIME_COL].min()} ~ {test_df[TIME_COL].max()}")
        print(f"PR-AUC: {pr_auc:.4f}")
        print(
            f"OOD fit/cal: {ood_artifact['fit_cycle_count']}/{ood_artifact['calibration_cycle_count']} | "
            f"PCA components={ood_artifact['pca'].n_components_} | "
            f"threshold={ood_artifact['threshold']:.6f}"
        )

    side_df = pd.DataFrame(side_results)
    cycle_df = pd.DataFrame(cycle_results)
    pred_all = pd.concat(prediction_parts, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    side_df.to_csv(side_results_path, index=False, encoding="utf-8-sig")
    cycle_df.to_csv(cycle_results_path, index=False, encoding="utf-8-sig")
    pred_all.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    print("\n[SIDE-LEVEL TEMPORAL + OOD]")
    print(side_df.round(4).to_string(index=False))
    print("\n[CYCLE-LEVEL TEMPORAL + OOD]")
    print(cycle_df.round(4).to_string(index=False))

    return side_df, cycle_df, pred_all


if __name__ == "__main__":
    evaluate_temporal_ood()
