"""Train and document the bias-aware churn model."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pickle
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ETHICAL_WARNING = "⚠️ ETHICAL ALERT: This model identifies financial strain. NEVER use to deny loans automatically."
MODEL_CARD_WARNING = "⚠️ NEVER use for automatic loan denial"


class ModelTrainer:
    """Train a calibrated churn model with subgroup fairness safeguards."""

    def __init__(self, X_train, y_train, region_col="region"):
        self.X_train = pd.DataFrame(X_train).copy()
        self.y_train = pd.Series(y_train).copy()
        self.region_col = region_col
        self.pipeline = None
        self.bias_audit_results = {}
        self.shap_values = None
        self.feature_interpretations = []
        self.deployment_approved = False

    def build_fair_pipeline(self):
        """Create preprocessing, balanced classification, and Platt calibration."""
        numeric_features = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [
            column for column in self.X_train.columns if column not in numeric_features
        ]
        transformers = []
        if numeric_features:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                        ]
                    ),
                    numeric_features,
                )
            )
        if categorical_features:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_features,
                )
            )
        if not transformers:
            raise ValueError("X_train must contain at least one feature column")

        preprocessor = ColumnTransformer(transformers=transformers)
        classifier = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        calibrated_classifier = CalibratedClassifierCV(
            estimator=classifier, method="sigmoid", cv=3
        )
        self.pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", calibrated_classifier),
            ]
        )
        return self.pipeline

    def audit_bias(self):
        """Measure cross-validated recall by region and enforce a 15% disparity limit."""
        if self.pipeline is None:
            self.build_fair_pipeline()
        if self.region_col not in self.X_train:
            raise ValueError(f"Region column {self.region_col!r} is missing from X_train")

        class_counts = self.y_train.value_counts()
        folds = min(5, int(class_counts.min())) if len(class_counts) > 1 else 0
        if folds < 2:
            raise ValueError("At least two examples of each target class are required for auditing")
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        predictions = cross_val_predict(
            self.pipeline, self.X_train, self.y_train, cv=splitter, method="predict"
        )
        regions = self.X_train[self.region_col].astype(str).str.lower()
        recalls = {}
        for region in sorted(regions.unique()):
            mask = regions == region
            if self.y_train[mask].sum() == 0:
                recalls[region] = np.nan
            else:
                recalls[region] = float(recall_score(self.y_train[mask], predictions[mask], zero_division=0))
        valid_recalls = [value for value in recalls.values() if not np.isnan(value)]
        disparity = max(valid_recalls) - min(valid_recalls) if valid_recalls else np.nan
        self.bias_audit_results = {
            "recall_by_region": recalls,
            "max_disparity": float(disparity) if not np.isnan(disparity) else None,
            "fairness_pass": bool(disparity <= 0.15) if not np.isnan(disparity) else False,
            "overall_recall": float(recall_score(self.y_train, predictions, zero_division=0)),
        }
        for region, recall in recalls.items():
            self.bias_audit_results[f"{region}_recall"] = recall
        if not self.bias_audit_results["fairness_pass"]:
            warnings.warn("Regional recall disparity exceeds the 15% fairness threshold.")
        return self.bias_audit_results

    def calibrate_probabilities(self):
        """Return calibrated churn probabilities from the fitted pipeline."""
        if self.pipeline is None or not hasattr(
            self.pipeline.named_steps.get("classifier"), "calibrated_classifiers_"
        ):
            raise RuntimeError("Call train() before requesting calibrated probabilities")
        return self.pipeline.predict_proba(self.X_train)[:, 1]

    def generate_shap_values(self, X=None, top_n=3):
        """Calculate top feature explanations when the optional SHAP package is installed."""
        X = self.X_train if X is None else pd.DataFrame(X)
        try:
            import shap
        except ImportError:
            self.shap_values = None
            self.feature_interpretations = []
            return []

        transformed = self.pipeline.named_steps["preprocessor"].transform(X)
        calibrated = self.pipeline.named_steps["classifier"]
        estimator = calibrated.calibrated_classifiers_[0].estimator
        names = self.pipeline.named_steps["preprocessor"].get_feature_names_out()
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(transformed)
        values = values[1] if isinstance(values, list) else values
        importance = np.abs(values).mean(axis=0)
        top_indices = np.argsort(importance)[::-1][:top_n]
        self.shap_values = values
        self.feature_interpretations = [
            {
                "feature": str(names[index]),
                "mean_absolute_shap": float(importance[index]),
                "business_interpretation": self._business_interpretation(str(names[index])),
            }
            for index in top_indices
        ]
        return self.feature_interpretations

    @staticmethod
    def _business_interpretation(feature):
        label = feature.split("__", 1)[-1].replace("_", " ")
        if any(term in label.lower() for term in ("debt", "income", "payment", "balance")):
            return f"Changes in {label} may signal financial strain and elevated churn risk."
        return f"{label.capitalize()} is associated with the model's churn-risk estimate; investigate context before action."

    def _get_recall(self):
        return self.bias_audit_results.get("overall_recall", 0.0)

    @staticmethod
    def _format_recall(value):
        return value if isinstance(value, str) else f"{value:.1%}"

    def generate_model_card(self, output_path="reports/model_card.md"):
        """Write limitations, failure modes, safeguards, and monitoring requirements."""
        recalls = self.bias_audit_results.get("recall_by_region", {})
        township = recalls.get("township", self.bias_audit_results.get("township_recall", "N/A"))
        urban = recalls.get("urban", self.bias_audit_results.get("urban_recall", "N/A"))
        shap_lines = "\n".join(
            f"- **{item['feature']}**: {item['business_interpretation']}"
            for item in self.feature_interpretations
        ) or "- SHAP is optional; install `shap` to generate feature explanations."
        content = f"""# Churn Prediction Model Card

## Performance
- Overall recall: {self._get_recall():.1%}
- Township recall: {self._format_recall(township)}
- Urban recall: {self._format_recall(urban)}
- Maximum regional recall disparity: {self.bias_audit_results.get('max_disparity', 'N/A')}

## Ethical Constraints
{MODEL_CARD_WARNING}
- Human review is required for every high-risk prediction.
- This model must not be used to exclude, price, or deny financial services.

## Critical Limitations
- Recall can vary across regions and may be unreliable for small subgroups.
- Customers with less than three months of history have high uncertainty.
- Historical labels and economic conditions can encode structural disadvantage.

## Failure Modes
- Distribution shift, missing values, and unrepresented communities can reduce accuracy.
- A calibrated probability is not a causal explanation or evidence of ability to repay.

## SHAP Business Interpretation
{shap_lines}

## Monitoring Plan
- Audit regional recall and calibration monthly; investigate disparity above 15%.
- Monitor missingness, drift, and subgroup sample sizes before retraining quarterly.
- Record human-review outcomes and prohibit automated adverse decisions.
"""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        return output

    def __str__(self):
        recalls = self.bias_audit_results.get("recall_by_region", {})
        township = self._format_recall(recalls.get("township", "N/A"))
        urban = self._format_recall(recalls.get("urban", "N/A"))
        disparity = self._format_recall(self.bias_audit_results.get("max_disparity", "N/A"))
        return (
            f"Recall: {self._get_recall():.0%} (Township: {township} | "
            f"Urban: {urban}) | Max disparity: {disparity}"
        )

    def train(self):
        """Fit, audit, explain, and return the ethically documented pipeline."""
        self.build_fair_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)
        self.audit_bias()
        if not self.bias_audit_results["fairness_pass"]:
            self.generate_model_card("reports/model_card.md")
            raise RuntimeError(
                "Deployment blocked: regional recall disparity exceeds the 15% threshold. "
                "Review the generated model card and retrain before serialization."
            )
        self.deployment_approved = True
        self.generate_shap_values()
        return self.pipeline


def _find_target(frame):
    candidates = ("churn", "churned", "churn_flag", "target", "label", "y")
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError("Could not infer target column; expected one of: " + ", ".join(candidates))


def main():
    data_path = Path("data/processed/engineered_features.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Required Milestone 2 dataset not found: {data_path}")
    frame = pd.read_csv(data_path)
    target = _find_target(frame)
    if "region" not in frame.columns:
        raise ValueError("The engineered dataset must contain a 'region' column for the fairness audit")
    trainer = ModelTrainer(frame.drop(columns=[target]), frame[target])
    trainer.train()
    trainer.generate_model_card("reports/model_card.md")
    Path("models").mkdir(parents=True, exist_ok=True)
    with open("models/churn_pipeline.pkl", "wb") as artifact:
        pickle.dump(trainer.pipeline, artifact)
    print(trainer)


if __name__ == "__main__":
    main()