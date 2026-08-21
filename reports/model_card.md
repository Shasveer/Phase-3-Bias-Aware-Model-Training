# Churn Prediction Model Card

## Performance
- Overall recall: 100.0%
- Township recall: 100.0%
- Urban recall: 100.0%
- Maximum regional recall disparity: 0.0

## Ethical Constraints
⚠️ NEVER use for automatic loan denial
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
- **categorical__customer_segment_stable**: Customer segment stable is associated with the model's churn-risk estimate; investigate context before action.
- **numeric__missed_payments**: Changes in missed payments may signal financial strain and elevated churn risk.
- **numeric__debt_to_income**: Changes in debt to income may signal financial strain and elevated churn risk.

## Monitoring Plan
- Audit regional recall and calibration monthly; investigate disparity above 15%.
- Monitor missingness, drift, and subgroup sample sizes before retraining quarterly.
- Record human-review outcomes and prohibit automated adverse decisions.
