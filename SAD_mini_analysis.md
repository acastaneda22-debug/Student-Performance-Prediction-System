# SAD Mini-Analysis — Student Performance Prediction System

| SAD element | Response |
|---|---|
| Problem statement | Academic risk is often noticed only after poor results. The prototype explores early identification of students who may need extra support, using available student indicators. |
| Users | Faculty member or academic analyst. |
| Inputs | Student attributes and academic indicators (study time, absences, past failures, G1, G2, etc.), supplied as a CSV file. |
| Process | Validate data (check required columns) → preprocess (impute, scale, one-hot encode) → apply trained Random Forest model → classify risk. |
| Outputs | AT RISK / NOT AT RISK label per student, downloadable results CSV, and a model evaluation summary (accuracy, precision, recall, F1-score). |
| Key limitation | A model trained on one Portuguese secondary-school dataset may not generalize to another institution or population; predictions must not be the sole basis for real student decisions. |

**Use case:** The user uploads a CSV → the system checks required columns → the model predicts risk → the system displays and exports the results.
