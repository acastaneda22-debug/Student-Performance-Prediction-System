# train_model.py
import pandas as pd
from ucimlrepo import fetch_ucirepo

# ---------- Activity A: retrieve and inspect the dataset ----------
# Fetch UCI Student Performance dataset (ID 320)
student = fetch_ucirepo(id=320)

# Combine student features and grade targets (G1, G2, G3)
df = pd.concat([student.data.features, student.data.targets], axis=1)
df.to_csv("student_data.csv", index=False)

print("Rows and columns:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values per column:")
print(df.isnull().sum())
# NOTE: the original UCI Mathematics file has 395 rows; the Portuguese file has 649.
# Check the row count printed above and see README.md ("Dataset note").

# ---------- Activity B: preprocess, train, evaluate ----------
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Target: 1 = at risk (G3 < 10); 0 = not at risk
df = df.drop_duplicates()
y = (df["G3"] < 10).astype(int)

# Never use G3 as a predictor (target leakage)
X = df.drop(columns=["G3"])

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = round(accuracy_score(y_test, predictions), 3)
report = classification_report(
    y_test, predictions,
    target_names=["Not at risk", "At risk"],
    zero_division=0,
)
print("Accuracy:", accuracy)
print(report)

# Save results so they can be copied into the submission
with open("evaluation_results.txt", "w") as f:
    f.write(f"Rows used (after drop_duplicates): {len(df)}\n")
    f.write(f"Accuracy: {accuracy}\n\n")
    f.write(report)

joblib.dump(model, "student_model.pkl")
print("Saved student_model.pkl and evaluation_results.txt")
