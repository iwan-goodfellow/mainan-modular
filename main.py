from eda import *
from preprocessing import *
from modeling import *

# Load data
df = ...  # Load dataset
target = "survived"  # Atur sesuai dataset
categorical_cols = [...]  # predefined
numerical_cols = [...]

X = df[categorical_cols + numerical_cols]
y = df[target]

# EDA
eda_univariate(df, categorical_cols, numerical_cols)
eda_bivariate(df, numerical_cols)
print(check_normality(df, numerical_cols))

# Preprocessing
preprocessor = build_preprocessor(
    cat_cols=categorical_cols,
    num_cols=numerical_cols,
    encoding="onehot", scaling=True
)

# Modeling
models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier()
}
results = evaluate_models(X, y, models, task_type="classification", preprocessor=preprocessor)
print(results)
