
# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# for model serialization
import joblib
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

# Xtrain/Xtest/ytrain/ytest are downloaded from the previous job's artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest  = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest  = pd.read_csv("ytest.csv").squeeze()

# Handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])



# Define base XGBoost model
xgb_model = XGBClassifier(scale_pos_weight=class_weight, random_state=42,eval_metric='logloss')


# Grid of parameters to choose from
parameters = {
      "xgbclassifier__n_estimators": [10, 30, 50],
    "xgbclassifier__scale_pos_weight": [1, 2, 5],
    "xgbclassifier__subsample": [0.7, 0.9, 1],
    "xgbclassifier__learning_rate": [0.05, 0.1, 0.2],
    "xgbclassifier__colsample_bytree": [0.7, 0.9, 1],
    "xgbclassifier__colsample_bylevel": [0.5, 0.7, 1]
}
# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, parameters, cv=5, scoring="recall", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_
    print("Best params:", grid_search.best_params_)

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)
    print(classification_report(ytest, y_pred_test))

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"]
    })

    # Save next to app.py so the Streamlit app can load it directly, and log
    # it as an MLflow artifact for traceability
    model_path = "week_3_mls/deployment/best_machine_failure_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")
