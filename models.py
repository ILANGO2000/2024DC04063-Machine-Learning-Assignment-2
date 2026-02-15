# Basic imports for building machine learning pipelines
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
import numpy as np


# Helper function to prepare column transformation
def prepare_transformer(df_input):

    # Separate columns based on data type
    categorical_fields = df_input.select_dtypes(include=["object"]).columns
    numerical_fields = df_input.select_dtypes(exclude=["object"]).columns

    transformer = ColumnTransformer(
        [
            ("numeric_block", "passthrough", numerical_fields),
            (
                "categorical_block",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_fields
            ),
        ]
    )

    return transformer


# Function to assemble the full pipeline
def assemble_pipeline(estimator, df_input, dense_required=False):

    column_transformer = prepare_transformer(df_input)

    pipeline_components = [
        ("feature_step", column_transformer)
    ]

    # Some algorithms need dense input format
    if dense_required:
        dense_step = FunctionTransformer(lambda data: np.asarray(data))
        pipeline_components.append(("dense_step", dense_step))

    pipeline_components.append(("estimator_step", estimator))

    model_pipeline = Pipeline(pipeline_components)

    return model_pipeline


# Function to choose algorithm and return ready pipeline
def fetch_pipeline(algorithm_label, df_input):

    algorithm_pool = {
        "Logistic Regression": LogisticRegression(max_iter=1000),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            random_state=42
        ),

        "KNN": KNeighborsClassifier(n_neighbors=5),

        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            n_jobs=-1,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=50,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=-1,
            eval_metric="logloss",
            random_state=42
        )
    }

    # Handle Naive Bayes separately
    if algorithm_label == "Naive Bayes":
        return assemble_pipeline(
            GaussianNB(),
            df_input,
            dense_required=True
        )

    selected_estimator = algorithm_pool[algorithm_label]

    return assemble_pipeline(selected_estimator, df_input)