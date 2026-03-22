"""
Hyperparameter Tuning with Hyperopt + SparkTrials

Distributed hyperparameter search using Tree of Parzen Estimators (TPE)
with MLflow tracking for every trial. Runs across Spark workers using
SparkTrials for parallelism.

Usage:
    spark-submit ml-workflows/hyperparameter_tuning.py \
        --max-evals 100 \
        --parallelism 8
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from xgboost.spark import SparkXGBClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval
from hyperopt import SparkTrials
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
import mlflow
from mlflow import MlflowClient
import argparse
import logging
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hyperparameter_tuning")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CATALOG = "analytics_platform"
EXPERIMENT_NAME = "/Shared/analytics_platform/hyperparameter_tuning"
MODEL_NAME = "purchase_prediction_model"
SEED = 42


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

SEARCH_SPACE = {
    "max_depth": hp.choice("max_depth", [4, 6, 8, 10, 12]),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.3)),
    "n_estimators": hp.choice("n_estimators", [100, 200, 300, 500, 800]),
    "subsample": hp.uniform("subsample", 0.5, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
    "min_child_weight": hp.choice("min_child_weight", [1, 3, 5, 7, 10]),
    "gamma": hp.uniform("gamma", 0.0, 0.5),
    "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-5), np.log(10.0)),
    "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-5), np.log(10.0)),
    "scale_pos_weight": hp.uniform("scale_pos_weight", 0.5, 5.0),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(spark: SparkSession) -> tuple:
    """Load and split training data."""
    fe_client = FeatureEngineeringClient()

    label_df = (
        spark.read.table(f"{CATALOG}.gold.training_labels")
        .select("user_id", "event_date", "label")
    )

    feature_lookups = [
        FeatureLookup(
            table_name=f"{CATALOG}.gold.feature_table",
            lookup_key=["user_id", "event_date"],
        )
    ]

    training_set = fe_client.create_training_set(
        df=label_df,
        feature_lookups=feature_lookups,
        label="label",
        exclude_columns=["user_id", "event_date"],
    )

    df = training_set.load_df().fillna(0)

    # Select numeric features
    exclude = {"label", "numeric_features_scaled", "numeric_features_raw"}
    numeric_types = {"double", "integer", "long", "float"}
    feature_cols = [
        c for c in df.columns
        if c not in exclude and df.schema[c].dataType.typeName() in numeric_types
    ]

    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="features", handleInvalid="keep",
    )
    assembled = assembler.transform(df).select("features", "label")

    # Stratified split
    pos = assembled.filter(F.col("label") == 1.0)
    neg = assembled.filter(F.col("label") == 0.0)
    pos_train, pos_test = pos.randomSplit([0.8, 0.2], seed=SEED)
    neg_train, neg_test = neg.randomSplit([0.8, 0.2], seed=SEED)

    train = pos_train.unionByName(neg_train).cache()
    test = pos_test.unionByName(neg_test).cache()

    logger.info("Data loaded — train: %d, test: %d, features: %d",
                train.count(), test.count(), len(feature_cols))

    return train, test, feature_cols, training_set


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def create_objective(train_df: DataFrame, test_df: DataFrame):
    """
    Create a Hyperopt objective function that trains an XGBoost model
    and returns the negative AUC (since Hyperopt minimizes).
    """
    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )

    def objective(params: Dict[str, Any]) -> Dict[str, Any]:
        """Train one model with the given hyperparameters and return loss."""
        with mlflow.start_run(nested=True):
            try:
                # Ensure integer params
                params["max_depth"] = int(params["max_depth"])
                params["n_estimators"] = int(params["n_estimators"])
                params["min_child_weight"] = int(params["min_child_weight"])

                mlflow.log_params(params)

                model = SparkXGBClassifier(
                    features_col="features",
                    label_col="label",
                    random_state=SEED,
                    num_workers=4,
                    use_gpu=False,
                    eval_metric="auc",
                    **params,
                ).fit(train_df)

                predictions = model.transform(test_df)
                auc = evaluator.evaluate(predictions)

                mlflow.log_metric("test_auc", auc)
                logger.info("Trial — AUC=%.4f | params=%s",
                            auc, {k: round(v, 4) if isinstance(v, float) else v for k, v in params.items()})

                return {"loss": -auc, "status": STATUS_OK, "auc": auc}

            except Exception as exc:
                logger.error("Trial failed: %s", exc)
                mlflow.log_param("error", str(exc))
                return {"loss": 0, "status": STATUS_FAIL}

    return objective


# ---------------------------------------------------------------------------
# Tuning orchestrator
# ---------------------------------------------------------------------------

def run_hyperparameter_tuning(
    spark: SparkSession,
    max_evals: int = 50,
    parallelism: int = 4,
) -> Dict[str, Any]:
    """Run distributed hyperparameter search with Hyperopt + SparkTrials."""

    # Load data
    train_df, test_df, feature_cols, training_set = load_training_data(spark)

    # Set up MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="hyperopt_tuning") as parent_run:
        parent_run_id = parent_run.info.run_id
        mlflow.log_param("max_evals", max_evals)
        mlflow.log_param("parallelism", parallelism)
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("search_algorithm", "tpe")
        mlflow.set_tag("pipeline_stage", "hyperparameter_tuning")

        # Create objective
        objective = create_objective(train_df, test_df)

        # SparkTrials for distributed execution
        spark_trials = SparkTrials(parallelism=parallelism)

        logger.info(
            "Starting Hyperopt search: max_evals=%d, parallelism=%d",
            max_evals, parallelism,
        )

        best_params = fmin(
            fn=objective,
            space=SEARCH_SPACE,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=spark_trials,
            rstate=np.random.default_rng(SEED),
        )

        # Resolve hp.choice indices to actual values
        best_resolved = space_eval(SEARCH_SPACE, best_params)
        best_resolved["max_depth"] = int(best_resolved["max_depth"])
        best_resolved["n_estimators"] = int(best_resolved["n_estimators"])
        best_resolved["min_child_weight"] = int(best_resolved["min_child_weight"])

        # Find best AUC from trials
        trial_results = [
            t["result"] for t in spark_trials.trials
            if t["result"]["status"] == STATUS_OK
        ]
        best_auc = max(r["auc"] for r in trial_results) if trial_results else 0.0

        mlflow.log_params({f"best_{k}": v for k, v in best_resolved.items()})
        mlflow.log_metric("best_auc", best_auc)
        mlflow.log_metric("total_trials", len(spark_trials.trials))
        mlflow.log_metric("successful_trials", len(trial_results))

        # Log all trial results
        all_trials = []
        for i, trial in enumerate(spark_trials.trials):
            trial_info = {
                "trial": i,
                "status": trial["result"]["status"],
                "auc": trial["result"].get("auc", None),
            }
            all_trials.append(trial_info)

        mlflow.log_dict(all_trials, "trial_results.json")

        logger.info("="*60)
        logger.info("Best parameters: %s", json.dumps(
            {k: round(v, 4) if isinstance(v, float) else v for k, v in best_resolved.items()},
            indent=2,
        ))
        logger.info("Best AUC: %.4f", best_auc)
        logger.info("="*60)

        # Retrain final model with best params
        logger.info("Retraining final model with best parameters...")
        with mlflow.start_run(run_name="best_model_retrained", nested=True) as best_run:
            mlflow.log_params(best_resolved)

            final_model = SparkXGBClassifier(
                features_col="features", label_col="label",
                random_state=SEED, num_workers=4, eval_metric="auc",
                **best_resolved,
            ).fit(train_df)

            evaluator = BinaryClassificationEvaluator(
                labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC",
            )
            final_preds = final_model.transform(test_df)
            final_auc = evaluator.evaluate(final_preds)
            mlflow.log_metric("final_test_auc", final_auc)

            # Register the final model
            fe_client = FeatureEngineeringClient()
            fe_client.log_model(
                model=final_model,
                artifact_path="model",
                flavor=mlflow.spark,
                training_set=training_set,
                registered_model_name=MODEL_NAME,
            )

            logger.info("Final model retrained — AUC=%.4f, registered as %s", final_auc, MODEL_NAME)

    return {
        "best_params": best_resolved,
        "best_auc": best_auc,
        "parent_run_id": parent_run_id,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Hyperopt")
    parser.add_argument("--max-evals", type=int, default=50)
    parser.add_argument("--parallelism", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    spark = SparkSession.builder.appName("HyperparameterTuning").getOrCreate()

    result = run_hyperparameter_tuning(
        spark=spark,
        max_evals=args.max_evals,
        parallelism=args.parallelism,
    )

    logger.info("Hyperparameter tuning complete: %s", json.dumps(
        {k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()},
        indent=2,
    ))


if __name__ == "__main__":
    main()
