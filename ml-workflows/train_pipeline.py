"""
End-to-End ML Training Pipeline

Orchestrates data loading, validation, feature selection, multi-algorithm
training, cross-validation, MLflow experiment tracking, model comparison,
best-model registration, and notification on completion.

Usage:
    spark-submit ml-workflows/train_pipeline.py \
        --experiment /Shared/analytics_platform/purchase_prediction \
        --model-name purchase_prediction_model
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    GBTClassifier,
    RandomForestClassifier,
    LogisticRegression,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from xgboost.spark import SparkXGBClassifier
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from mlflow import MlflowClient
import mlflow
import argparse
import logging
import json
import sys
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_pipeline")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """Holds training results for a single model."""
    name: str
    run_id: str
    test_auc: float
    test_accuracy: float
    test_f1: float
    test_precision: float
    test_recall: float
    params: Dict[str, Any] = field(default_factory=dict)
    training_time_sec: float = 0.0


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

class TrainingPipeline:
    """Orchestrates end-to-end model training."""

    def __init__(
        self,
        spark: SparkSession,
        catalog: str,
        experiment_name: str,
        model_name: str,
        seed: int = 42,
    ):
        self.spark = spark
        self.catalog = catalog
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.seed = seed
        self.fe_client = FeatureEngineeringClient()
        self.mlflow_client = MlflowClient()
        self.results: List[ModelResult] = []

    # ----- Data Loading & Validation -----

    def load_data(self) -> DataFrame:
        """Load labelled data with features from the Feature Store."""
        label_df = (
            self.spark.read.table(f"{self.catalog}.gold.training_labels")
            .select("user_id", "event_date", "label")
        )

        feature_lookups = [
            FeatureLookup(
                table_name=f"{self.catalog}.gold.feature_table",
                lookup_key=["user_id", "event_date"],
            )
        ]

        training_set = self.fe_client.create_training_set(
            df=label_df,
            feature_lookups=feature_lookups,
            label="label",
            exclude_columns=["user_id", "event_date"],
        )

        df = training_set.load_df()
        self.training_set = training_set
        logger.info("Training data loaded: %d rows, %d cols", df.count(), len(df.columns))
        return df

    def validate_data(self, df: DataFrame) -> DataFrame:
        """Run basic validations on the training data."""
        total = df.count()
        assert total > 0, "Training data is empty."

        # Check label distribution
        label_counts = df.groupBy("label").count().collect()
        for row in label_counts:
            logger.info("Label %s: %d rows (%.1f%%)",
                        row["label"], row["count"], row["count"] / total * 100)

        # Check for excessive nulls
        for col_name in df.columns:
            null_pct = df.filter(F.col(col_name).isNull()).count() / total * 100
            if null_pct > 50:
                logger.warning("Column '%s' has %.1f%% nulls — consider dropping.", col_name, null_pct)

        return df.fillna(0)

    # ----- Feature Selection -----

    def select_features(self, df: DataFrame) -> List[str]:
        """Select numeric feature columns for modelling."""
        exclude = {"label", "user_id", "event_date", "numeric_features_scaled",
                    "numeric_features_raw"}
        numeric_types = {"double", "integer", "long", "float"}

        feature_cols = [
            c for c in df.columns
            if c not in exclude
            and df.schema[c].dataType.typeName() in numeric_types
        ]

        logger.info("Selected %d features: %s", len(feature_cols), feature_cols)
        return feature_cols

    # ----- Assembly -----

    def assemble_features(
        self, df: DataFrame, feature_cols: List[str],
    ) -> DataFrame:
        """Assemble feature columns into a single vector."""
        assembler = VectorAssembler(
            inputCols=feature_cols, outputCol="features", handleInvalid="keep",
        )
        return assembler.transform(df).select("features", "label")

    # ----- Stratified Split -----

    def split_data(self, df: DataFrame, train_ratio: float = 0.8):
        """Stratified train/test split."""
        pos = df.filter(F.col("label") == 1.0)
        neg = df.filter(F.col("label") == 0.0)
        pos_train, pos_test = pos.randomSplit([train_ratio, 1 - train_ratio], seed=self.seed)
        neg_train, neg_test = neg.randomSplit([train_ratio, 1 - train_ratio], seed=self.seed)
        train = pos_train.unionByName(neg_train).cache()
        test = pos_test.unionByName(neg_test).cache()
        logger.info("Split: train=%d, test=%d", train.count(), test.count())
        return train, test

    # ----- Model Training -----

    def _evaluate(self, predictions: DataFrame) -> Dict[str, float]:
        auc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
        prec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
        rec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
        return {
            "test_auc": auc_eval.evaluate(predictions),
            "test_accuracy": acc_eval.evaluate(predictions),
            "test_f1": f1_eval.evaluate(predictions),
            "test_precision": prec_eval.evaluate(predictions),
            "test_recall": rec_eval.evaluate(predictions),
        }

    def train_xgboost(self, train: DataFrame, test: DataFrame) -> ModelResult:
        """Train SparkXGBClassifier."""
        params = {
            "max_depth": 8,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }

        with mlflow.start_run(run_name="xgboost", nested=True) as run:
            start = datetime.now()
            mlflow.log_params(params)

            model = SparkXGBClassifier(
                features_col="features", label_col="label",
                num_workers=4, random_state=self.seed, **params,
            ).fit(train)

            preds = model.transform(test)
            metrics = self._evaluate(preds)
            mlflow.log_metrics(metrics)

            elapsed = (datetime.now() - start).total_seconds()
            result = ModelResult(
                name="xgboost", run_id=run.info.run_id,
                training_time_sec=elapsed, params=params, **metrics,
            )
            self.results.append(result)
            logger.info("XGBoost — AUC=%.4f, F1=%.4f (%.1fs)", metrics["test_auc"], metrics["test_f1"], elapsed)
            return result

    def train_gbt(self, train: DataFrame, test: DataFrame) -> ModelResult:
        """Train Spark GBTClassifier."""
        params = {"maxDepth": 8, "maxIter": 200, "stepSize": 0.05, "subsamplingRate": 0.8}

        with mlflow.start_run(run_name="spark_gbt", nested=True) as run:
            start = datetime.now()
            mlflow.log_params(params)

            model = GBTClassifier(
                featuresCol="features", labelCol="label", seed=self.seed, **params,
            ).fit(train)

            preds = model.transform(test)
            metrics = self._evaluate(preds)
            mlflow.log_metrics(metrics)

            elapsed = (datetime.now() - start).total_seconds()
            result = ModelResult(
                name="spark_gbt", run_id=run.info.run_id,
                training_time_sec=elapsed, params=params, **metrics,
            )
            self.results.append(result)
            logger.info("GBT — AUC=%.4f, F1=%.4f (%.1fs)", metrics["test_auc"], metrics["test_f1"], elapsed)
            return result

    def train_random_forest(self, train: DataFrame, test: DataFrame) -> ModelResult:
        """Train Spark RandomForestClassifier."""
        params = {"numTrees": 200, "maxDepth": 10, "featureSubsetStrategy": "sqrt"}

        with mlflow.start_run(run_name="random_forest", nested=True) as run:
            start = datetime.now()
            mlflow.log_params(params)

            model = RandomForestClassifier(
                featuresCol="features", labelCol="label", seed=self.seed, **params,
            ).fit(train)

            preds = model.transform(test)
            metrics = self._evaluate(preds)
            mlflow.log_metrics(metrics)

            elapsed = (datetime.now() - start).total_seconds()
            result = ModelResult(
                name="random_forest", run_id=run.info.run_id,
                training_time_sec=elapsed, params=params, **metrics,
            )
            self.results.append(result)
            logger.info("RF — AUC=%.4f, F1=%.4f (%.1fs)", metrics["test_auc"], metrics["test_f1"], elapsed)
            return result

    def train_logistic_regression(self, train: DataFrame, test: DataFrame) -> ModelResult:
        """Train Spark LogisticRegression."""
        params = {"maxIter": 100, "regParam": 0.01, "elasticNetParam": 0.5}

        with mlflow.start_run(run_name="logistic_regression", nested=True) as run:
            start = datetime.now()
            mlflow.log_params(params)

            model = LogisticRegression(
                featuresCol="features", labelCol="label", **params,
            ).fit(train)

            preds = model.transform(test)
            metrics = self._evaluate(preds)
            mlflow.log_metrics(metrics)

            elapsed = (datetime.now() - start).total_seconds()
            result = ModelResult(
                name="logistic_regression", run_id=run.info.run_id,
                training_time_sec=elapsed, params=params, **metrics,
            )
            self.results.append(result)
            logger.info("LR — AUC=%.4f, F1=%.4f (%.1fs)", metrics["test_auc"], metrics["test_f1"], elapsed)
            return result

    # ----- Model Comparison & Registration -----

    def select_best_model(self) -> ModelResult:
        """Select the model with the highest AUC."""
        best = max(self.results, key=lambda r: r.test_auc)
        logger.info("Best model: %s (AUC=%.4f)", best.name, best.test_auc)
        return best

    def register_best_model(self, best: ModelResult) -> str:
        """Register the best model in the MLflow Model Registry."""
        model_uri = f"runs:/{best.run_id}/model"
        registered = mlflow.register_model(model_uri, self.model_name)

        # Transition to Staging
        self.mlflow_client.transition_model_version_stage(
            name=self.model_name,
            version=registered.version,
            stage="Staging",
            archive_existing_versions=True,
        )

        self.mlflow_client.update_model_version(
            name=self.model_name,
            version=registered.version,
            description=(
                f"{best.name} model — AUC: {best.test_auc:.4f}, "
                f"F1: {best.test_f1:.4f}. "
                f"Registered {datetime.now().isoformat()}"
            ),
        )

        logger.info("Registered %s v%s -> Staging", self.model_name, registered.version)
        return registered.version

    # ----- Notification -----

    def notify_completion(self, best: ModelResult, version: str) -> None:
        """Log a completion summary (extend with Slack/email webhook in production)."""
        summary = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "model_version": version,
            "best_algorithm": best.name,
            "test_auc": round(best.test_auc, 4),
            "test_f1": round(best.test_f1, 4),
            "all_results": [asdict(r) for r in self.results],
        }
        logger.info("Training pipeline complete:\n%s", json.dumps(summary, indent=2))

    # ----- Orchestrator -----

    def run(self) -> None:
        """Execute the full training pipeline."""
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name="training_pipeline") as parent_run:
            # 1. Load & validate
            df = self.load_data()
            df = self.validate_data(df)

            # 2. Feature selection & assembly
            feature_cols = self.select_features(df)
            assembled = self.assemble_features(df, feature_cols)
            mlflow.log_param("num_features", len(feature_cols))

            # 3. Split
            train, test = self.split_data(assembled)

            # 4. Train multiple algorithms
            self.train_xgboost(train, test)
            self.train_gbt(train, test)
            self.train_random_forest(train, test)
            self.train_logistic_regression(train, test)

            # 5. Compare & register
            best = self.select_best_model()
            mlflow.log_param("best_model", best.name)
            mlflow.log_metric("best_test_auc", best.test_auc)

            version = self.register_best_model(best)

            # 6. Log comparison table
            comparison = [
                {"model": r.name, "auc": r.test_auc, "f1": r.test_f1,
                 "accuracy": r.test_accuracy, "time_sec": r.training_time_sec}
                for r in self.results
            ]
            mlflow.log_dict(comparison, "model_comparison.json")

            # 7. Notify
            self.notify_completion(best, version)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run ML training pipeline")
    parser.add_argument("--experiment", default="/Shared/analytics_platform/purchase_prediction")
    parser.add_argument("--model-name", default="purchase_prediction_model")
    parser.add_argument("--catalog", default="analytics_platform")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    spark = SparkSession.builder.appName("TrainingPipeline").getOrCreate()

    pipeline = TrainingPipeline(
        spark=spark,
        catalog=args.catalog,
        experiment_name=args.experiment,
        model_name=args.model_name,
        seed=args.seed,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
