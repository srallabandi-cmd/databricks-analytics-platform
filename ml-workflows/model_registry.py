"""
MLflow Model Registry Operations

Manages the full model lifecycle: versioning, stage transitions, model comparison,
automated validation before promotion, and webhook notifications.

Usage:
    python ml-workflows/model_registry.py --action promote --model purchase_prediction_model
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
import mlflow
import argparse
import logging
import json
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("model_registry")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of a model validation check."""
    check_name: str
    passed: bool
    message: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None


# ---------------------------------------------------------------------------
# Model Registry Manager
# ---------------------------------------------------------------------------

class ModelRegistryManager:
    """Manages MLflow Model Registry operations."""

    def __init__(self, spark: SparkSession, model_name: str, catalog: str = "analytics_platform"):
        self.spark = spark
        self.model_name = model_name
        self.catalog = catalog
        self.client = MlflowClient()

    # ----- Versioning -----

    def list_versions(self, stages: Optional[List[str]] = None) -> List[ModelVersion]:
        """List all model versions, optionally filtered by stage."""
        if stages:
            versions = []
            for stage in stages:
                versions.extend(self.client.get_latest_versions(self.model_name, stages=[stage]))
            return versions
        else:
            # Get all versions using search
            return list(self.client.search_model_versions(f"name='{self.model_name}'"))

    def get_version_details(self, version: str) -> Dict:
        """Get detailed information about a specific model version."""
        mv = self.client.get_model_version(self.model_name, version)
        run = self.client.get_run(mv.run_id)
        return {
            "name": mv.name,
            "version": mv.version,
            "stage": mv.current_stage,
            "status": mv.status,
            "run_id": mv.run_id,
            "description": mv.description,
            "creation_timestamp": mv.creation_timestamp,
            "last_updated_timestamp": mv.last_updated_timestamp,
            "tags": dict(mv.tags) if mv.tags else {},
            "metrics": run.data.metrics,
            "params": run.data.params,
        }

    # ----- Stage Transitions -----

    def transition_stage(
        self,
        version: str,
        target_stage: str,
        archive_existing: bool = True,
    ) -> ModelVersion:
        """Transition a model version to a new stage."""
        valid_stages = {"None", "Staging", "Production", "Archived"}
        if target_stage not in valid_stages:
            raise ValueError(f"Invalid stage: {target_stage}. Valid: {valid_stages}")

        mv = self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage=target_stage,
            archive_existing_versions=archive_existing,
        )
        logger.info("Model %s v%s transitioned to %s", self.model_name, version, target_stage)
        return mv

    def promote_to_staging(self, version: str) -> ModelVersion:
        """Promote a model version to Staging."""
        return self.transition_stage(version, "Staging")

    def promote_to_production(self, version: str) -> ModelVersion:
        """Promote a model version to Production (archives current Production)."""
        return self.transition_stage(version, "Production", archive_existing=True)

    def archive_version(self, version: str) -> ModelVersion:
        """Archive a model version."""
        return self.transition_stage(version, "Archived")

    # ----- Model Comparison -----

    def compare_versions(self, version_a: str, version_b: str) -> Dict:
        """Compare metrics between two model versions."""
        details_a = self.get_version_details(version_a)
        details_b = self.get_version_details(version_b)

        comparison = {
            "version_a": version_a,
            "version_b": version_b,
            "stage_a": details_a["stage"],
            "stage_b": details_b["stage"],
            "metrics_comparison": {},
        }

        all_metrics = set(details_a["metrics"].keys()) | set(details_b["metrics"].keys())
        for metric in all_metrics:
            val_a = details_a["metrics"].get(metric)
            val_b = details_b["metrics"].get(metric)
            diff = None
            if val_a is not None and val_b is not None:
                diff = round(val_b - val_a, 6)
            comparison["metrics_comparison"][metric] = {
                f"v{version_a}": val_a,
                f"v{version_b}": val_b,
                "difference": diff,
            }

        return comparison

    def compare_staging_vs_production(self) -> Optional[Dict]:
        """Compare current Staging model against Production."""
        staging = self.client.get_latest_versions(self.model_name, stages=["Staging"])
        production = self.client.get_latest_versions(self.model_name, stages=["Production"])

        if not staging:
            logger.warning("No Staging model found.")
            return None
        if not production:
            logger.warning("No Production model found.")
            return None

        return self.compare_versions(production[0].version, staging[0].version)

    # ----- Validation Before Promotion -----

    def validate_model(
        self,
        version: str,
        validation_data: Optional[DataFrame] = None,
        min_auc: float = 0.70,
        max_auc_degradation: float = 0.02,
    ) -> Tuple[bool, List[ValidationResult]]:
        """
        Run automated validation checks before promoting a model.

        Checks:
        1. Model can be loaded successfully
        2. Model can generate predictions
        3. AUC meets minimum threshold
        4. AUC does not degrade vs. current Production model
        """
        results: List[ValidationResult] = []

        # Check 1: Model loading
        model_uri = f"models:/{self.model_name}/{version}"
        try:
            model = mlflow.spark.load_model(model_uri)
            results.append(ValidationResult(
                check_name="model_load",
                passed=True,
                message=f"Model loaded successfully from {model_uri}",
            ))
        except Exception as exc:
            results.append(ValidationResult(
                check_name="model_load",
                passed=False,
                message=f"Failed to load model: {exc}",
            ))
            return False, results

        # Check 2: Prediction generation
        if validation_data is not None:
            try:
                predictions = model.transform(validation_data)
                pred_count = predictions.count()
                results.append(ValidationResult(
                    check_name="prediction_generation",
                    passed=pred_count > 0,
                    message=f"Generated {pred_count} predictions",
                ))

                # Check 3: Minimum AUC
                evaluator = BinaryClassificationEvaluator(
                    labelCol="label",
                    rawPredictionCol="rawPrediction",
                    metricName="areaUnderROC",
                )
                auc = evaluator.evaluate(predictions)
                results.append(ValidationResult(
                    check_name="min_auc_threshold",
                    passed=auc >= min_auc,
                    message=f"AUC={auc:.4f} vs threshold={min_auc}",
                    metric_value=auc,
                    threshold=min_auc,
                ))

                # Check 4: AUC degradation vs Production
                prod_versions = self.client.get_latest_versions(self.model_name, stages=["Production"])
                if prod_versions:
                    prod_run = self.client.get_run(prod_versions[0].run_id)
                    prod_auc = prod_run.data.metrics.get("test_auc", 0)
                    degradation = prod_auc - auc
                    results.append(ValidationResult(
                        check_name="auc_degradation",
                        passed=degradation <= max_auc_degradation,
                        message=(
                            f"Degradation={degradation:.4f} vs max_allowed={max_auc_degradation}. "
                            f"Candidate AUC={auc:.4f}, Production AUC={prod_auc:.4f}"
                        ),
                        metric_value=degradation,
                        threshold=max_auc_degradation,
                    ))
                else:
                    results.append(ValidationResult(
                        check_name="auc_degradation",
                        passed=True,
                        message="No Production model to compare against. Skipping.",
                    ))

            except Exception as exc:
                results.append(ValidationResult(
                    check_name="prediction_generation",
                    passed=False,
                    message=f"Failed to generate predictions: {exc}",
                ))
        else:
            # Use logged metrics if no validation data
            details = self.get_version_details(version)
            logged_auc = details["metrics"].get("test_auc", 0)
            results.append(ValidationResult(
                check_name="min_auc_threshold",
                passed=logged_auc >= min_auc,
                message=f"Logged AUC={logged_auc:.4f} vs threshold={min_auc}",
                metric_value=logged_auc,
                threshold=min_auc,
            ))

        all_passed = all(r.passed for r in results)

        for r in results:
            status = "PASSED" if r.passed else "FAILED"
            logger.info("  [%s] %s: %s", status, r.check_name, r.message)

        return all_passed, results

    def validated_promotion(
        self,
        version: str,
        target_stage: str = "Production",
        validation_data: Optional[DataFrame] = None,
    ) -> bool:
        """Validate a model and promote if all checks pass."""
        logger.info("Validating model %s v%s for promotion to %s...",
                     self.model_name, version, target_stage)

        passed, results = self.validate_model(version, validation_data)

        if passed:
            self.transition_stage(version, target_stage)
            self.client.set_model_version_tag(
                self.model_name, version, "validated", "true"
            )
            self.client.set_model_version_tag(
                self.model_name, version, "promoted_at", datetime.now().isoformat()
            )
            logger.info("Model %s v%s promoted to %s.", self.model_name, version, target_stage)
        else:
            failed_checks = [r.check_name for r in results if not r.passed]
            logger.warning(
                "Promotion BLOCKED for %s v%s. Failed checks: %s",
                self.model_name, version, failed_checks,
            )

        return passed

    # ----- Webhook Notifications -----

    def send_webhook_notification(
        self,
        webhook_url: str,
        event: str,
        version: str,
        details: Optional[Dict] = None,
    ) -> bool:
        """Send a notification about a model registry event."""
        payload = {
            "text": f"Model Registry Event: {event}",
            "model_name": self.model_name,
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
        }

        try:
            response = requests.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            response.raise_for_status()
            logger.info("Webhook notification sent for event '%s'.", event)
            return True
        except requests.RequestException as exc:
            logger.error("Failed to send webhook: %s", exc)
            return False

    def setup_registry_webhooks(self, webhook_url: str) -> None:
        """Register webhooks for model registry events (Databricks-specific)."""
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.ml import RegistryWebhook, HttpUrlSpec

        w = WorkspaceClient()

        events = [
            "MODEL_VERSION_CREATED",
            "MODEL_VERSION_TRANSITIONED_STAGE",
            "MODEL_VERSION_TAG_SET",
        ]

        for event in events:
            try:
                w.model_registry.create_webhook(
                    events=[event],
                    model_name=self.model_name,
                    http_url_spec=HttpUrlSpec(
                        url=webhook_url,
                        enable_ssl_verification=True,
                    ),
                    description=f"Notify on {event} for {self.model_name}",
                )
                logger.info("Webhook registered for event: %s", event)
            except Exception as exc:
                logger.warning("Failed to register webhook for %s: %s", event, exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Model Registry operations")
    parser.add_argument("--model", default="purchase_prediction_model")
    parser.add_argument("--action", choices=["list", "compare", "promote", "validate", "archive"])
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--target-stage", default="Production")
    parser.add_argument("--webhook-url", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    spark = SparkSession.builder.appName("ModelRegistry").getOrCreate()

    manager = ModelRegistryManager(spark, args.model)

    if args.action == "list":
        versions = manager.list_versions()
        for v in versions:
            details = manager.get_version_details(v.version)
            print(f"v{v.version} [{v.current_stage}] — AUC: {details['metrics'].get('test_auc', 'N/A')}")

    elif args.action == "compare":
        comparison = manager.compare_staging_vs_production()
        if comparison:
            print(json.dumps(comparison, indent=2))

    elif args.action == "promote":
        if not args.version:
            staging = manager.client.get_latest_versions(args.model, stages=["Staging"])
            if not staging:
                logger.error("No Staging model to promote.")
                return
            args.version = staging[0].version

        success = manager.validated_promotion(args.version, args.target_stage)
        if success and args.webhook_url:
            manager.send_webhook_notification(
                args.webhook_url, "model_promoted",
                args.version, {"target_stage": args.target_stage},
            )

    elif args.action == "validate":
        if not args.version:
            logger.error("--version is required for validate action.")
            return
        passed, results = manager.validate_model(args.version)
        print(f"Validation {'PASSED' if passed else 'FAILED'}")
        for r in results:
            print(f"  {'PASS' if r.passed else 'FAIL'}: {r.check_name} — {r.message}")

    elif args.action == "archive":
        if not args.version:
            logger.error("--version is required for archive action.")
            return
        manager.archive_version(args.version)


if __name__ == "__main__":
    main()
