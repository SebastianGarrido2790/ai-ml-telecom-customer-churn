"""
Pipeline Stage 05: Late Fusion Model Training.

Conductor-only stage that delegates all business logic to the
LateFusionTrainer and LateFusionEvaluator components. This stage
has no algorithmic responsibility — it wires configuration to
components and triggers execution in the correct order.

Execution:
    uv run python -m src.pipeline.stage_05_model_training
"""

from src.components.model_training.evaluator import LateFusionEvaluator
from src.components.model_training.trainer import LateFusionTrainer
from src.config.configuration import ConfigurationManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

STAGE_NAME = "Stage 05: Late Fusion Model Training"


def main() -> None:
    """Executes the Late Fusion model training and evaluation pipeline.

    Workflow:
        1. Load ModelTrainingConfig from ConfigurationManager.
        2. Run LateFusionTrainer.train() → serializes three model artifacts.
        3. Run LateFusionEvaluator.evaluate() → logs MLflow runs and writes
           evaluation_report.json for DVC tracking.
    """
    logger.info(f"{'=' * 60}")
    logger.info(f">>>  {STAGE_NAME}  <<<")
    logger.info(f"{'=' * 60}")

    config_manager = ConfigurationManager()
    config = config_manager.get_model_training_config()

    logger.info("Initiating Late Fusion training.")
    trainer = LateFusionTrainer(config=config)
    trainer.train()

    logger.info("Initiating model evaluation and MLflow logging.")
    evaluator = LateFusionEvaluator(config=config)
    report = evaluator.evaluate()

    structured_recall = report["structured_baseline"]["metrics"]["recall"]
    fusion_recall = report["late_fusion_stacked"]["metrics"]["recall"]
    recall_lift = report["late_fusion_stacked"]["recall_lift"]
    f1_lift = report["late_fusion_stacked"]["f1_lift"]

    logger.info(f"{'=' * 60}")
    logger.info("TRAINING COMPLETE — Results Summary")
    logger.info(f"  Structured Baseline Recall : {structured_recall:.4f}")
    logger.info(f"  Late Fusion Recall         : {fusion_recall:.4f}")
    logger.info(f"  Recall Lift                : {recall_lift:+.4f}")
    logger.info(f"  F1 Lift                    : {f1_lift:+.4f}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
