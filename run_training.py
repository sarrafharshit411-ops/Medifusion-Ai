"""
MediFusion AI - Master Training Orchestrator
Runs all training pipelines in sequence:
1. Image model (ResNet18)
2. Symptom model (XGBoost)
3. Clinical model (XGBoost)
4. Fusion models (ConcatMLP + GatedAttention)
5. Evaluation & report generation
"""

import os
import sys
import time
import logging

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.utils.helpers import load_config, set_seed, setup_logging, ensure_dir

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    config = load_config(os.path.join(project_root, "configs", "config.yaml"))
    set_seed(config["training"]["seed"])
    
    # Ensure output dirs exist
    ensure_dir(os.path.join(project_root, config["paths"]["models_dir"]))
    ensure_dir(os.path.join(project_root, config["paths"]["reports_dir"]))
    
    total_start = time.time()
    
    # --- Step 1: Train Image Model ---
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1/6: Training Image Model (ResNet18 - Chest X-Ray)")
    logger.info("=" * 60)
    try:
        from src.training.train_image import train_image_model
        image_results = train_image_model(config)
        logger.info(f"[OK] Image model trained. Test accuracy: {image_results['test_accuracy']:.4f}")
    except Exception as e:
        logger.error(f"[FAIL] Image model training failed: {e}", exc_info=True)
    
    # --- Step 1.5: Train Advanced Image Models ---
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1.5/6: Training Advanced Image Models (Brain Tumor, Skin Cancer, Retinopathy, Blood Cell)")
    logger.info("=" * 60)
    try:
        from src.training.train_multi_image import train_all_image_models
        adv_image_results = train_all_image_models(config)
        for key, res in adv_image_results.items():
            if "error" not in res:
                logger.info(f"[OK] {key} model: test acc = {res['test_accuracy']:.4f}")
            else:
                logger.warning(f"[SKIP] {key} model: {res['error']}")
    except Exception as e:
        logger.error(f"[FAIL] Advanced image model training failed: {e}", exc_info=True)
    
    # --- Step 2: Train Symptom Model ---
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2/5: Training Symptom Model (XGBoost)")
    logger.info("=" * 60)
    try:
        from src.training.train_symptoms import train_symptom_model
        symptom_results = train_symptom_model(config)
        logger.info(f"[OK] Symptom model trained. Test accuracy: {symptom_results['test_accuracy']:.4f}")
    except Exception as e:
        logger.error(f"[FAIL] Symptom model training failed: {e}", exc_info=True)
    
    # --- Step 3: Train Clinical Model ---
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3/5: Training Clinical Model (XGBoost)")
    logger.info("=" * 60)
    try:
        from src.training.train_clinical import train_clinical_model
        clinical_results = train_clinical_model(config)
        logger.info(f"[OK] Clinical model trained. Test accuracy: {clinical_results['test_accuracy']:.4f}")
    except Exception as e:
        logger.error(f"[FAIL] Clinical model training failed: {e}", exc_info=True)
    
    # --- Step 4: Train Fusion Models ---
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4/5: Training Fusion Models")
    logger.info("=" * 60)
    try:
        from src.training.train_fusion import train_fusion_model
        fusion_results = train_fusion_model(config)
        for name, res in fusion_results.items():
            logger.info(f"[OK] {name} fusion: val acc = {res['best_val_accuracy']:.4f}")
    except Exception as e:
        logger.error(f"[FAIL] Fusion model training failed: {e}", exc_info=True)
    
    # --- Step 5: Evaluation ---
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5/5: Running Evaluation")
    logger.info("=" * 60)
    try:
        from src.training.evaluate import run_evaluation
        all_metrics = run_evaluation(config)
        logger.info("[OK] Evaluation complete. Reports saved to reports/")
    except Exception as e:
        logger.error(f"[FAIL] Evaluation failed: {e}", exc_info=True)
    
    total_time = time.time() - total_start
    logger.info(f"\n{'=' * 60}")
    logger.info(f"ALL TRAINING COMPLETE in {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
