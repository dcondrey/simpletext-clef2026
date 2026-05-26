"""
SimpleText Task 2 v5: Main entry point.

Runs the full pipeline: train → predict → generate submissions.

Usage:
    uv run python main.py                    # full pipeline (embed + NLI)
    uv run python main.py --no-nli           # skip NLI (faster)
    uv run python main.py --predict-only     # skip training, just predict
    uv run python main.py --no-nli --no-embed  # CPU-only, no neural features
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="SimpleText Task 2 v5 pipeline")
    parser.add_argument("--no-nli", action="store_true", help="Skip NLI cross-encoder")
    parser.add_argument("--no-embed", action="store_true", help="Skip embedding similarity")
    parser.add_argument("--predict-only", action="store_true", help="Skip training, run inference only")
    args = parser.parse_args()

    use_embed = not args.no_embed
    use_nli = not args.no_nli

    if not args.predict_only:
        log.info("=" * 60)
        log.info("  PHASE 1: TRAINING")
        log.info("=" * 60)
        from train import run_training
        artifacts = run_training(use_embed=use_embed, use_nli=use_nli)
        log.info("\nTraining complete. Artifacts saved to models_v5/")

    log.info("\n" + "=" * 60)
    log.info("  PHASE 2: INFERENCE + SUBMISSIONS")
    log.info("=" * 60)
    from predict import run_inference
    run_inference(use_embed=use_embed, use_nli=use_nli)

    log.info("\n" + "=" * 60)
    log.info("  DONE")
    log.info("=" * 60)
    log.info("Submission files are in submissions_v5/")
    log.info("To submit to Codabench, upload the .zip files:")
    log.info("  Task 2.1: submissions_v5/writerslogic_Task21b_v4_strong.zip")
    log.info("  Task 2.2: submissions_v5/writerslogic_Task22b_v4_strong.zip")
    log.info("  Task 2.1 (fast): submissions_v5/writerslogic_Task21b_v4_fast.zip")
    log.info("  Task 2.2 (fast): submissions_v5/writerslogic_Task22b_v4_fast.zip")


if __name__ == "__main__":
    main()
