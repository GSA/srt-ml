#!/usr/bin/env python3
import argparse
import logging
import json
from srt_ml.predict.predict import Predict

def predict_parser():
    """
    Sets up and returns an argument parser for processing multiple files.
    """
    parser = argparse.ArgumentParser(
        description="Process multiple files using the Predict class."
    )
    parser.add_argument(
        "-m",
        "--model",
        default="clf_ajbuckingham_roc_auc.pkl",
        help="Model file name to use for prediction (relative to the binaries folder)",
    )
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        required=True,
        help="List of files to process. (Provide full paths)",
    )
    return parser

def main():
    """
    Main entry point for the CLI tool.
    Parses command-line arguments, instantiates the Predict class, and processes the provided files.
    """
    parser = predict_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Instantiate the Predict class using the specified model file.
        predictor = Predict(best_model_path=args.model)
        # Process multiple files and obtain predictions.
        file_predictions = predictor.process_multiple_files(args.files)
        # Output the predictions as JSON.
        print(json.dumps(file_predictions))
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        raise

if __name__ == "__main__":
    main()
