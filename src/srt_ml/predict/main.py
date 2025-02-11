import argparse
import logging
from srt_ml.predict.predict import Predict
import json


def predict_parser():
    parser = argparse.ArgumentParser(description="Process multiple files using the Predict class.")
    
    parser.add_argument(
        "-m",
        "--model",
        default="clf_ajbuckingham_roc_auc.pkl",
        help="Model name to use for prediction",
    )
    
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        required=True,
        help="List of files to process. (Full Path)",
    )
    return parser

def main():
    parser = predict_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        predict = Predict(best_model_path=args.model)
        file_predictions = predict.process_multiple_files(args.files)

        print(json.dumps(file_predictions))
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        raise

if __name__ == "__main__":
    main()