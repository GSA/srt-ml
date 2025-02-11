#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path
from srt_ml.predict.predict import Predict

# Use a logs directory within the user's home folder instead of /opt/ml/logs
logs_dir = Path.home() / "srt_ml_logs"
logs_dir.mkdir(parents=True, exist_ok=True)

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(str(logs_dir / "worker.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Analyze raw text using Predict.analyze_text"
    )
    parser.add_argument(
        "--text",
        required=True,
        help="The raw text to analyze"
    )
    parser.add_argument(
        "--model",
        default="clf_ajbuckingham_roc_auc.pkl",
        help="Model file name (relative to the binaries folder)"
    )
    args = parser.parse_args()

    # Determine the model path relative to this file's parent directory
    current_dir = Path(__file__).parent
    model_path = current_dir.parent / 'binaries' / args.model

    # Instantiate Predict and call its analyze_text method
    predictor = Predict(best_model_path=model_path)
    prediction = predictor.analyze_text(args.text)

    # Output the prediction result as JSON
    print(json.dumps({"prediction": prediction}))

if __name__ == "__main__":
    main()
