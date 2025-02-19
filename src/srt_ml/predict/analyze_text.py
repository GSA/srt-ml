#!/usr/bin/env python3
import sys
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
    # Read the entire input from STDIN
    input_data = sys.stdin.read()
    try:
        data = json.loads(input_data)
    except Exception as e:
        logger.error("Failed to parse input JSON: " + str(e))
        sys.exit(1)

    # Expecting a JSON object with a "documents" property mapping filenames to texts
    documents = data.get("documents", {})
    if not documents:
        logger.error("No documents provided in input.")
        sys.exit(1)

    # Determine the model path relative to this file's parent directory
    model_name = "clf_ajbuckingham_roc_auc.pkl"
    current_dir = Path(__file__).parent
    model_path = current_dir.parent / 'binaries' / model_name

    # Instantiate Predict and process each text input
    predictor = Predict(best_model_path=model_path)
    results = {}

    for fname, text in documents.items():
        prediction = predictor.analyze_text(text)
        results[fname] = prediction

    # Output the prediction results as JSON to STDOUT
    print(json.dumps({"predictions": results}))

if __name__ == "__main__":
    main()
