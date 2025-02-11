#!/usr/bin/env python3
import sys
import json
import logging
from pathlib import Path
import dill as pickle  # Add this import

# Get the absolute path to the model
current_dir = Path(__file__).parent
model_path = Path('/opt/ml/src/srt_ml/binaries/clf_ajbuckingham_roc_auc.pkl')  # Update model path

sys.path.append(str(current_dir.parent.parent))  # Add project root to Python path
from srt_ml.predict.predict import Predict

# Ensure the logs directory exists
logs_dir = Path('/opt/ml/logs')
logs_dir.mkdir(parents=True, exist_ok=True)

# Setup logging
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("/opt/ml/logs/worker.log"),  # Logs to a file
            logging.StreamHandler(sys.stderr)  # Logs to stderr
        ]
    )
except Exception as e:
    # Fallback to stderr only if the file handler fails
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stderr)
        ]
    )
    print(f"Warning: Unable to log to file. Logging to stderr only. Error: {e}", file=sys.stderr)

logger = logging.getLogger(__name__)

logger.info(f"Current directory: {current_dir}")
logger.info(f"Model path: {model_path}")
logger.info(f"Model exists: {model_path.exists()}")

def analyze_text(text: str):
    """
    Analyze text using the existing Predict class
    """
    try:
        # Create mock data structure to match what Predict expects
        mock_data = [{
            'attachments': [{
                'text': text
            }],
            'solnbr': 'direct-input',
            'agency': 'direct-input'
        }]
        
        # Initialize predictor with specific model path
        predictor = Predict(
            best_model_path=model_path,
            data=mock_data
        )
        
        # Make prediction
        result = predictor.insert_predictions()
        
        # Extract the prediction details from the first document's first attachment
        prediction_result = result[0]['attachments'][0]
        
        return {
            'prediction': prediction_result['prediction'],
            'decision_boundary': prediction_result['decision_boundary'],
            'text': text[:1000]  # Return first 1000 chars of original text
        }
        
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        return {
            'error': str(e)
        }

def main():
    try:
        # Read input text from stdin
        text = sys.stdin.read()
        
        if not text:
            result = {'error': 'No input text provided'}
        else:
            result = analyze_text(text)
        
        # Output clean JSON result to stdout
        print(json.dumps(result))
        
    except Exception as e:
        # Log the error (stderr) and output error in JSON (stdout)
        logger.error(f"Error in main: {str(e)}")
        print(json.dumps({'error': str(e)}))
        sys.exit(1)

if __name__ == '__main__':
    main()
