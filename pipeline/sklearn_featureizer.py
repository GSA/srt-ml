import argparse
import csv
from io import StringIO
import json
import os
import subprocess as sb
import sys
import warnings
warnings.filterwarnings('once')
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import pandas as pd
from sagemaker_containers.beta.framework import (content_types, 
                                                 encoders, 
                                                 env, 
                                                 modules, 
                                                 transformer, 
                                                 worker)
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

try:
    import nltk  
except ImportError:
    # pip install nltk without going the custom dockerfile route
    # Although featurizers.py uses nltk, put the pip installs here so that
    # the gride searching doesn't constantly make these calls
    sb.call([sys.executable, "-m", "pip", "install", "nltk"]) 
    import nltk

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet',quiet=True)

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt',quiet=True)

try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger',quiet=True)

from featurizers import TextPreprocessor
from train import randomized_grid_search


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) 
                    if file.endswith('.csv')]
    
    if len(input_files) == 0:
        raise ValueError(('There is no file in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    elif len(input_files) != 1:
        raise ValueError(('There is more than one file in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    
    input_file = input_files[0]
    
    df = pd.read_csv(input_file)
    df.columns = ['target', 'text']
    df = df.astype({'target': np.float64, 'text': str})             

    pipeline= Pipeline(steps=[
               ('preprocessor', TextPreprocessor()),
               ('vectorizer', TfidfVectorizer(analyzer=str.split,
                                              lowercase=False)), #tokens are already lowercase
               ('select', SelectKBest(chi2)),
               ('estimator', SGDClassifier(class_weight="balanced",
                                           max_iter=1000,
                                           tol=1e-3))])
    
    print("Fitting model...")
    model = randomized_grid_search(
        df, 
        pipeline,
        n_iter_search=30)
    print("Done fitting model!")
    
    print("Saving model...")
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    print("Done saving the model!")
    
    
def input_fn(input_data, content_type):
    """Parse input data payload
    
    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == 'text/csv':
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), header=None)
        
        if len(df.columns) == 2:
            # This is a labelled example, which includes the target
            df.columns = ['target', 'text']
            df = df.astype({'target': np.float64, 'text': str})
        elif len(df.columns) == 1:
            # This is an unlabelled example.
            df.columns = ['text']
            df = df.astype({'text': str})
        else:
            raise ValueError("Invalid payload. Payload must contain either two columns \
                (target, text) or one column (text)")

        return df
    
    else:
        raise ValueError("{} not supported by script!".format(content_type))
        

def output_fn(inferences, accept):
    """Format inferences output
    
    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for inference in inferences.tolist():
            try:
                target, decision_boundary, prediction = inference
                instances.append({"decision boundary": decision_boundary,
                                  "prediction": prediction,
                                  "target": target})
            except ValueError:
                decision_boundary, prediction = inference
                instances.append({"decision boundary": decision_boundary,
                                  "prediction": prediction})
        json_output = {"instances": instances}
        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeError("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
    """Call predict on the estimator given input data.
    """
    input_data = input_data['text']
    
    y_preds = model.predict(input_data)
    
    #get the index of the positive class (i.e. 1, compliant)
    positive_class_idx = list(model.classes_).index(1)
    try:
        y_scores = model.predict_proba(input_data)[:,positive_class_idx]
    except AttributeError:
        y_scores = model.decision_function(input_data)
    inferences = np.column_stack((y_scores, y_preds))
        
    if 'target' in input_data:
        # Return the label (as the first column) alongside the inferences
        return np.insert(inferences, 0, input_data['target'], axis=1)
    else:
        return inferences
    

def model_fn(model_dir):
    """Deserialize fitted model
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    
    return model

