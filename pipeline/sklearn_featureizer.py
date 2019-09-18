from __future__ import print_function

import argparse
import csv
from io import StringIO
import json
import os
import re
import shutil
import subprocess as sb 
import sys
import time

import numpy as np
import pandas as pd
from sagemaker_containers.beta.framework import (content_types, 
                                                 encoders, 
                                                 env, 
                                                 modules, 
                                                 transformer, 
                                                 worker)
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

from featurizers import TextPreprocessor


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
    
    text_transformer = Pipeline(steps=[
        ('preprocessor', TextPreprocessor()),
        ('vectorizer', TfidfVectorizer(analyzer=str.split,
                                       ngram_range=(1,2),
                                       sublinear_tf=True)),
        ('select', TruncatedSVD(n_components=100, n_iter=2))])

    preprocessor = ColumnTransformer(transformers=[('txt', text_transformer, ['text'])])
    
    print("Fitting preprocessor...")
    preprocessor.fit(df)
    print("Done fitting preprocessor!")
    
    print("Saving model...")
    joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))
    print("Done saving the model!")
    
    
def input_fn(input_data, content_type):
    """Parse input data payload
    
    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == 'text/csv':
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), header = None)
        
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
        

def output_fn(prediction, accept):
    """Format prediction output
    
    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeError("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
    """Preprocess input data, which is a pandas dataframe
    
    We implement this because the default predict_fn uses .predict(), but our model is a 
    preprocessor so we want to use .transform().
    """

    features = model.transform(input_data)
    
    if 'target' in input_data:
        # Return the label (as the first column) and the set of features.
        return np.insert(features, 0, input_data['target'], axis=1)
    else:
        # Return only the set of features
        return features
    

def model_fn(model_dir):
    """Deserialize fitted model
    """
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    
    return preprocessor
