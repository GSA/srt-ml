from __future__ import print_function

import argparse
import csv
from io import StringIO
import json
import os
import re
import shutil
import string
import subprocess as sb 
import sys
import time

#try:
    #import nltk  
#except ImportError:
    # pip install nltk without going the custom dockerfile route
    #sb.call([sys.executable, "-m", "pip", "install", "nltk"]) 
    #import nltk

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline

from sagemaker_containers.beta.framework import (content_types, 
                                                encoders, 
                                                env, 
                                                modules, 
                                                transformer, 
                                                worker)

#try:
    #nltk.data.find('wordnet')
#except LookupError:
    #nltk.download('wordnet')

#try:
    #nltk.data.find('punkt')
#except LookupError:
    #nltk.download('punkt')

#class LemmaTokenizer(object):

    #def __init__(self):
        #self.wnl = nltk.stem.WordNetLemmatizer()
    
    #def __call__(self, doc):
        #return [self.wnl.lemmatize(t) for t in nltk.word_tokenize(doc)]


class TextPreprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.control_regex = re.compile(r'[\s]|[a-z]|\b508\b|\b1973\b')
        self.token_pattern = re.compile(r'(?u)\b\w\w+\b')
        self.stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 
                          'there', 'about', 'once', 'during', 'out', 'very', 'having', 
                          'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                          'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off',
                          'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
                          'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 
                          'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more',
                          'himself', 'this', 'down', 'should', 'our', 'their', 'while',
                          'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no',
                          'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been',
                          'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that',
                          'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now',
                          'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too',
                          'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom',
                          't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing',
                          'it', 'how', 'further', 'was', 'here', 'than'}


    def fit(self, X, y = None):
        return self 
    

    def _preprocessing(self, doc):
        # split at any white space and rejoin using a single space. Then lowercase.
        doc_lowered = " ".join(doc.split()).lower()
        # map punctuation to space
        translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) 
        doc_lowered = doc_lowered.translate(translator)
        tokens = "".join(self.control_regex.findall(doc_lowered)).split()
        processed_text = []
        for token in tokens:
            if token in self.stopwords:
                continue
            m = self.token_pattern.search(token)
            if not m:
                continue
            word = m.group().strip()
            processed_text.append(word)
        
        processed_text = " ".join(processed_text)
        
        return processed_text
    
    
    def transform(self, X, y = None):
        X = X['text'].apply(self._preprocessing)
        
        return X


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
    
    df = pd.read_csv(input_file, 
                     header=None,
                     names=['target', 'text'],
                     dtype={'target': np.float64, 'text': str})               
    
    text_transformer = Pipeline(steps=[
        ('cleaner', TextPreprocessor()),
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
        df = pd.read_csv(StringIO(input_data))

        if len(df.columns) == 2:
            # This is a labelled example
            df.columns = ['target', 'text']
        elif len(df.columns) == 1:
            # This is an unlabelled example.
            df.columns = ['text']
        else:
            n_cols = len(df.columns)
            df_head = df.head(2)
            msg = "Input csv has too many columns of {}:  {}".format(n_cols, df_head)
            raise ValueError(msg)

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
    """Preprocess input data
    
    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:
    
        rest of features either one hot encoded or standardized
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
