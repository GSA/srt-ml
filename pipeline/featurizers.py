import re
import string
import subprocess as sb
import sys

from sklearn.base import BaseEstimator, TransformerMixin


try:
    import nltk  
except ImportError:
    # pip install nltk without going the custom dockerfile route
    sb.call([sys.executable, "-m", "pip", "install", "nltk"]) 
    import nltk

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')

from nltk.corpus import wordnet

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

class TextPreprocessor(BaseEstimator, TransformerMixin):
    
        def __init__(self):
            self.wnl = nltk.stem.WordNetLemmatizer()
            self.tag_map = {"J": wordnet.ADJ,
                            "N": wordnet.NOUN,
                            "V": wordnet.VERB,
                            "R": wordnet.ADV}
            self.control_regex = re.compile(r'[a-z]{2,}?|^508$|^1973$')
            self.token_pattern = re.compile(r'(?u)^\w\w+?$')
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
        
        def _keep_token(self, token):
            is_punc = token in string.punctuation
            if is_punc:
                return
            lowered = token.lower()
            is_stopword = lowered in self.stopwords
            if is_stopword:
                return
            is_not_match = True if not self.control_regex.search(lowered) else False
            if is_not_match:
                return
            is_not_token_pattern = True if not self.token_pattern.search(lowered) else False
            if is_not_token_pattern:
                return
            
            return True

        def _preprocessing(self, doc):
            pos_tagged_tokens = nltk.pos_tag(nltk.word_tokenize(doc))
            lemmas = []
            for token, pos_tag in pos_tagged_tokens:
                if not self._keep_token(token):
                    continue
                pos_tag = self.tag_map.get(pos_tag[0])
                if not pos_tag:
                    lemmas.append(self.wnl.lemmatize(token).lower())
                    continue
                lemmas.append(self.wnl.lemmatize(token, pos = pos_tag).lower())
             
            return " ".join(lemmas)
                
    
        def transform(self, X, y = None):
            X = X.apply(self._preprocessing)
            
            return X
