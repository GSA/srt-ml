import re
import string

from sklearn.base import BaseEstimator, TransformerMixin

#try:
    #import nltk  
#except ImportError:
    # pip install nltk without going the custom dockerfile route
    #sb.call([sys.executable, "-m", "pip", "install", "nltk"]) 
    #import nltk

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
