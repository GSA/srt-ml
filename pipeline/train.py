import warnings
warnings.filterwarnings('once')
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV


class log_uniform():
    """Provides an instance of a log-uniform distribution with an rvs method for use in
    hyperparameter grid. The low param is intuitively the number of decimal places to which
    the left end of the distrubution will go (e.g. lowe=-5 means the min value might be 1.1e-05).
    High is the number of decimals to which the right end of the distrubution will go (e.g.
    high=3 means the max might be 999.2). The resultant log-uniform distribution will be skewed
    right, with the majority of values being < 1.

    Note that the rvs method must have the size and random_state kwargs to be sklearn compatible
    """
    def __init__(self, low=-5, high=3, base=10):
        self.low = low
        self.high = high
        self.base = base

    def rvs(self, size=1, random_state=None):
        return np.power(self.base, np.random.uniform(self.low, self.high))


def randomized_grid_search(df, pipeline, objective_metric_name='roc_auc', n_iter_search=1):
    """Randomized grid search of a pipeline of transformers using objective_metric_name as the scorer.
    
    Arguments:
        df {pandas.DataFrame} -- a data frame with at least a 'text' and 'target' column
        pipeline {sklearn.Pipeline} -- a pipeline of sklearn transformers accepting text as the input
    
    Keyword Arguments:
        objective_metric_name {str} -- the scorer to use for refitting (default: {'roc_auc'})
        n_iter_search {int} -- how many grid search iterations (default: {1})
    
    Returns:
        [type] -- [description]
    """
    scoring = {'accuracy': metrics.make_scorer(metrics.accuracy_score),
               'roc_auc': metrics.make_scorer(metrics.roc_auc_score),
               'precision': metrics.make_scorer(metrics.average_precision_score),
               'recall':metrics.make_scorer(metrics.recall_score)}
    
    X = df['text']
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y,
                                                        stratify=y,
                                                        test_size=0.2)
    hyperparam_grid = {
                       "vectorizer__ngram_range": [(1,1), (1,2)],
                       "vectorizer__max_df": stats.uniform(.8,.95),
                       "vectorizer__sublinear_tf": [True, False],
                       "select__k": stats.randint(100,200),
                       "estimator__alpha": log_uniform(-5,2),
                       "estimator__penalty": ['l2','l1','elasticnet'],                       
                       "estimator__loss": ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']
                       }
    
    random_search = RandomizedSearchCV(pipeline, 
                                       param_distributions=hyperparam_grid, 
                                       scoring=scoring, 
                                       refit=objective_metric_name,
                                       n_iter=n_iter_search, 
                                       cv=5,
                                       n_jobs=-1, 
                                       verbose=1,
                                       random_state=123)
    
    random_search.fit(X_train, y_train)
    best_estimator = random_search.best_estimator_
    
    return best_estimator
