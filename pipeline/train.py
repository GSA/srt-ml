import numpy as np
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV


class log_uniform():        
    """
    Provides an instance of the log-uniform distribution with an .rvs() method. 
    Meant to be used with RandomizedSearchCV, particularly for params like alpha/C/gamma.
    
    Attributes:
        a (int or float): the exponent of the beginning of the range 
        b (int or float): the exponent of the end of range. 
        base (int or float): the base of the logarithm. 10 by default.
    """
    
    def __init__(self, a=-1, b=0, base=10):
        self.loc = a
        self.scale = b - a
        self.base = base

    def rvs(self, size=1, random_state=None):
        uniform = stats.uniform(loc=self.loc, scale=self.scale)
        return np.power(self.base, uniform.rvs(size=size, random_state=random_state))


def randomized_grid_search(df, pipeline, objective_metric_name='roc_auc', n_iter_search=10):
    """
    Performs a randomized grid search `n_iter_search` times using the pipeline provided and 
    the `objective_metric_name` as a scoring metric during refittig.
    
    Attributes:
        df (pandas DataFrame):  the training data
        pipeline (sklearn Pipeline object): a pipeline of transformers, ending in an estimator 
        objective_metric_name (str): the scorer used to evaluate the predictions on the test set. `roc_auc` by
                      default. Available options include:  accuracy, roc_auc, precision, fbeta, recall.
                      Note: for fbeta, beta is set to 1.5 to favor recall of the positive class.
    """
    scoring = {'accuracy': metrics.make_scorer(metrics.accuracy_score),
               'roc_auc': metrics.make_scorer(metrics.roc_auc_score),
               'precision': metrics.make_scorer(metrics.average_precision_score),
               'fbeta':metrics.make_scorer(metrics.fbeta_score,beta=.5),
               'recall':metrics.make_scorer(metrics.recall_score)}
    
    X = df['text']
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y,
                                                        stratify=y,
                                                        test_size=0.2)
    hyperparam_grid = {
                       "vectorizer__ngram_range": [(1,1), (1,2)],
                       "vectorizer__min_df": stats.randint(1,3),
                       "vectorizer__max_df": stats.uniform(.95,.3),
                       "vectorizer__sublinear_tf": [True, False],
                       "select__n_components": [10,100,200,500,1000,1500,2000,5000],
                       "select__n_iter": stats.randint(5,1000),
                       "estimator__alpha": log_uniform(-5,2),
                       "estimator__penalty": ['l2','l1','elasticnet'],
                       "estimator__loss": ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                       }
    
    random_search = RandomizedSearchCV(pipeline, 
                                       param_distributions = hyperparam_grid, 
                                       scoring = scoring, 
                                       refit = objective_metric_name,
                                       n_iter = n_iter_search, 
                                       cv = 5,
                                       n_jobs = -1, 
                                       verbose = 1)
    
    random_search.fit(X_train, y_train)
    best_estimator = random_search.best_estimator_
    
    return best_estimator
