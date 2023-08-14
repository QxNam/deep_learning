from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin

class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = None
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def f1_score(self, y_true, y_pred):
        return metrics.f1_score(y_true, y_pred, average='macro')
    
    def accuracy_score(self, y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)
    
    def confusion_matrix(self, y_true, y_pred):
        return metrics.confusion_matrix(y_true, y_pred)