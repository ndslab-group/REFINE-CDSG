from skmultiflow.meta import AdaptiveRandomForestClassifier

class ARFAModel:
    def __init__(self, random_state):
          self.arff = AdaptiveRandomForestClassifier()
    
    def train(self, X_train, y_train):
        self.arff.fit(X_train, y_train)

    def retrain(self, X_train, y_train):
        self.arff.partial_fit(X_train, y_train) 

    def predict_proba_fn(self, X):
        return self.arff.predict_proba(X)
    
    def predict_proba(self, X):
        return self.arff.predict_proba(X)

    def predict_fn(self, X):
        y_pred = self.arff.predict(X)
        return y_pred