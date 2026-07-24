from datetime import date
import os
from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report
import joblib
from skmultiflow.drift_detection import ADWIN
from skmultiflow.meta import AdaptiveRandomForestClassifier
import pickle

class ARFModel:
    def __init__(self, random_state):
        self.arff_noadpt = AdaptiveRandomForestClassifier(n_estimators=30, drift_detection_method=None, warning_detection_method=None, random_state=17)

    def train(self, X_train, y_train):
        self.arff_noadpt.fit(X_train, y_train)

    def retrain(self, X_train, y_train):
        self.arff_noadpt.partial_fit(X_train, y_train)

    def predict_proba_fn(self, X):
        return self.arff_noadpt.predict_proba(X)

    def predict_fn(self, X):
        y_pred = self.arff_noadpt.predict(X)
        return y_pred