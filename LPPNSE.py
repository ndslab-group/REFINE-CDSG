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
from skmultiflow.meta import LearnPPNSEClassifier
import pickle

class LearnPPModel:
    def __init__(self, random_state, n_estimators=5, window_size=2000, error_threshold=0.12, W=10_000):
        self.lppnse = LearnPPNSEClassifier(
            n_estimators=n_estimators,
            window_size=window_size,
        )

    def train(self, X_train, y_train):
        self.lppnse.fit(X_train, y_train, classes=[0,1])

    def retrain(self, X_train, y_train):
        self.lppnse.partial_fit(X_train, y_train, classes=[0,1])

    def predict_proba_fn(self, X):
        return self.lppnse.predict_proba(X)
    
    def predict_proba(self, X):
        return self.lppnse.predict_proba(X)

    def predict_fn(self, X):
        y_pred = self.lppnse.predict(X)
        return y_pred
    

