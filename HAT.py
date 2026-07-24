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
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
import pickle

class HoeffdingAdaptiveTreeModel:
    def __init__(self, random_state, W=10_000):
        self.lppnse = HoeffdingAdaptiveTreeClassifier(
            grace_period=50,           
            split_confidence=1e-3,     
            tie_threshold=0.01,       
            leaf_prediction='nb',     
            nb_threshold=20,
            bootstrap_sampling=True,
        )

    def train(self, X_train, y_train):
        self.lppnse.fit(X_train, y_train, classes=[0,1])

    def retrain(self, X_train, y_train):
        self.lppnse.partial_fit(X_train, y_train)

    def predict_proba_fn(self, X):
        return self.lppnse.predict_proba(X)

    def predict_fn(self, X):
        y_pred = self.lppnse.predict(X)
        return y_pred
    

