import pandas as pd 
import numpy as np 
from tqdm import tqdm
import os
import pickle
import matplotlib.pyplot as plt

from KSWIN import KSWIN_cdd


class ElaborateKSWIN:
    def __init__(self, W, bias_spatial, output_folder):
        self.W = W
        self.bias_spatial = bias_spatial
        self.kswin = KSWIN_cdd(W=50)
        self.output_folder = output_folder
    
    def check_drift(self, y_true, y_pred):
        '''
        detects drift using KSWIN, but with the predictions of a base estimator,
        in this case directly taken from the case study of a base classifier
        such as the random forest
        '''
        return self.kswin.detected_change_error(y_window=y_true, y_pred_window=y_pred)
    
    def computeDrift(self, model_used):
        '''
        this function loads the predictions made by a base model (RF)
        and calculates the drift appropriately.
        - Folder path is where the models are saved
        - model_used will be RF
        - the model to load is the one trained during training
        '''
        # Determine the base path and load the model
        output_dir = f'{self.output_folder}/results/{model_used}'
        model_filename = f"{model_used}_results_{self.W}_{self.bias_spatial}.pkl"
        model_path = os.path.join(output_dir, model_filename)
        with open(model_path, "rb") as f:
            results = pickle.load(f)
        
        # lists of true and predicted values
        y_true_list = [r[0] for r in results] 
        y_pred_list = [r[1] for r in results]

        # now detect drift by considering the true and predicted values
        drift_detected = []
        for i in tqdm(range(len(y_true_list)), desc=f"Loading predictions for model {model_used}"):
            y_true = y_true_list[i]
            y_pred = y_pred_list[i]
            drift_ = self.check_drift(y_pred=y_pred, y_true=y_true)
            drift_detected.append(1 if drift_ else 0)
        # return drift_detected
        print("Saving drift -- KSWIN")
        self.saveDrift(drift_detected, model_used)
    
    def saveDrift(self, array_to_save, model_used):
        array_np = np.array(array_to_save)
        #### create the path
        output_dir = f'{self.output_folder}/results/kswin'
        os.makedirs(output_dir, exist_ok=True)
        model_filename = f"kswin_{model_used}_results_{self.W}_{self.bias_spatial}.npy"
        npy_array_path = os.path.join(output_dir, model_filename)
        np.save(npy_array_path, array_np)

        print(f"Saving completed at path {npy_array_path}")
        # check that everything went well
        array_loaded = np.load(npy_array_path)
       
        ######### plot
        plt.figure(figsize=(12, 4))
        x = np.arange(len(array_loaded))
        plt.plot(x, array_loaded, '-o', markersize=3)
        drift_idx = np.where(array_loaded == 1)[0]
        plt.scatter(drift_idx, np.ones_like(drift_idx), s=50, label='Drift detected')
        plt.xlabel("Window")
        plt.ylabel("Drift")
        plt.yticks([0, 1], ["No Drift", "Drift"])
        plt.title(f"KSWIN Drift Detection - {model_used}")
        plt.grid(True)
        plt.legend()
        plot_path = os.path.join(output_dir, f"kswin_{model_used}_{self.W}_{self.bias_spatial}.pdf")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.show()
        print(f"Plot saved to {plot_path}")