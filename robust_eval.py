from robustevaluationpipeline import RobustEvaluationPipelineOffline
from robustevaluationpipeline_online import RobustEvaluationPipelineOnline
from evaluate_KSWIN import ElaborateKSWIN
import os
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from D3 import D3
from tqdm import tqdm 
import csv
import numpy as np
import json

class RobustEval: 
    def __init__(self, dataset_name, streamed_filename, clustering_method, drift_type, bias_spatial, W, model_name, slope=None):
        self.dataset_name = dataset_name
        self.streamed_filename = streamed_filename
        self.clustering_method = clustering_method
        self.drift_type = drift_type
        self.bias_spatial = bias_spatial
        self.W = W
        if slope:
            self.output_folder = f'online_evaluation_{self.dataset_name}_{self.clustering_method}_{self.drift_type}_bias{self.bias_spatial}_W{self.W}_slope{slope}'
        else:
            self.output_folder = f'online_evaluation_{self.dataset_name}_{self.clustering_method}_{self.drift_type}_bias{self.bias_spatial}_W{self.W}'
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.model_name = model_name
        self.offline_pipeline = RobustEvaluationPipelineOffline(dataset_name, streamed_filename, self.output_folder, clustering_method, drift_type, bias_spatial, W, model_name)
        self.online_pipeline = RobustEvaluationPipelineOnline(dataset_name, streamed_filename, self.output_folder, clustering_method, drift_type, bias_spatial, W, model_name)
        
    def run_offline_evaluation(self, split_indedx=None, skip=False):
        self.offline_pipeline.pipeline_offline(split_indedx=split_indedx, skip=skip)
    
    def run_train_models(self):
       self.offline_pipeline.training_loop(model_name=self.model_name)

    def run_online_evaluation(self, _only_evaluate=True):
        self.online_pipeline.run_online_evaluation(_only_evaluate)

    def run_kswin(self):
        kscdd = ElaborateKSWIN(W=self.W, output_folder=self.output_folder, bias_spatial=self.bias_spatial)
        kscdd.computeDrift(model_used=self.model_name)

    def run_d3_on_batches(self, w=10_000, rho=0.2, auc=0.80):
        def check_true(y, y_hat):
            return int(y == y_hat)
        X_train, y_train, _, _, _ = self.offline_pipeline.retrieve_windows_offline()
        n_features = X_train.shape[1]

        stream_clf = HoeffdingTree()
        D3_win = D3(w, rho, n_features, auc)

        initial_samples = int(w * rho)
        stream_clf.partial_fit(X_train[:initial_samples], y_train[:initial_samples])
        for i in range(initial_samples):
            D3_win.addInstance(X_train[i, :], y_train[i])

        drift_windows = []
        stream_acc = []
        stream_record = []
        stream_true = 0
        total_samples = initial_samples

        windows = self.online_pipeline.extract_data_online()

        for idx, window in enumerate(tqdm(windows, desc="Processing batches with D3")):
            X_batch, y_batch = window
            y_pred_batch = stream_clf.predict(X_batch)
            correct = sum([check_true(y_batch[i], y_pred_batch[i]) for i in range(len(y_batch))])
            stream_true += correct
            total_samples += len(y_batch)
            stream_acc.append(stream_true / total_samples)
            stream_record.extend([check_true(y_batch[i], y_pred_batch[i]) for i in range(len(y_batch))])

            drift_flag = False

            
            for i in range(X_batch.shape[0]):
                x_sample = X_batch[i, :].reshape(1, -1)  
                y_sample = np.array([y_batch[i]])        

                if not D3_win.isEmpty() and D3_win.driftCheck():
                    drift_flag = True

                    # Reset modello e retrain su finestra corrente
                    X_curr = D3_win.getCurrentData()
                    y_curr = D3_win.getCurrentLabels()
                    stream_clf = HoeffdingTree()
                    stream_clf.partial_fit(X_curr, y_curr)

                D3_win.addInstance(X_batch[i, :], y_batch[i])
                stream_clf.partial_fit(x_sample, y_sample)

            if drift_flag:
                drift_windows.append([idx,1])
                print("Drift detected!!!!")
            else: 
                drift_windows.append([idx,0])

        drift_path = f"{self.output_folder}_D3_driftdetector.csv"
        with open(drift_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["drift_batch_index"])
            for dw in drift_windows:
                writer.writerow([dw])

        print(f"Saved D3 drift windows to {drift_path}")
        return stream_acc, drift_windows




    

models = ['rf', 'xgb', 'arf', 'arfa', 'axgb', 'hat', 'lppnse']
for model in models: 

    re = RobustEval(
        dataset_name='cicids', ## cicids, bccc_cpacket, xiiot, etc etc
        streamed_filename='streaming_recurrent_0.7_refine.csv', ## streamed dataset to use for evaluation, is stored in results_cdsg_directory_{dataset_name}/{clustering_method}
        clustering_method='kmeans',
        drift_type='recurrent', # incremental, gradual, sudden, recurrent
        bias_spatial=0.7,
        W=20_000,
        model_name=model
    )

    ###### splitting indexes for offline and online
    re.run_offline_evaluation(split_indedx=280_000, skip=False) 
    
    ###### training on offline dataset
    re.run_train_models()

    ##### online evaluation on windows
    re.run_online_evaluation(_only_evaluate=False) ## if True, no predictions are extracted

    ### kswin and D3 from CDD module
    re.run_kswin()
    re.run_d3_on_batches(w=20_000)