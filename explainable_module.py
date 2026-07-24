import numpy as np 
import pandas as pd 
import os 
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from robustevaluationpipeline import RobustEvaluationPipelineOffline
from robustevaluationpipeline_online import RobustEvaluationPipelineOnline

from train_ARFA import ARFATrainer
from train_RF import RFTrainer
from train_LPPNSE import LPPNSETrainer 
from train_XGB import XGBTrainer

import shap
from multiprocessing import Pool, cpu_count


def _compute_shap_batch_worker_single(args):
    model, background, batch, max_evals = args
    explainer = shap.PermutationExplainer(model.predict_proba, background)
    # print("BACKGROUND SHAPE:", background.shape)
    # print("BATCH SHAPE:", batch.shape)
    return explainer(batch, max_evals=max_evals)

class ExtractXAIOnline:
    def __init__(self, dataset_name, streamed_filename, clustering_method, drift_type, bias_spatial, W, model_name, slope=None):
        self.dataset_name = dataset_name
        self.streamed_filename = streamed_filename
        self.clustering_method = clustering_method
        self.drift_type = drift_type
        self.bias_spatial = bias_spatial
        self.W = W
        if slope:
            self.online_folder = f'online_evaluation_{self.dataset_name}_{self.clustering_method}_{self.drift_type}_bias{self.bias_spatial}_W{self.W}_slope{slope}'
        else:
            self.online_folder = f'online_evaluation_{self.dataset_name}_{self.clustering_method}_{self.drift_type}_bias{self.bias_spatial}_W{self.W}'
        self.model_name = model_name
        self.offline_pipeline = RobustEvaluationPipelineOffline(dataset_name, streamed_filename, self.online_folder, clustering_method, drift_type, bias_spatial, W, model_name)
        self.online_pipeline = RobustEvaluationPipelineOnline(dataset_name, streamed_filename, self.online_folder, clustering_method, drift_type, bias_spatial, W, model_name)
        if slope:
            self.output_folder = f'XAI_online_evaluation_{self.dataset_name}_{self.clustering_method}_{self.drift_type}_bias{self.bias_spatial}_W{self.W}_slope{slope}'
        else:
            self.output_folder = f'XAI_online_evaluation_{self.dataset_name}_{self.clustering_method}_{self.drift_type}_bias{self.bias_spatial}_W{self.W}'
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.colonne = None
        
    def extract_data_offline(self):
        X_train, y_train, X_val, y_val, self.colonne = self.offline_pipeline.retrieve_windows_offline()
        return X_train, y_train, X_val, y_val
    
    def extract_data_online(self):
        windows  = self.online_pipeline.extract_data_online()
        return windows

    def test_model(self, model, X_test):
        y_pred = model.predict(X_test)
        return y_pred
    
    def testing_loop(self):
        windows = self.extract_data_online() 
        windows_indexs = ['train'] + list(range(len(windows)))

        X_train, _, _, _ = self.extract_data_offline()

        self.colonne = [ col for col in self.colonne if col != 'label' ]

        print(f"Colonne: {self.colonne}")

        background_size = min(20, X_train.shape[0])
        background_indices = np.random.choice(X_train.shape[0], background_size, replace=False)
        background_t = X_train[background_indices]

        if self.model_name == 'rf':
            model = RFTrainer(data_dir=self.online_folder, W=self.W)
        elif self.model_name == 'arfa':
            model = ARFATrainer(data_dir=self.online_folder, W=self.W)
        elif self.model_name == 'xgb':
            model = XGBTrainer(data_dir=self.online_folder, W=self.W)
        elif self.model_name == 'lppnse':
            model = LPPNSETrainer(data_dir=self.online_folder, W=self.W)
        
        for pos, idx in enumerate(tqdm(windows_indexs, desc='Estrazione SHAP')):
            current_model = model.load_model(w_index=idx, folder_path=self.online_folder)
            current_batch = X_train if idx == 'train' else windows[idx][0]

            # post (t)
            explanation_post = self.xai_batched_parallel(data_values=current_batch, model=current_model, background_data=background_t) 
            print(f"Salvataggio explanations POST per batch {idx}")
            self.save_explainer(explainer=explanation_post, filename=f'shap_post_{idx}', w_index=idx)

            # if not last batch, take the next one (t+1)
            if pos != len(windows_indexs) - 1:
                next_idx = windows_indexs[pos + 1]
                next_batch = X_train if next_idx == 'train' else windows[next_idx][0]
                explanation_pre = self.xai_batched_parallel(data_values=next_batch, model=current_model, background_data=background_t) 
                print(f"Salvataggio explanations PRE per batch {idx}")
                self.save_explainer(explainer=explanation_pre, filename=f'shap_pre_{next_idx}', w_index=next_idx)

       
    def xai_batched_parallel(self, data_values, model, batch_size=1_000, background_data=None):
        batches = [
            data_values[i:i + batch_size]
            for i in range(0, len(data_values), batch_size)
        ]

        explainer = shap.PermutationExplainer(
            model.predict_proba,
            background_data
        )
    
        max_evals = 2 * background_data.shape[1] + 20

        n_jobs = min(cpu_count(), 10) 

        with Pool(processes=n_jobs) as pool:
            results = list(
                tqdm(
                    pool.imap(
                        _compute_shap_batch_worker_single,
                        [(model, background_data, batch, max_evals) for batch in batches]
                    ),
                    total=len(batches),
                    desc="Parallel SHAP",
                    unit="batch"
                )
            )
        return self.explanation_concatenate(results)
    
    def explanation_concatenate(self, explanations):
        # explanations is a list of shap.Explanation objects
        values = np.concatenate([e.values for e in explanations], axis=0)
        # base_values can be per-sample or single value (depends on model)
        base_values = explanations[0].base_values
        if isinstance(base_values, np.ndarray) and base_values.ndim == 1:
            base_values = np.concatenate(
                [e.base_values for e in explanations], axis=0
            )
        data = None
        if explanations[0].data is not None:
            data = np.concatenate([e.data for e in explanations], axis=0)
        
        # feature_names = explanations[0].feature_names
        output_names = explanations[0].output_names
        
        return shap.Explanation(
            values=values,
            base_values=base_values,
            data=data,
            feature_names=self.colonne,
            output_names=output_names
        )

    def save_explainer(self, explainer, filename, w_index):
        full_folder = f"{self.output_folder}/{self.model_name}/{w_index}"
        os.makedirs(full_folder, exist_ok=True)
        file_path = os.path.join(full_folder, filename)
        with open(file_path, 'wb') as f:
            pickle.dump(explainer, f)
        print(f"Explainer saved to {file_path}")


#  dataset_name, streamed_filename, clustering_method, drift_type, bias_spatial, W, model_name, slope=None

prove_da_fare = [
    ('bccc_cpacket', 'streaming_incremental_0.7_refine.csv', 'kmeans', 'incremental', 0.7, 10_000, 'lppnse', None),
]


for dataset_name, streamed_filename, clustering_method, drift_type, bias_spatial, W, model_name, slope in prove_da_fare:
    print(f"=== Avvio estrazione SHAP: {dataset_name} | {model_name} | {drift_type} | W={W} | slope={slope} ===")
    extract_online = ExtractXAIOnline(
        dataset_name=dataset_name,
        streamed_filename=streamed_filename,
        clustering_method=clustering_method,
        drift_type=drift_type,
        bias_spatial=bias_spatial,
        W=W,
        model_name=model_name,
        slope=slope
    )
    extract_online.testing_loop()

