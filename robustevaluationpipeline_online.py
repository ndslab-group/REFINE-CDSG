from train_ARF import ARFTrainer
from train_ARFA import ARFATrainer
from train_HAT import HatTrainer
from train_LPPNSE import LPPNSETrainer
from train_RF import RFTrainer
from train_XGB import XGBTrainer
from train_AXGB import AXGBTrainer

from tqdm import tqdm
import numpy as np
from robustevaluationpipeline import RobustEvaluationPipelineOffline
import os 
import pickle

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score
from matplotlib.ticker import MaxNLocator
        
class RobustEvaluationPipelineOnline:
    def __init__(self, dataset_name, streamed_filename, output_folder, clustering_method, drift_type, bias_spatial, W, model_name):
        self.dataset_name = dataset_name
        self.streamed_filename = streamed_filename
        self.output_folder = output_folder
        self.clustering_method = clustering_method
        self.drift_type = drift_type
        self.bias_spatial = bias_spatial
        self.W = W
        self.model_name = model_name

    def load_model(self):
        if self.model_name == 'rf':
            model = RFTrainer(data_dir=self.output_folder, W=self.W)
            model.load_model(w_index='train', folder_path=self.output_folder)
        if self.model_name == 'xgb':
            model = XGBTrainer(data_dir=self.output_folder, W=self.W)
            model.load_model(w_index='train', folder_path=self.output_folder)
        if self.model_name == 'axgb':
            model = AXGBTrainer(data_dir=self.output_folder, W=self.W)
            model.load_model(w_index='train', folder_path=self.output_folder)
        if self.model_name == 'arf':
            model = ARFTrainer(data_dir=self.output_folder, W=self.W)
            model.load_model(w_index='train', folder_path=self.output_folder)
        if self.model_name == 'arfa':
            model = ARFATrainer(data_dir=self.output_folder, W=self.W)
            model.load_model(w_index='train', folder_path=self.output_folder)
        if self.model_name == 'hat':
            model = HatTrainer(data_dir=self.output_folder, W=self.W)
            model.load_model(w_index='train', folder_path=self.output_folder)
        if self.model_name == 'lppnse':
            model = LPPNSETrainer(data_dir=self.output_folder, W=self.W)
            model.load_model(w_index='train', folder_path=self.output_folder)
        return model

    def extract_data_online(self):
        offline = RobustEvaluationPipelineOffline(
            dataset_name=self.dataset_name,
            streamed_filename=self.streamed_filename,
            output_folder=self.output_folder,
            clustering_method=self.clustering_method,
            drift_type=self.drift_type,
            bias_spatial=self.bias_spatial,
            W=self.W,
            model_name=self.model_name 
        )

        windows, colonne = offline.readStream()
        return windows
    
    def test_model(self, model, X_test):
        y_pred = model.predict(X_test)
        return y_pred
    
    def testing_loop(self):
        model = self.load_model() # caricamento del modello
        windows = self.extract_data_online() # caricamento delle finestre online
        results = []

        for i in tqdm(range(len(windows)), desc="Online evaluation"):
            print("Evaluation: ")
            X_test, y_test = windows[i]
            print(np.unique(y_test))
            y_pred = self.test_model(model=model, X_test=X_test)
            results.append([y_test, y_pred])
            ### retrain del modello
            if self.model_name not in ['rf', 'xgb']:
                model.retrain_model(X_retrain=X_test, y_retrain=y_test)
            model.save_model(w_index=i, folder_path=self.output_folder)

        ##### salvataggio dei risultati
        output_dir = f'{self.output_folder}/results/{self.model_name}'
        os.makedirs(output_dir, exist_ok=True)
        model_filename = f"{self.model_name}_results_{self.W}_{self.bias_spatial}.pkl"
        model_path = os.path.join(output_dir, model_filename)
        with open(model_path, "wb") as f:
            pickle.dump(results, f)

    def evaluate_results(self):
        W = self.W
        output_dir = f'{self.output_folder}/results/{self.model_name}'
        model_filename = f"{self.model_name}_results_{self.W}_{self.bias_spatial}.pkl"
        model_path = os.path.join(output_dir, model_filename)
        with open(model_path, "rb") as f:
            results = pickle.load(f)

        # estrazione di y_true ed y_pred dai risultati
        y_true_list = [r[0] for r in results] 
        y_pred_list = [r[1] for r in results]

        acc_list = []
        f1_list = []
        recall_list = []

        for yt, yp in zip(y_true_list, y_pred_list):
            yt_flat = np.ravel(yt)
            yp_flat = np.ravel(yp)
            acc_list.append(accuracy_score(yt_flat, yp_flat))
            f1_list.append(f1_score(yt_flat, yp_flat, average='macro'))
            recall_list.append(recall_score(yt_flat, yp_flat, average='binary'))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(acc_list, label='Accuracy', marker='o', color='skyblue', linewidth=2.5)
        ax.plot(f1_list, label='F1-score', marker='s', color='salmon', linewidth=2.5)
        ax.plot(recall_list, label='Recall', marker='x', color='darkgreen', linewidth=2.5)
        ax.set_xlabel("Streaming Batch", fontsize=16, fontweight='bold')
        # ax.set_ylabel("Score", fontsize=16, fontweight='bold')
        ax.set_title(f"{self.model_name.upper()} | W={self.W} | K={self.bias_spatial}", fontsize=18, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis='both', labelsize=14)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        ax.legend()
        legend = ax.legend(fontsize=14)
        for text in legend.get_texts():
            text.set_fontweight('bold')
        ax.grid(True)
        # Salva grafico in PDF
        pdf_path = os.path.join(output_dir, f"{self.model_name}_metrics.pdf")
        plt.tight_layout()
        plt.savefig(pdf_path, format='pdf')
        plt.close(fig)

        print(f"Grafico andamento metriche salvato in: {pdf_path}")

    
    def run_online_evaluation(self, _only_evaluate):
        if not _only_evaluate:
            print("Testing loop: ")
            self.testing_loop()
        print("Evaluate results: ")
        self.evaluate_results()