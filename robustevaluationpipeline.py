import pickle
import pandas as pd 
import numpy as np
import os
from tqdm import tqdm
from train_ARF import ARFTrainer
from train_ARFA import ARFATrainer
from train_HAT import HatTrainer
from train_LPPNSE import LPPNSETrainer
from train_RF import RFTrainer
from train_XGB import XGBTrainer
from train_AXGB import AXGBTrainer

class RobustEvaluationPipelineOffline:
    def __init__(self, dataset_name, streamed_filename, output_folder,clustering_method, drift_type, bias_spatial, W, model_name):
        self.dataset_name = dataset_name
        self.streamed_filename = streamed_filename
        self.output_folder = output_folder
        self.clustering_method = clustering_method
        self.drift_type = drift_type
        self.bias_spatial = bias_spatial
        self.W = W
        self.indici_offline = {}
        self.indici_online = {}
        self.random_state = 17
        
    def load_streamed_dataset(self):
        '''
        filename is directly the name of the csv file used
        '''
        folder_path = f'results_cdsg_directory_{self.dataset_name}/{self.clustering_method}'
        file_path = os.path.join(folder_path, self.streamed_filename)
        df_tmp = pd.read_csv(file_path)
        columns_to_drop = ['concept', 'WIN']
        # if self.dataset_name == 'xiiot':
        #     columns_to_drop = ['concept', 'WIN']
        # else:
        #     columns_to_drop = ['concept', 'WIN', 'flow_id']
        df_tmp.drop(columns=columns_to_drop, inplace=True)
        df_tmp.rename(columns={'macro_clusters': 'label'}, inplace=True)
        return df_tmp
    
    def load_data(self, _is_offline):
        '''
        Load the dataset online or offline splitted
        '''
        filename = f'online_evaluation_{self.dataset_name}_{self.clustering_method}_{self.drift_type}_bias{self.bias_spatial}_W{self.W}'
        if _is_offline:
            file_path = os.path.join(filename, 'offline_streaming.csv')
        else:
            file_path = os.path.join(filename, 'online_streaming.csv')
        df = pd.read_csv(file_path)
        if self.dataset_name != 'xiiot':
            df.drop(columns=['flow_id'], inplace=True)
        return df

    def _splitting_offline_online(self, data, split_index):
        '''
        split_index is the index that divides the offline dataset from the online streaming dataset
        '''
        offline_df = data.iloc[:split_index]
        online_df = data.iloc[split_index:]
        return offline_df, online_df

    def split_stage(self, split_index=20_000):
        streamed_df = self.load_streamed_dataset()
        offline_df, online_df = self._splitting_offline_online(streamed_df, split_index)
        output_folder = f'online_evaluation_{self.dataset_name}_{self.clustering_method}_{self.drift_type}_bias{self.bias_spatial}_W{self.W}'
        os.makedirs(output_folder, exist_ok=True)
        offline_df.to_csv(os.path.join(output_folder, 'offline_streaming.csv'), index=False)
        online_df.to_csv(os.path.join(output_folder, 'online_streaming.csv'), index=False)

    def extract_indexes_offline(self, data, _skip=False):
        attacchi = data[data['label'] == 1]
        normali = data[data['label'] == 0]
      
        print(f'Finestra W={self.W}: attacchi disponibili={len(attacchi)}, normali disponibili={len(normali)}')
        
        if _skip:
            print('Skip offline stage: viene restituita tutto il dataset offline')
            sampled_data = data.sample(n=len(data), random_state=self.random_state, replace=False)
        else:
            n_normal = int(self.W * self.bias_spatial)
            n_attack = self.W - n_normal
            print(f'Numero di normali: {n_normal}\nNumero di attacchi: {n_attack}')
            sampled_attack = attacchi.sample(n=n_attack, random_state=self.random_state, replace=False)
            sampled_normal = normali.sample(n=n_normal, random_state=self.random_state, replace=False)
            #### unisco i campioni estratti e procedo con lo shuffling del dataset successivo
            sampled_data = pd.concat([sampled_attack, sampled_normal])
            sampled_data = sampled_data.sample(frac=1, random_state=self.random_state)
        indexes = sampled_data['index'].tolist() # ritorna gli indici presi per il training set
        return indexes
    
    def extract_train_indexes(self, offline_df, skip=False):
        """
        Extracts the dataframe indices to be used in the training windows for each specified value of W.
        Specifically, it returns the train index values, which will be called later to retrieve the data
        """
        data = offline_df 
        data = data.reset_index(drop=True)
        data['index'] = data.index
       
        for w in [self.W]:
            data = data.sample(frac=1).reset_index(drop=True)
            indexes = self.extract_indexes_offline(data, _skip=skip)
            self.indici_offline[w] = indexes
            print(f'Campioni estratti per la finestra di dimensione {w}: {len(indexes)}')
        self.save_indexes()
    
    def save_indexes(self):
        # folder = f'online_evaluation_{self.dataset_name}_{self.clustering_method}_{self.drift_type}_bias{self.bias_spatial}_W{self.W}'
        os.makedirs(self.output_folder, exist_ok=True)
        output_file = os.path.join(self.output_folder, f"indices.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(self.indici_offline, f)

    def load_indexes(self):
        # folder = f'online_evaluation_{self.dataset_name}_{self.clustering_method}_{self.drift_type}_bias{self.bias_spatial}_W{self.W}'
        output_file = os.path.join(self.output_folder, f"indices.pkl")
        with open(output_file, "rb") as f:
            self.indici_offline = pickle.load(f)
               
    def offline_stage(self, skip=False):
        ''' 
        Loading the dataset offline, extracting the indices for the training set, and saving them to a CSV file
        '''
        offline_df = self.load_data(_is_offline=True)
        print(f"Offline df: {len(offline_df)}")
        self.extract_train_indexes(offline_df, skip=skip)
    
    def pipeline_offline(self, split_indedx=None, skip=False):
        if split_indedx is not None:
            self.split_stage(split_index=split_indedx)
        self.offline_stage(skip=skip)
            
    def retrieve_windows_offline(self):
        data = self.load_data(_is_offline=True)
        colonne = list(data.columns)
        self.load_indexes() 
        print(f'Finestre usate: {self.indici_offline.keys()}')
        # data = self.offline # carico i dati di train
        print(data.shape)
        data = data.reset_index(drop=True)
        data['index'] = data.index
        train_windows = {}
        val_windows = {}
        indexes = self.indici_offline[self.W]
        # Training window: solo gli indici selezionati
        train_window = data[data['index'].isin(indexes)]
        # Validation set: tutti gli altri
        val_window = data[~data['index'].isin(indexes)]
        ### faccio sottocampionamento della validazione al 30%
        val_window = (
            val_window.groupby("label", group_keys=False)
            .sample(frac=0.3, random_state=42)
        )
        # aggiungo ai dizionari
        train_windows[self.W] = train_window
        val_windows[self.W] = val_window
        print(f"Finestra {self.W}. train={len(train_window)}, val={len(val_window)}")
        # dividi train, validation e test in X e y
        print(f'Attacchi in train: {len(train_window[train_window["label"] == 1])}')
        print(f'Benigni in train: {len(train_window[train_window["label"] == 0])}')
        print(f'Attacchi in val: {len(val_window[val_window["label"] == 1])}')
        print(f'Benigni in val: {len(val_window[val_window["label"] == 0])}')
        X_train = train_window.iloc[:,:-2].to_numpy() # toglie index e label 
        y_train = train_window.iloc[:,-2].to_numpy() # prende la label
        # validation e test 
        X_val = val_window.iloc[:,:-2].to_numpy() # toglie index e label
        y_val = val_window.iloc[:,-2].to_numpy() # prende la label
        # stampa delle dimensione
        print(f'Training shape: ({X_train.shape}, {y_train.shape})')
        print(f'Validation shape: ({X_val.shape}, {y_val.shape})')
        return X_train, y_train, X_val, y_val, colonne 
    
    def readStream(self):
        online_df = self.load_data(_is_offline=False)
        colonne = list(online_df.columns)
        windows = []
        start_idx = 0
        max_len_stream = len(online_df)
        max_len_window = self.W
        n_windows = max_len_stream // self.W
        print(f'Dimensione finestra: {self.W}\nNumero finestre: {n_windows}')
        for _ in tqdm(range(n_windows)):
            print(f'Start index: {start_idx}')
            if (start_idx+self.W) > max_len_stream:
                window = online_df.iloc[start_idx:,:]
                # windows.append(last_window)
            else:
                end_idx = start_idx+max_len_window
                window = online_df.iloc[start_idx:end_idx,:]
            X_test = window.iloc[:,:-1].to_numpy()
            y_test = window.iloc[:,-1].to_numpy()
            windows.append([X_test, y_test])
            start_idx += max_len_window
        return windows, colonne

    def trainARF(self, X_train, y_train):
        arf = ARFTrainer(data_dir=self.output_folder, W=self.W)
        arf.train_eval_arf(X_train=X_train, y_train=y_train)
        arf.save_model(w_index='train', folder_path=self.output_folder)

    def trainARFA(self, X_train, y_train):
        arfa = ARFATrainer(data_dir=self.output_folder, W=self.W)
        arfa.train_eval_arfa(X_train=X_train, y_train=y_train)
        arfa.save_model(w_index='train', folder_path=self.output_folder)

    def trainAXGB(self, X_train, y_train):
        axgb = AXGBTrainer(data_dir=self.output_folder, W=self.W)
        axgb.train_eval_axgb(X_train=X_train, y_train=y_train)
        axgb.save_model(w_index='train', folder_path=self.output_folder)

    def trainLPPNSE(self, X_train, y_train):
        lppnse = LPPNSETrainer(data_dir=self.output_folder, W=self.W)
        lppnse.train_eval_model(X_train=X_train, y_train=y_train)
        lppnse.save_model(w_index='train', folder_path=self.output_folder)

    def trainHAT(self, X_train, y_train):
        hat = HatTrainer(data_dir=self.output_folder, W=self.W)
        hat.train_eval_model(X_train=X_train, y_train=y_train)
        hat.save_model(w_index='train', folder_path=self.output_folder)

    def trainRF(self, X_train, y_train):
        rf = RFTrainer(data_dir=self.output_folder, W=self.W)
        rf.train_eval_rf(X_train=X_train, y_train=y_train)
        rf.save_model(w_index='train', folder_path=self.output_folder)

    def trainXGB(self, X_train, y_train):
        xgb = XGBTrainer(data_dir=self.output_folder, W=self.W)
        xgb.train_eval_xgb(X_train=X_train, y_train=y_train)
        xgb.save_model(w_index='train', folder_path=self.output_folder)

    def training_loop(self, model_name):
        X_train, y_train, _, _, _ = self.retrieve_windows_offline()
        print(f'Model name: {model_name}')
        if model_name == 'rf':
            self.trainRF(X_train=X_train, y_train=y_train)
        if model_name == 'xgb':
            self.trainXGB(X_train=X_train, y_train=y_train)
        if model_name == 'axgb':
            self.trainAXGB(X_train=X_train, y_train=y_train)
        if model_name == 'arf':
            self.trainARF(X_train=X_train, y_train=y_train)
        if model_name == 'arfa':
            self.trainARFA(X_train=X_train, y_train=y_train)
        if model_name == 'lppnse':
            self.trainLPPNSE(X_train=X_train, y_train=y_train)
        if model_name == 'hat':
            self.trainHAT(X_train=X_train, y_train=y_train)
    