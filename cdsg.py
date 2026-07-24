from datetime import date
import os
from concept_generator import ConceptGenerator
from streamer_generator import StreamerGen
from utils import Utility
import csv
import pandas as pd
from pathlib import Path


class CDSG:
    def __init__(self, directory_name='datasets', filename='source_dataset.csv', directory_stream='results_cdsg_directory'):
        print("Avviare CDSG")
        df = self.opencsv(directory_name, filename)
        self.directory_name = directory_name 
        self.filename = filename
        self.directory_stream = directory_stream
        # rimozione di eventuali colonne e rinominazione delle colonne
        self.source_df = df.drop(columns=["activity", "timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "protocol"]) # BCCC-cPacket
        # self.source_df = df.drop(columns=["timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "protocol", "Attack Type"]) # drop features for BCCC-CICIDS
        # self.source_df = df.drop(columns=["Date", "Timestamp", "activity"]) # drop features for X-IIoTID
        # self.source_df['label'] = self.source_df['label'].map({'Benign': 0, 'Attack': 1, 'Normal':0}) ## for X-IIoTID e BCCC-CICIDS
        self.source_df['label'] = self.source_df['label'].map({'Benign': 0, 'Attack': 1, 'Suspicious':1}) ## for BCCC-cPacket
        self.col_data = self.source_df.columns
        
    def opencsv(self, directory_name, filename):
        df = pd.read_csv(os.path.join(directory_name,filename))
        return df

    def concept_generator(self, n_macro_clusters, list_micro_clusters, perc_neg, list_samples, clustering_technique, mode_classifier='m2m', _skip_micro=False):
        data_path = os.path.join(os.getcwd(), self.directory_name)
        cg = ConceptGenerator(
                df_file=os.path.join(data_path, self.filename), 
                data_path=data_path,
                col_data=len(self.col_data)-1, # the last column is the label one
                col_names=self.col_data,
                clustering_technique = clustering_technique,
                directory_stream = self.directory_stream,
                mode_classifier=mode_classifier
        )
        if mode_classifier != 'o2m':
            if _skip_micro:
                cg.pipeline_only_generation(
                    n_samples=-1,
                    perc_neg=perc_neg, 
                    list_samples=list_samples, 
                    save_dataframe=True,
                    filename=f'concept_cf_{clustering_technique}_2.csv',
                    need_preprocess=True, 
                    preprocess_df=True)
            else:
                cg.pipeline(
                    n_macro_clusters=n_macro_clusters, 
                    list_micro_clusters=list_micro_clusters, 
                    n_samples=-1,
                    perc_neg=perc_neg, 
                    list_samples=list_samples, 
                    save_dataframe=True,
                    filename=f'concept_cf_{clustering_technique}.csv',
                    need_preprocess=True, 
                    preprocess_df=True)
        else:
            ### many2many and one2many differs only in the cluster aggregation step, hence the only generation pipeline is the same for both
            cg.pipeline_one2many(
                n_samples=-1,
                perc_neg=perc_neg, 
                list_samples=list_samples, 
                save_dataframe=True,
                filename=f'concept_cf_{clustering_technique}_o2m.csv',
            )
        
    def incremental_concept_generator(self, list_micro_clusters_incremental, clustering_technique):
        data_path = os.path.join(os.getcwd(), self.directory_name)
        cg = ConceptGenerator(
                df_file=os.path.join(data_path, self.filename), 
                data_path=data_path,
                col_data=len(self.col_data)-1, # togliamo la label
                col_names=self.col_data,
                clustering_technique = clustering_technique,
                directory_stream = self.directory_stream,
                mode_classifier='m2m'
        )
        cg.incremental_generator(
            df_name_path=f'concept_cf_{clustering_technique}.csv',
            list_micro_clusters_incremental=list_micro_clusters_incremental,
            filename=f'concept_incremental_{clustering_technique}.csv',
            save_dataframe=True
        )

    def stream_generator(self, k, drift_type, drift_temporal_annotations, clustering_technique, intensity='auto', slope=1.0, spatial_bias_lists=None):
        # k is defines as the ratio between benignant and malicious samples on the total number. Hence the ratio of malicious samples is defines as malicious_k = 1-k
        malignant_k = 1-k  
        
        cartella = os.path.join(os.getcwd(), self.directory_stream, clustering_technique)
        if not os.path.exists(cartella):
            os.makedirs(cartella)
        
        file_path = os.path.join(cartella, f'concept_cf_{clustering_technique}.csv') ## loading the concept dataset
        
        streamer = StreamerGen(
            dataframe_name=file_path, 
            plotting=True, 
            col_data=len(self.col_data)-1,
            clustering_technique=clustering_technique,
            directory_stream=self.directory_stream) # sto passando il dataset già processato con i valori di macro_clusters!
        
        data = streamer.getdataframe() #### dataframe used for streaming initialized

        if drift_type == 'sudden':
            win_size, start_drift = drift_temporal_annotations
            streamer.sudden_drift_generator(
                win_size=win_size, 
                start_drift=start_drift, 
                perc_malignant_concept=malignant_k,
                perc_malignant_drift=malignant_k, 
                intensity_mode=intensity,
                save_stream=True, 
                filename=f"streaming_sudden_{k}_refine.csv",
                nome_file_parametri=f"params_sudden_stream_{k}_refine"
                )

        if drift_type == 'recurrent_sudden':
            win_size, start_drift, rec_drift = drift_temporal_annotations
            streamer.recurrent_drift_generator(
                win_size=win_size, 
                start_drift=start_drift, 
                rec_drift=rec_drift,
                perc_malignant_concept=malignant_k,
                perc_malignant_drift=malignant_k, 
                intensity_mode=intensity,
                save_stream=True, 
                filename=f"streaming_recurrent_{k}_refine.csv",
                nome_file_parametri=f"params_recurrent_stream_{k}_refine"
                ) 
            
        if drift_type == 'gradual':
            win_size, start_drift, width_drift = drift_temporal_annotations
            streamer.gradual_drift_generator(
                win_size=win_size, 
                start_drift=start_drift, 
                width_drift=width_drift,
                perc_malignant_concept=malignant_k,
                perc_malignant_drift=malignant_k, 
                save_stream=True, 
                filename=f"streaming_gradual_{k}_{slope}_refine.csv",
                nome_file_parametri=f"params_gradual_stream_{k}_{slope}_refine",
                slope=slope
                )
            
        if drift_type == 'incremental':
            win_size, list_starts = drift_temporal_annotations
            malignant_k_list = [1-x for x in spatial_bias_lists]
            source_df_filename = f'concept_incremental_{clustering_technique}.csv'
            streamer.incremental_drift_generator(
                win_size=win_size, # 120_000
                list_starts=list_starts, # [0, start_B, start_C], # ---> A, B, C
                concept_df_incremental_filename=source_df_filename,
                spatial_biases=malignant_k_list,
                save_stream=True, 
                filename=f"streaming_incremental_{k}_refine.csv",
                nome_file_parametri=f"params_incremental_{k}_refine",
            )

    def run_cdsg(self, _runcg, _run_ds, n_macro_clusters, list_micro_clusters, perc_neg, list_samples, k, drift_type, drift_temporal_annotations, clustering_technique, intensity='auto', mode_classifier='m2m', _skip_micro=False, slope=1.0, spatial_biases_list=None):
        # n_macro_clusters, list_micro_clusters, perc_neg, list_samples ----> concept generator parameters
        # k, drift_type, drift_temporal_annotations, intensity='auto' ---> drift streamer parameters
        # runcg: run concept generator, for concepts generation and partition in concept and drift datasets
        # runds: run drift streamer, for the effective generation of the streaming, starting from the dataset divided into concepts
        if _runcg:
            self.concept_generator(n_macro_clusters, list_micro_clusters, perc_neg, list_samples, clustering_technique, mode_classifier, _skip_micro)
        
        if _run_ds:
            if drift_type != 'incremental':
                self.stream_generator(k, drift_type, drift_temporal_annotations, clustering_technique, slope=slope, spatial_bias_lists=None)
            else:
                self.stream_generator(k, drift_type, drift_temporal_annotations, clustering_technique, slope=slope, spatial_bias_lists=spatial_biases_list)
    
    def run_only_cg(self, n_macro_clusters, list_micro_clusters, perc_neg, list_samples, clustering_technique, mode_classifier='m2m', _skip_micro=False):
        # n_macro_clusters, list_micro_clusters, perc_neg, list_samples ----> concept generator parameters
        # clustering technique: tecnica di clustering (supervised, agglomerative, kmeans)
        # runcg: run concept generator, for concepts generation and partition in concept and drift datasets
        self.concept_generator(n_macro_clusters, list_micro_clusters, perc_neg, list_samples, clustering_technique, mode_classifier, _skip_micro)
    

    def run_only_ds(self, k, drift_type, drift_temporal_annotations, clustering_technique, spatial_biases_list=None, intensity='auto', slope=1.0):
        # k, drift_type, drift_temporal_annotations, intensity='auto' ---> drift streamer parameters
        # runds: run drift streamer, for the effective generation of the streaming, starting from the dataset divided into concepts
        if drift_type != 'incremental':
            self.stream_generator(k, drift_type, drift_temporal_annotations, clustering_technique, slope=slope, spatial_bias_lists=None)
        else:
            self.stream_generator(k, drift_type, drift_temporal_annotations, clustering_technique, slope=slope, spatial_bias_lists=spatial_biases_list)
