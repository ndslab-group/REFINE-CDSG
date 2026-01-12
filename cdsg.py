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
        # rimozione di eventuali colonne e rinominazione delle colonne
        self.source_df = df.drop(columns=["activity", "timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "protocol"]) # rimozione delle features che non ci servono) # rimozione delle features che non ci servono
        self.source_df['label'] = self.source_df['label'].map({'Benign': 0, 'Attack': 1})
        self.col_data = self.source_df.columns
        
    def opencsv(self, directory_name, filename):
        df = pd.read_csv(os.path.join(directory_name,filename))
        return df

    def concept_generator(self, n_macro_clusters, list_micro_clusters, perc_neg, list_samples):
        data_path = os.path.join(os.getcwd(), self.directory_name)
        cg = ConceptGenerator(
                df_file=os.path.join(data_path, self.filename), 
                col_data=len(self.col_data)-1, # togliamo la label
                col_names=self.col_data
        )
        # generazione del concetto e salvataggio del dataset con concetto e drift.
        cg.pipeline(
            n_macro_clusters=n_macro_clusters, 
            list_micro_clusters=list_micro_clusters, 
            n_samples=-1,
            perc_neg=perc_neg, 
            list_samples=list_samples, 
            save_dataframe=True,
            filename='concept_cf.csv',
            need_preprocess=True, 
            preprocess_df=True)

    def stream_generator(self, k, drift_type, drift_temporal_annotations, intensity='auto'):
        # k viene definito come rapporto di benigni sul totale, mentre malignant_k è il complementare,
        # che definisce il rapporto di maligni sul totale: malignant_k = 1 -k
        malignant_k = 1-k  
        # devo riprendere il file del concetto per poterlo passare allo streamer
        cartella = os.path.join(os.getcwd(), "results_cdsg_directory")
        if not os.path.exists(cartella):
            os.makedirs(cartella)
        
        file_path = os.path.join(cartella, f"{'concept_cf.csv'}")
        
        streamer = StreamerGen(
            dataframe_name=file_path, 
            plotting=True, 
            col_data=len(self.col_data)-1) # sto passando il dataset già processato con i valori di macro_clusters!
        
        data = streamer.getdataframe()

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
                nome_file_parametri=f"params_sudden_stream_{k}refine"
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
                nome_file_parametri=f"params_recurrent_stream_{k}refine"
                ) 


    def run_cdsg(self, _runcg, _run_ds, n_macro_clusters, list_micro_clusters, perc_neg, list_samples, k, drift_type, drift_temporal_annotations, intensity='auto'):
        # n_macro_clusters, list_micro_clusters, perc_neg, list_samples ----> parametri del concept generator
        # k, drift_type, drift_temporal_annotations, intensity='auto' ---> parametri del drift streamer
        # runcg: run concept generator, per la generazione dei concetti e della suddivisione in concept e drift dataset
        # runds: run drift streamer, per la generazione effettiva dello streaming, partendo però dal dataset suddivisio in concetti.
        if _runcg:
            self.concept_generator(n_macro_clusters, list_micro_clusters, perc_neg, list_samples)
        
        if _run_ds:
            self.stream_generator(k, drift_type, drift_temporal_annotations)
        