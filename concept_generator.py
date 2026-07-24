from datetime import date
import pickle
import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from utils import Utility
from ClassifierClusters import ClassifierClusters
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from ClassifierClusters_o2m import ClassifierClustersO2M
from sklearn.preprocessing import MinMaxScaler
import csv
from DEC import DEC
from pathlib import Path
import json

class ConceptGenerator:
    def __init__(self, df_file, col_data, col_names, data_path, clustering_technique, directory_stream, mode_classifier):
        self.df = pd.read_csv(df_file)
        # self.df = self.df.drop(columns=[""]) ## remove column if needed (es: flow_id, timestamp, etc) from SOURCE dataset
        self.df = self.df.drop(columns=["activity", "timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "protocol"]) # drop features for BCCC-cPacket
        # self.df = self.df.drop(columns=["timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "protocol", "Attack Type"]) # drop features for BCCC-CICIDS
        # self.df = self.df.drop(columns=["Date", "Timestamp", "activity"]) # drop features for X-IIoTID
        # self.df['label'] = self.df['label'].map({'Normal': 0, 'Attack': 1})   # for X-IIoTID and BCCC-CICIDS
        self.df['label'] = self.df['label'].map({'Benign': 0, 'Attack': 1, 'Suspicious':1}) # for BCCC - cPacket
        print(self.df.iloc[:,-1].value_counts(normalize=True))
        self.df = self.df.rename(columns={'label': 'macro_clusters'})
        self.data_path = data_path
        self.col_data = col_data
        self.mode_classifier = mode_classifier ### could be m2m or o2m
        # directory stream/clustering_technique
        self.clustering_technique = clustering_technique
        self.directory_stream = directory_stream
        self.utility = Utility(col_data=col_data, col_names=col_names)

    def get_df(self):
        return self.df

    def print_df(self, df, n_row=None):
        if n_row:
            print(df.head(n_row))
        else:
            print(df.head())

    def macro_clustering(self, X, n_clusters=2):
        clusters = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
        return clusters.labels_

    def micro_clustering(self, X, n_clusters):
        if self.clustering_technique == 'kmeans':
            clusters = KMeans(n_clusters=n_clusters, random_state=27, n_init="auto").fit(X)
        elif self.clustering_technique == 'agglomerative':
            clusters = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
        elif self.clustering_technique == 'dec':
            model = DEC(n_clusters=n_clusters)
            centers, labels = model.fit_predict(X.to_numpy())
            return centers, labels
        return (clusters.cluster_centers_, clusters.labels_)

    def micro_macro_clustering(self, df, n_macro_clusters=2, list_micro_clusters=[], need_preprocess=True):
        '''
        Step 1: data preprocessing for the creation of macro- and micro-clusters

        Function for creating a macro-micro cluster partitioning

        * `n_macro_clusters` divides the input DataFrame into macro-clusters
        * `list_micro_clusters` contains two elements indicating, respectively, the number of clusters for each macro-cluster

        '''
        if "flow_id" in df.columns:
            flow_ids = df["flow_id"].copy()
        if len(list_micro_clusters) != n_macro_clusters:
            raise ValueError("Invalid number error! You must specify the number of micro clusters per macro cluster in the list. The list must have a size equal to the number of macroclusters.")
        
        new_df = df.copy(deep=True)
        
        if not need_preprocess:
            # if there is no need the macro_cluster label is already present in the dataframe
            X = df.drop(columns=["flow_id"]) if "flow_id" in df.columns else df
            macro_clusters = self.macro_clustering(X, n_clusters=n_macro_clusters)
            new_df['macro_clusters'] = macro_clusters # definitions of macro clusters

        new_df['micro_clusters'] = -1
        
        # centroids for microclusters
        centers_micro_clusters = []
        for i in range(n_macro_clusters):
            macro_df = new_df[new_df['macro_clusters'] == i]
            X_micro = macro_df.drop(columns=["flow_id"]) if "flow_id" in macro_df.columns else macro_df
            centers, labels_micro = self.micro_clustering(X_micro, n_clusters=list_micro_clusters[i])
            centers_micro_clusters.append(centers)
            new_df.loc[new_df['macro_clusters'] == i, 'micro_clusters'] = labels_micro
        
        # add again flow_id se esiste
        if "flow_id" in df.columns:
            new_df["flow_id"] = flow_ids.values
        
        return (new_df, centers_micro_clusters)
    

    def classifier_clusters(self, dataframe, plotting=True, loading=False):
        '''
        This function allows you to associate a similarity metric between clusters.
        It requires the following inputs:
        - data frame divided into macro clusters
        - returns:
        - partition into micro clusters
        - similarity matrix
        - internally calls a class to train a classifier (which may differ).
        The classifier must have two important methods:
        - fit
        - predict_proba

        - externally calls two methods:
        - train classifier
        - test classifier (which returns accuracy, f1 score, classification report)
        Furthermore, given the samples, it allows you to calculate the similarity matrix.
        The method returns the similarity matrix found and the dictionary of neighboring clusters.
        '''
        if "flow_id" in dataframe.columns:
            flow_ids = dataframe["flow_id"].copy()
        X = dataframe.drop(columns=["flow_id"]) if "flow_id" in dataframe.columns else dataframe
        if self.mode_classifier != 'o2m':
            # train classifiers for each macro cluster
            clfclt = ClassifierClusters(dataframe=X, col_data=self.col_data, folder_save=f'{self.directory_stream}/{self.clustering_technique}')
            if loading:
                # only loading
                with open(f"{self.directory_stream}/{self.clustering_technique}/pos_dict.pkl","rb") as f:
                    pos_dict = pickle.load(f)
                with open(f"{self.directory_stream}/{self.clustering_technique}/neg_dict.pkl", "rb") as f:
                    neg_dict = pickle.load(f)
                pos_mst = clfclt.extract_minimum_spanning_tree(dict_adj=pos_dict)
                neg_mst = clfclt.extract_minimum_spanning_tree(dict_adj=neg_dict)
                return pos_mst, neg_mst
            
            # classifiers trined on benignant and malicious samples
            print("Step 2: training classifiers")
            models = clfclt.train_test_loop(plots=plotting)
            
            # dict from adjacency similarities matrix
            print("Step 3: similarities matrix")
            pos_dict, neg_dict = clfclt.compute_graphs(dataframe=X, clfs=models)
            ### save of the dict
            data_path = Path(self.directory_stream)
            data_path.mkdir(parents=True, exist_ok=True)
            with open(f"{self.directory_stream}/{self.clustering_technique}/pos_dict.pkl", "wb") as f:
                pickle.dump(pos_dict, f)
            with open(f"{self.directory_stream}/{self.clustering_technique}/neg_dict.pkl", "wb") as f:
                pickle.dump(neg_dict, f)

            print(f"pos_dict salvato in: {self.directory_stream}/{self.clustering_technique}/pos_dict.pkl")
            print(f"neg_dict salvato in: {self.directory_stream}/{self.clustering_technique}neg_dict.pkl")
            
            # extracting dict from MST using Kruscal
            print("Step 4: MST and Kruscal")
            pos_mst = clfclt.extract_minimum_spanning_tree(dict_adj=pos_dict)
            neg_mst = clfclt.extract_minimum_spanning_tree(dict_adj=neg_dict)
            return pos_mst, neg_mst
        else:
            clf_o2m = ClassifierClustersO2M(dataframe=X, col_data=self.col_data, folder_save=f'{self.directory_stream}/{self.clustering_technique}/o2m')
        
            print("Step 2: training classifiers")
            models = clf_o2m.train_test_loop()
            
            print("Step 3: similarities matrix")
            pos_dict, neg_dict = clf_o2m.compute_graphs(dataframe=X, all_models=models)
            
            data_path = Path(self.directory_stream)
            data_path.mkdir(parents=True, exist_ok=True)
            with open(f"{self.directory_stream}/{self.clustering_technique}/o2m/pos_dict.pkl", "wb") as f:
                pickle.dump(pos_dict, f)
            with open(f"{self.directory_stream}/{self.clustering_technique}/o2m/neg_dict.pkl", "wb") as f:
                pickle.dump(neg_dict, f)

            print(f"pos_dict salvato in: {self.directory_stream}/{self.clustering_technique}/o2m/pos_dict_o2m.pkl")
            print(f"neg_dict salvato in: {self.directory_stream}/{self.clustering_technique}/o2m/neg_dict_02m.pkl")

            print("Step 4: MST")
            pos_mst = clf_o2m.extract_minimum_spanning_tree(dict_adj=pos_dict)
            neg_mst = clf_o2m.extract_minimum_spanning_tree(dict_adj=neg_dict)
            return pos_mst, neg_mst


    def extract_samples_dict(self, dataframe):
        '''
        extracts for a given dataframe how many samples are present for each macro and micro cluster in the format:
        { macro: {micro: x_samples} }
        '''
        ret_dict = {}
        ret_dict[0] = {} # macro cluster 0
        ret_dict[1] = {} # macro cluster 1
        
        values_macro0 = dataframe.loc[dataframe['macro_clusters'] == 0, 'micro_clusters'].value_counts()
        values_macro1 = dataframe.loc[dataframe['macro_clusters'] == 1, 'micro_clusters'].value_counts()
        
        print('Values macro benevoli ', values_macro0)
        print('Values macro malevoli ', values_macro1)
        
        filename = f"{self.directory_stream}/{self.clustering_technique}/macro_values.txt" if self.mode_classifier != 'o2m' else f"{self.directory_stream}/{self.clustering_technique}/o2m/macro_values.txt"
        with open(filename, "w") as f:
            print("Questa è la partizione dei micro clusters per ogni macro clusters", file=f)
            print("Quando verranno presi i primi X micro cluster per creare il concept bisognerà tenerne conto per lo streamer", file=f)
            print(f"Values macro benevoli: {values_macro0}\n", file=f)
            print(f"Values macro malevoli: {values_macro1}\n", file=f)
        
        for i in range(len(values_macro0)):
            ret_dict[0][i] = values_macro0[i]
        for i in range(len(values_macro1)):
            ret_dict[1][i] = values_macro1[i]
        return ret_dict

    def samples_fraction(self, mst_dict, samples_dict, n_samples, n_clusters=None):
        '''
        Returns the nodes from which to sample by breadth-first searching the input mst tree.
        If samples dict is -1, then it takes entire partitions IN ORDER.
        If samples dict is not -1, then it samples the chosen number of samples, and until it samples exactly
        a number of samples equal to the selected one, it takes entire or partial partitions.
        '''
        list_nodes = []
        for node, sons in mst_dict.items():
            if node not in list_nodes:
                list_nodes.append(node)
                while sons:
                    son = sons.pop(0)
                    if son[0] not in list_nodes:
                        list_nodes.append(son[0])

        partitions = {}
        if n_samples == -1:
            for i in range(n_clusters):
                node = list_nodes[i]
                x_samples = samples_dict[node]
                partitions[node] = x_samples
        else:
            while (n_samples > 0):
                node = list_nodes.pop(0)
                x_samples = samples_dict[node]
                if x_samples > n_samples:
                    partitions[node] =  n_samples
                else:
                    partitions[node] = x_samples
                n_samples -= x_samples
        return partitions

    def create_concept(self, dataframe, pos_mst, neg_mst, n_samples, perc_neg, list_samples=[], plotting=True):
        '''
        This function creates a concept. pos_mst and neg_mst are the minimum spanning trees.
        If we want a fixed number of samples in the concept, then we specify n_samples and perc_neg.
        However, this could lead to incomplete datasets, resulting in greater sample sparsity: useful for fooling a classifier.
        If n_samples is -1, then we use list_samples, which specifies the number of clusters to use for positive and negative samples.
        So if it's -1, the value is automatically added!
        List samples is composed of two elements: n_positive_clusters, n_negative_clusters, taking into account that these are already sorted.
        Returns a dataframe in the format:
        [dim1, dim2, ...dimk, macro, micro, concept]
        '''
        print(f'Dimensione totale del dataframe: {len(dataframe)}')
        print(f'Viene richiesto un concetto da {n_samples} campioni, di cui {perc_neg*100}% maligni')
        samples_dict = self.extract_samples_dict(dataframe=dataframe)
        print(f'Dizionario dei campioni: {samples_dict}')
        if n_samples == -1:
            samples_pos = self.samples_fraction(mst_dict=pos_mst, samples_dict=samples_dict[0], n_samples=-1, n_clusters=list_samples[0])
            samples_neg = self.samples_fraction(mst_dict=neg_mst, samples_dict=samples_dict[1], n_samples=-1, n_clusters=list_samples[1])
        else:
            n_neg_samples = round(n_samples * perc_neg)
            n_pos_samples = n_samples - n_neg_samples
            print(f'Campioni positivi: {n_pos_samples} \nCampioni negativi: {n_neg_samples}')
            samples_pos = self.samples_fraction(mst_dict=pos_mst, samples_dict=samples_dict[0], n_samples=n_pos_samples)
            samples_neg = self.samples_fraction(mst_dict=neg_mst, samples_dict=samples_dict[1], n_samples=n_neg_samples)
        print(f'Partizione benevoli: {samples_pos}')
        print(f'Partizione malevoli: {samples_neg}')
        print(f'Dizionario dei campioni: {samples_dict}')
        
        name_file = f"{self.directory_stream}/{self.clustering_technique}/elaborazione_concetto_2.txt" if self.mode_classifier != "o2m" else f"{self.directory_stream}/{self.clustering_technique}/o2m/elaborazione_concetto_o2m.txt"
        with open(name_file, "w", encoding="utf-8") as f:
            f.write(f"Partizione benevoli: {samples_pos}\n")
            f.write(f"Partizione malevoli: {samples_neg}\n")
            f.write(f"Dizionario dei campioni: {samples_dict}\n")

        df_copy = dataframe.copy(deep=True)
        df_copy['concept'] = 0 
        
        pos_df = df_copy[df_copy['macro_clusters'] == 0].copy()
        neg_df = df_copy[df_copy['macro_clusters'] == 1].copy()
        
        for cluster, n_samples_cluster in samples_pos.items():
            print(f'Cluster {cluster}, samples {n_samples_cluster}')
            samples_indices_pos = np.random.choice(pos_df[pos_df['micro_clusters'] == cluster].index, size=n_samples_cluster, replace=False)
            pos_df.loc[samples_indices_pos, 'concept'] = 1
        
        for cluster, n_samples_cluster in samples_neg.items():
            print(f'Cluster {cluster}, samples {n_samples_cluster}')
            samples_indices_neg = np.random.choice(neg_df[neg_df['micro_clusters'] == cluster].index, size=n_samples_cluster, replace=False)
            neg_df.loc[samples_indices_neg, 'concept'] = 1
        
        unique_df = pd.concat([pos_df, neg_df], axis=0, ignore_index=True)
        print(f"Expected dim: {len(pos_df) + len(neg_df)}, real dim: {len(unique_df)}")
        return unique_df
    
    def abnomaly_scorer(self, dataframe):
        inliers = dataframe[dataframe['concept'] == 1]
        outliers = dataframe[dataframe['concept'] == 0]
        X_train = inliers.iloc[:,:self.col_data].to_numpy()
        X_anomaly = outliers.iloc[:,:self.col_data].to_numpy()
        clf = IsolationForest(n_estimators=50, warm_start=True)
        clf.fit(X_train)
        scores = -clf.score_samples(X_anomaly)
        return scores
    
    def fit_scorer(self, dataframe):
        inliers = dataframe[dataframe['concept'] == 1]
        X_train = inliers.iloc[:,:self.col_data].to_numpy()
        clf = IsolationForest(n_estimators=50, warm_start=True)
        clf.fit(X_train)
        return clf
    
    def apply_abnomaly_scorer(self, dataframe, plot_step=True):
        abn_df = dataframe.copy()
        if "flow_id" in dataframe.columns:
            flow_ids = dataframe["flow_id"].copy()
        abn_df = abn_df.drop(columns=["flow_id"]) if "flow_id" in dataframe.columns else dataframe
        
        clf = self.fit_scorer(dataframe=abn_df)
        outlier_scores = clf.decision_function(abn_df[abn_df['concept']==0].iloc[:,:self.col_data].to_numpy())

        scaler = MinMaxScaler(feature_range=(0, 2))
        scaled_scores = scaler.fit_transform(outlier_scores.reshape(-1, 1)).flatten()

        truncated_scores = np.trunc(scaled_scores * 10) / 10
        truncated_scores = 2.0 - truncated_scores

        abn_df['anomaly'] = 0.0
        abn_df.loc[abn_df['concept'] == 0, 'anomaly'] = truncated_scores
        X_anomaly = abn_df[abn_df['concept'] == 0].iloc[:,:self.col_data].to_numpy()
        y_anomaly = abn_df[abn_df['concept'] == 0].loc[:,'anomaly'].to_numpy()
        if plot_step:
            self.utility.plot_contours_outlier(X_anomaly, y_anomaly, abn_df[abn_df['concept'] == 0], clf)
        
        if "flow_id" in dataframe.columns:
            abn_df.insert(0, 'flow_id', flow_ids.values)
        return abn_df

    def bfs_order(self, mst_dict):
        list_nodes = []
        for node, sons in mst_dict.items():
            if node not in list_nodes:
                list_nodes.append(node)
                sons_copy = list(sons)  
                while sons_copy:
                    son = sons_copy.pop(0)
                    if son[0] not in list_nodes:
                        list_nodes.append(son[0])
        return list_nodes

    def incremental_generator(self, df_name_path, list_micro_clusters_incremental, save_dataframe=False, filename='concept_df_incremental.csv'):
        '''
            Incremental drift must start from the dataset that has already been created with the concepts and dictionaries, since the work has already been done and requires
            only that concept B be split. Currently, three incremental concepts are supported: A, B, and C.
            The dataset used here is therefore the ready-to-use `concept_cf` dataset, to which we add A, B, and C in the `concept` column instead of 0 and 1.
            The format of the input dataset will be:
            [dim1, dim2, dim3, ..., macro_clusters, micro_clusters, concept, anomaly]
            Parameters:
            - List of microclusters for the incremental model, [n_A, n_B, n_C]
            Currently, simply adding an n_X is sufficient to add a concept.
            n_x = [a, b] where a is the number of normal partitions and b is the number of anomalous partitions
        '''
        data_path = Path(f'{self.directory_stream}/{self.clustering_technique}')
        file_path = data_path / df_name_path
        df = pd.read_csv(file_path)
        
        with open(f"{self.directory_stream}/{self.clustering_technique}/pos_dict.pkl", "rb") as f:
            pos_dict = pickle.load(f)
        with open(f"{self.directory_stream}/{self.clustering_technique}/neg_dict.pkl", "rb") as f:
            neg_dict = pickle.load(f)
        
        clfclt = ClassifierClusters(dataframe=df, col_data=self.col_data, folder_save=f'{self.directory_stream}/{self.clustering_technique}')
        pos_mst = clfclt.extract_minimum_spanning_tree(dict_adj=pos_dict)
        neg_mst = clfclt.extract_minimum_spanning_tree(dict_adj=neg_dict)
        
        ret_dict = {}
        ret_dict[0] = {}  # macro cluster 0, benignant POS
        ret_dict[1] = {}  # macro cluster 1, anomalous NEG
        
        values_macro0 = df.loc[df['macro_clusters'] == 0, 'micro_clusters'].value_counts()
        values_macro1 = df.loc[df['macro_clusters'] == 1, 'micro_clusters'].value_counts()
        for i in range(len(values_macro0)):
            ret_dict[0][i] = values_macro0[i]
        for i in range(len(values_macro1)):
            ret_dict[1][i] = values_macro1[i]

        bfs_neg = self.bfs_order(neg_mst)  
        bfs_pos = self.bfs_order(pos_mst)  

        ##### creation of partition
        # n_A, n_B, n_C = [[3,3], [2,2], [1,1]]
        n_A, n_B, n_C = list_micro_clusters_incremental
        concept_labels = ['A', 'B', 'C']
        concept_splits = [n_A, n_B, n_C]

        concept_partitions = {}
        idx_neg, idx_pos = 0, 0  

        for label, (x_pos, x_neg) in zip(concept_labels, concept_splits):
            nodes_neg = bfs_neg[idx_neg : idx_neg + x_neg]
            nodes_pos = bfs_pos[idx_pos : idx_pos + x_pos]
            idx_neg += x_neg
            idx_pos += x_pos

            concept_partitions[label] = {
                'pos': {node: ret_dict[0][node] for node in nodes_pos},  # macro=0 → pos (benignant)
                'neg': {node: ret_dict[1][node] for node in nodes_neg}   # macro=1 → neg (anomalous)
            }
        df['concept'] = None  # o potresti inizializzarla a NaN

        for label, parts in concept_partitions.items():
            # nodes neg (macro=1)
            for node, n_samples in parts['neg'].items():
                mask = (df['macro_clusters'] == 1) & (df['micro_clusters'] == node)
                idx = df[mask].index[:n_samples]  
                df.loc[idx, 'concept'] = label

            # nodes pos (macro=0)
            for node, n_samples in parts['pos'].items():
                mask = (df['macro_clusters'] == 0) & (df['micro_clusters'] == node)
                idx = df[mask].index[:n_samples]
                df.loc[idx, 'concept'] = label
        
        if save_dataframe:
            self.save_df(df=df, filedf=filename)
            print("Dataframe with incremental concepts A,B,C saved")
            partitions_path = data_path / 'concept_partitions_incremental.json'
            partitions_serializable = {
                label: {
                    'neg': {str(k): int(v) for k, v in parts['neg'].items()},
                    'pos': {str(k): int(v) for k, v in parts['pos'].items()}
                }
                for label, parts in concept_partitions.items()
            }
            with open(partitions_path, 'w') as f:
                json.dump(partitions_serializable, f, indent=4)
            print("Concept partitions saved in concept_partitions_incremental.json")

    def pipeline(self, n_macro_clusters, list_micro_clusters, n_samples, perc_neg, list_samples=[], plotting_step=False, save_plots=False, save_dataframe=False, filename='concept_df.csv', need_preprocess=True, preprocess_df=True):
        '''
        Concept creation pipeline. Expected dataframe format: [dim1, dim2, dim3, ..., dimk].
        If no preprocessing is required, then the expected dataframe format is: [dim1, dim2, dim3, ..., dimk, macro_clusters]. Therefore, disable the `need_process` flag.
        If, on the other hand, the dataframe passed as an argument has already been preprocessed by various algorithms, then the expected format must be:
        [dim1, dim2, dim3, ..., dimk, macro_clusters, micro_clusters]; therefore, disable the `preprocessed_dataframe` flag.
        Parameters:
        - n_macro_clusters: Specify this value if you are dealing with binary classification (benign/malignant)
        - list_micro_clusters: List of the number of microclusters for each macrocluster. You must also specify whether the passed dataframe has already been processed
        - n_samples, perc_neg, and list_samples are all parameters for the `create_concept` function; they have already been specified and detailed
        - plotting_step: allows you to plot all steps, which is useful if the dataset is two-dimensional. Otherwise, provide an additional method for dimensionality reduction
            Disable if the dataset is not two-dimensional
        - save_plots and save_dataframe save the intermediate results obtained from processing
        - filename: final name of the output dataframe
        - need_preprocess is used to perform micro-macro clustering; if disabled, it performs only micro clustering. However,
            it requires that the user already pass a DataFrame with the `macro_clusters` label.
        - preprocess_df is used to perform micro-macro clustering or not; if disabled, it simply creates a copy of the DataFrame passed as a parameter.
            It also requires that the user pass a dataframe with the labels `macro_clusters` and `micro_clusters`.
        Important: If the `need_preprocess` and `process_df` flags are disabled, checks must still be included regarding `list_samples`
        `list_micro_clusters` and `n_macro_clusters` ---> these are important for automated internal operations
        '''
        print("Start pipeline")
        ################ preprocessing step
        if preprocess_df:
            print("Step 1: preprocessing")
            if need_preprocess:
                print("Step 1a: datatframe needs only preprocessing micro-clustering")
            if self.clustering_technique == 'supervised':
                new_df = self.df.copy(deep=True) ### dataset must have a column named "micro_clusters"
            else:
                new_df, centers_ = self.micro_macro_clustering(self.df.copy(deep=True), n_macro_clusters=n_macro_clusters, list_micro_clusters=list_micro_clusters, need_preprocess=need_preprocess)
        else:
            print("Step 1b: dataframe doesn't need preprocessing")
            new_df = self.df.copy(deep=True)
        # format dataframe: [dim1, dim2, macro, micro]
        
        if save_dataframe and preprocess_df:
            self.save_df(df=new_df, filedf=f'prepared_df_{self.clustering_technique}.csv')
        
        ############## concept creation 
        pos_mst, neg_mst = self.classifier_clusters(dataframe=new_df, plotting=plotting_step)
        print(f'MST benignant: {pos_mst}')
        print(f'MST maliicous: {neg_mst}')
        print("Step 5: concept creation")
        concept_df = self.create_concept(dataframe=new_df.copy(deep=True), pos_mst=pos_mst, neg_mst=neg_mst, n_samples=n_samples, perc_neg=perc_neg, list_samples=list_samples, plotting=plotting_step)
        # format of dataframe: [dim1, dim2, dimk, macro, micro, concept]
        # print(concept_df['concept'].value_counts())
        if save_dataframe:
            self.save_df(df=concept_df, filedf=f'concept_df_noabn_{self.clustering_technique}.csv')
        
        ################# anomaly scorer
        print("Step 6: classification of external points to concept as outliers scores")
        abn_df = self.apply_abnomaly_scorer(dataframe=concept_df, plot_step=plotting_step) # format [dim1, dim2, macro, micro, concept, anomaly]
        print("Anomalies: ", abn_df.loc[abn_df['concept'] == 0, 'anomaly'].value_counts())

        if save_dataframe:
            self.save_df(df=abn_df, filedf=filename)

    def pipeline_only_generation(self, n_samples, perc_neg, list_samples=[], plotting_step=False, save_plots=False, save_dataframe=False, filename='concept_df_2.csv', need_preprocess=True, preprocess_df=True, loading=True):
        '''
        only concept generation
        '''
        print("Starting pipeline ONLY for generation and clusterization")
        new_df = pd.read_csv(f'{self.directory_stream}/{self.clustering_technique}/prepared_df_{self.clustering_technique}.csv')
        
        ############## concept creation
        pos_mst, neg_mst = self.classifier_clusters(dataframe=new_df, plotting=plotting_step, loading=loading)
        print(f'MST benignant: {pos_mst}')
        print(f'MST malicious: {neg_mst}')
        print("Step 5: concept creation")
        concept_df = self.create_concept(dataframe=new_df.copy(deep=True), pos_mst=pos_mst, neg_mst=neg_mst, n_samples=n_samples, perc_neg=perc_neg, list_samples=list_samples, plotting=plotting_step)
        # dataframe: [dim1, dim2, dimk, macro, micro, concept]
        if save_dataframe:
            self.save_df(df=concept_df, filedf=f'concept_df_noabn_{self.clustering_technique}_2.csv')
        
        ################# anomaly scorer
        print("Step 6: classification of external points to concept as outliers scores")
        abn_df = self.apply_abnomaly_scorer(dataframe=concept_df, plot_step=plotting_step) # format [dim1, dim2, macro, micro, concept, anomaly]
        print("Anomalies: ", abn_df.loc[abn_df['concept'] == 0, 'anomaly'].value_counts())

        if save_dataframe:
            self.save_df(df=abn_df, filedf=filename)

    def pipeline_one2many(self, n_samples, perc_neg, list_samples=[], plotting_step=False, save_dataframe=False, filename='concept_df.csv'):
        '''
        Pipeline for creating a concept using a one-to-many approach. Expected dataframe format: [dim1, dim2, dim3, ..., dimk].
        Compared to the traditional pipeline, the preprocessing step involving the clustering algorithm is assumed to have already been completed, so that common data is available.

        [dim1, dim2, dim3, ..., dimk, macro_clusters, micro_clusters] (prepared_df.csv)
        Parameters:
        - n_samples, perc_neg, list_samples are all inputs for the `create_concept` function; already specified and detailed
        - plotting_step: allows plotting of all steps, useful if the dataset is two-dimensional. Otherwise, provide an additional method for dimensionality reduction
            Disable if the dataset is not two-dimensional
        - filename: final name of the output dataframe
        '''
        print("Start of the pipeline for one2many")
        new_df = pd.read_csv(f'{self.directory_stream}/{self.clustering_technique}/prepared_df_{self.clustering_technique}.csv')
        
        ############## concept creation
        pos_mst, neg_mst = self.classifier_clusters(dataframe=new_df, plotting=plotting_step)
        print(f'MST benignant: {pos_mst}')
        print(f'MST malicious: {neg_mst}')
        print("Step 5: concept creation")
        concept_df = self.create_concept(dataframe=new_df.copy(deep=True), pos_mst=pos_mst, neg_mst=neg_mst, n_samples=n_samples, perc_neg=perc_neg, list_samples=list_samples, plotting=plotting_step)
        # dataframe: [dim1, dim2, dimk, macro, micro, concept]
        # print(concept_df['concept'].value_counts())
        if save_dataframe:
            self.save_df(df=concept_df, filedf=f'concept_df_noabn_{self.clustering_technique}.csv')
        
        ################# anomaly scorer
        print("Step 6: classification of external points to concept as outliers scores")
        abn_df = self.apply_abnomaly_scorer(dataframe=concept_df, plot_step=plotting_step) # format [dim1, dim2, macro, micro, concept, anomaly]
        print("Anomalies: ", abn_df.loc[abn_df['concept'] == 0, 'anomaly'].value_counts())

        if save_dataframe:
            self.save_df(df=abn_df, filedf=filename)
            print("Dataframe concept saved with anomalies")

    def save_df(self, df, filedf):
        if self.mode_classifier != 'o2m':
            data_path = Path(f'{self.directory_stream}/{self.clustering_technique}')
        else:
            data_path = Path(f'{self.directory_stream}/{self.clustering_technique}/o2m')
        data_path.mkdir(parents=True, exist_ok=True)
        file_path = data_path / filedf
        df.to_csv(file_path, index=False)
        print(f"File salvato in: {file_path}")


