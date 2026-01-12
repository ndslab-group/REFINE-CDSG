from datetime import date
import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from utils import Utility
from ClassifierClusters import ClassifierClusters
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import csv
from pathlib import Path

class ConceptGenerator:
    def __init__(self, df_file, col_data, col_names):
        self.df = pd.read_csv(df_file)
        # self.df = self.df.drop(columns=[""]) ## remove column if needed (es: flow_id, timestamp, etc) from SOURCE dataset
        self.df['label'] = self.df['label'].map({'Benign': 0, 'Attack': 1})
        self.df = self.df.rename(columns={'label': 'macro_clusters'})
        # Rimuovi le righe con NaN
        # Sostituisci gli infiniti con NaN
        # self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # self.df = self.df.dropna()
        self.col_data = col_data
        self.utility = Utility(col_data=col_data, col_names=col_names)

    def get_df(self):
        return self.df

    def print_df(self, df, n_row=None):
        '''
        Permette di stampare un dataframe specifico con uno specifico numero di righe
        '''
        if n_row:
            print(df.head(n_row))
        else:
            print(df.head())

    def macro_clustering(self, X, n_clusters=2):
        clusters = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
        return clusters.labels_

    def micro_clustering(self, X, n_clusters):
        clusters = KMeans(n_clusters=n_clusters, random_state=27, n_init="auto").fit(X)
        # clusters = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
        return (clusters.cluster_centers_, clusters.labels_)


    def micro_macro_clustering(self, df, n_macro_clusters=2, list_micro_clusters=[], need_preprocess=True):
        '''
        Step 1: preprocessing dei dati per la creazione di macro micro clusters
        Funzione per la creazione di una suddivisione in macro-micro clusters
        - n_macro_clusters divide il dataframe passato in input in macro_clusters
        - list_micro_clusters contiene due elementi indicanti rispettivamente il numero di clusters per
        ogni macro cluster
        '''
        flow_ids = df["flow_id"].copy()
        if len(list_micro_clusters) != n_macro_clusters:
            raise ValueError("Errore numero invalido! Bisogna specificare nella lista i micro clusters per macro cluster\nLa lista deve avere dimensione pari al numero di macroclusters")
        
        new_df = df.copy(deep=True)
        if not need_preprocess:
            # se non c'è bisogno l'etichetta macro_cluster è già presente nel dataframe
            X = df.drop(columns=["flow_id"]) if "flow_id" in df.columns else df
            macro_clusters = self.macro_clustering(X, n_clusters=n_macro_clusters)
            # creazione di due nuove features
            new_df['macro_clusters'] = macro_clusters # definizione dei macro clusters

        new_df['micro_clusters'] = -1
        
        # centroidi per microclusters, per future works
        centers_micro_clusters = []
        for i in range(n_macro_clusters):
            macro_df = new_df[new_df['macro_clusters'] == i]
            X_micro = macro_df.drop(columns=["flow_id"]) if "flow_id" in macro_df.columns else macro_df
            centers, labels_micro = self.micro_clustering(X_micro, n_clusters=list_micro_clusters[i])
            centers_micro_clusters.append(centers)
            new_df.loc[new_df['macro_clusters'] == i, 'micro_clusters'] = labels_micro
        
        # Riaggiungiamo flow_id se esiste
        if "flow_id" in df.columns:
            new_df["flow_id"] = flow_ids.values
        
        return (new_df, centers_micro_clusters)
    

    def classifier_clusters(self, dataframe, plotting=True):
        '''
        Questa funzione permette di associare una metrica di similarità tra clusters.
        Richiede in input:
        - dataframe con suddivisione in macro clusters
        Restituisce:
        - partizione in micro clusters
        - matrice di similarità
        Internamente richiama una classe per poter addestrare un classificatore (che può differire).
        Il classificatore deve avere due metodi importanti:
        - fit
        - predict_proba
        vengono messi in esterno due metodi:
        - train classifier
        - test classifier (che restituisce accuracy, f1 score, classification report)
        Inoltre, dati i campioni permette di calcolare la matrice di similarità.
        Il metodo difatti restituisce la matrice di similarità trovata e il dizionario di cluster vicini
        '''
        flow_ids = dataframe["flow_id"].copy()
        X = dataframe.drop(columns=["flow_id"]) if "flow_id" in dataframe.columns else dataframe

        # addestro il classificatore per i macro-clusters
        clfclt = ClassifierClusters(dataframe=X, col_data=self.col_data)
        
        # classificatori addestrati su campioni benevoli e malevoli
        print("Step 2: addestramento classificatore")
        models = clfclt.train_test_loop(plots=plotting)
        
        # dizionari dalla matrice di adiacenza
        print("Step 3: matrice di similarità")
        pos_dict, neg_dict = clfclt.compute_graphs(dataframe=X, clfs=models)

        # estrazione dei dizionari dei minimum spanning tree applicando Kruscal
        print("Step 4: albero minimo ricoprente")
        pos_mst = clfclt.extract_minimum_spanning_tree(dict_adj=pos_dict)
        neg_mst = clfclt.extract_minimum_spanning_tree(dict_adj=neg_dict)
        return pos_mst, neg_mst

    def extract_samples_dict(self, dataframe):
        '''
        estrae per un dato dataframe quanti campioni sono presenti per ogni macro e micro clusters nel formato:
        { macro: {micro: x_samples} }
        '''
        ret_dict = {}
        ret_dict[0] = {} # macro cluster 0
        ret_dict[1] = {} # macro cluster 1
        
        values_macro0 = dataframe.loc[dataframe['macro_clusters'] == 0, 'micro_clusters'].value_counts()
        values_macro1 = dataframe.loc[dataframe['macro_clusters'] == 1, 'micro_clusters'].value_counts()
        
        print('Values macro benevoli ', values_macro0)
        print('Values macro malevoli ', values_macro1)
        
        with open("macro_values.txt", "w") as f:
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
        restituisce i nodi da dove andare a prelevare tramite visita in ampiezza della'lbero mst dato in input
        Se samples dict è -1, allora prende intere partizioni IN ORDINE,
        Se samples dict è diverso da -1 allora campiona il numero di campioni scelto, e finché non campiona esattamente
        un numero di campioni pari a quello selezionato va a prendere intere o partizioni parziali
        '''
        list_nodes = []
        
        # aggiungo i nodi andando ad esplorare l'albero appena creato
        # partendo dall'assunzione che i nodi verranno inseriti secondo un determinato ordine
        for node, sons in mst_dict.items():
            if node not in list_nodes:
                list_nodes.append(node)
                while sons:
                    son = sons.pop(0)
                    if son[0] not in list_nodes:
                        list_nodes.append(son[0])
       
        # creazione delle partizioni sulla base dei campioni n_samples da estrarre
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
        questa funzione crea un concetto. pos_mst e neg_mst sono gli alberi minimi ricoprenti.
        se vogliamo un numero fissato di campioni nel concetto allora specifichiamo n_samples e perc_neg,
        tuttavia questo potrebbe portare ad avere dataset non pieni, quindi maggiore sparsità di campioni: utile per ingannare un classificatore.
        Se invece n_samples è -1 allora usiamo list_samples che specifica il numero di cluster da utilizzare per positivi e negativi.
        Dunque se è -1 il valore viene sommato automaticamente!
        List samples è composto da due elementi: n_cluster_positivi, n_cluster_negativi, tenendo conto che questi sono già ordinati
        Ritorna un dataframe nel formato:
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
        
        with open("elaborazione_concetto.txt", "w", encoding="utf-8") as f:
            f.write(f"Partizione benevoli: {samples_pos}\n")
            f.write(f"Partizione malevoli: {samples_neg}\n")
            f.write(f"Dizionario dei campioni: {samples_dict}\n")

        df_copy = dataframe.copy(deep=True)
        df_copy['concept'] = 0 # creazione della nuova colonna del concetto
        if plotting:
            self.utility.plot_dataset(dataset=df_copy, feature='concept', title='Concept nel dataframe (dovrebbero essere tutti 0)')
        pos_df = df_copy[df_copy['macro_clusters'] == 0].copy()
        neg_df = df_copy[df_copy['macro_clusters'] == 1].copy()
        
        # aggiungo i campioni positivi
        for cluster, n_samples_cluster in samples_pos.items():
            print(f'Cluster {cluster}, samples {n_samples_cluster}')
            samples_indices_pos = np.random.choice(pos_df[pos_df['micro_clusters'] == cluster].index, size=n_samples_cluster, replace=False)
            pos_df.loc[samples_indices_pos, 'concept'] = 1
        
        if plotting:
            self.utility.plot_dataset(dataset=pos_df, feature='concept', title='Concept nei campioni benevoli', save=True, name_file='concept_benevoli_test')

        # aggiungo i campioni negativi
        for cluster, n_samples_cluster in samples_neg.items():
            print(f'Cluster {cluster}, samples {n_samples_cluster}')
            samples_indices_neg = np.random.choice(neg_df[neg_df['micro_clusters'] == cluster].index, size=n_samples_cluster, replace=False)
            neg_df.loc[samples_indices_neg, 'concept'] = 1

        if plotting:
            self.utility.plot_dataset(dataset=neg_df, feature='concept', title='Concept nei campioni malevoli', save=True, name_file='concept_malevoli_test')
        
        unique_df = pd.concat([pos_df, neg_df], axis=0, ignore_index=True)
        print(f"Dimensione attesa: {len(pos_df) + len(neg_df)}, dimensione reale: {len(unique_df)}")
        if plotting:
            self.utility.plot_dataset(dataset=unique_df, feature='concept', title='Unique dataframe after concat')
        return unique_df
    
    def abnomaly_scorer(self, dataframe):
        '''
        Addestra un anomaly scorer da applicare ai miei "anomaly"
        '''
        # addestro sul clusters
        inliers = dataframe[dataframe['concept'] == 1]
        outliers = dataframe[dataframe['concept'] == 0]
        X_train = inliers.iloc[:,:self.col_data].to_numpy()
        X_anomaly = outliers.iloc[:,:self.col_data].to_numpy()

        clf = IsolationForest(n_estimators=50, warm_start=True)
        clf.fit(X_train)
        # calcolo gli score di anomalia con i samples
        scores = -clf.score_samples(X_anomaly)
        return scores
    
    def fit_scorer(self, dataframe):
        # addestro sul clusters
        inliers = dataframe[dataframe['concept'] == 1]
        X_train = inliers.iloc[:,:self.col_data].to_numpy()
        # serve per la determinazione del classificatore per le anomalie, in questo caso usato IsolationForest
        clf = IsolationForest(n_estimators=50, warm_start=True)
        clf.fit(X_train)
        return clf
    
    def apply_abnomaly_scorer(self, dataframe, plot_step=True):
        # esclusione della colonna dei flow_id dalle feature usate per il calcolo
        abn_df = dataframe.copy()
        flow_ids = dataframe["flow_id"].copy()
        abn_df = abn_df.drop(columns=["flow_id"]) if "flow_id" in dataframe.columns else dataframe
        
        # da usare con Isolation Forest o qualsiasi altro modello di anomalia
        clf = self.fit_scorer(dataframe=abn_df)
        outlier_scores = clf.decision_function(abn_df[abn_df['concept']==0].iloc[:,:self.col_data].to_numpy())

        scaler = MinMaxScaler(feature_range=(0, 2))
        scaled_scores = scaler.fit_transform(outlier_scores.reshape(-1, 1)).flatten()

        # opzionale, serve per inserire in maniera meglio una classificazione con soglia
        truncated_scores = np.trunc(scaled_scores * 10) / 10
        truncated_scores = 2.0 - truncated_scores

        abn_df['anomaly'] = 0.0
        abn_df.loc[abn_df['concept'] == 0, 'anomaly'] = truncated_scores
        X_anomaly = abn_df[abn_df['concept'] == 0].iloc[:,:self.col_data].to_numpy()
        y_anomaly = abn_df[abn_df['concept'] == 0].loc[:,'anomaly'].to_numpy()
        if plot_step:
            self.utility.plot_contours_outlier(X_anomaly, y_anomaly, abn_df[abn_df['concept'] == 0], clf)
        
        # riaggiungo i flow_id nel caso dei dataset per permettere il riconoscimnento successivo dei record
        abn_df.insert(0, 'flow_id', flow_ids.values)
        return abn_df

    def pipeline(self, n_macro_clusters, list_micro_clusters, n_samples, perc_neg, list_samples=[], plotting_step=False, save_plots=False, save_dataframe=False, filename='concept_df.csv', need_preprocess=True, preprocess_df=True):
        '''
        Pipeline di creazione di un concept. Formato del dataframe atteso: [dim1, dim2, dim3, ..., dimk].
        Se non è necessario preprocessing allora il formato del dataframe atteso è: [dim1, dim2, dim3, ..., dimk, macro_clusters]. Disabilitare quindi il need_process flag
        Se il dataframe passato come argomento invece è già preprocessato da algoritmi diversi allora il formato atteso deve essere:
        [dim1, dim2, dim3, ..., dimk, macro_clusters, micro_clusters]; disabilitare quindi il flag preprocessed_dataframe.
        Parametri:
        - n_macro_clusters: da specificare se stiamo trattando di classificazioen binaria benevola/malevola
        - list_micro_clusters: lista di # di micro cluster per ogni macro cluster. Va specificata anche se il dataframe passato è già processato
        - n_sampels, perc_neg, list_samples sono tutte informazioni per il create_concept; già specificate e dettagliate
        - plotting_step: permette di plottare tutti gli step, utile se il dataset è bidimensionale. Prevedere altrimenti un metodo aggiuntivo per fare riduzione dimensionalità
            Disabilitare se il dataset non è bidimensionale
        - save_plots e save_dataframe salvano i risultati intermedi ottenuti dalla elaborazione
        - filename: nome finale del dataframe in output
        - need_preprocess serve per effettuare clusterizzazione micro_macro, se disabilitato fa solo clusterizzazione micro. Prevede
            però che l'utente passi già un dataframe con la label macro_clusters
        - preprocess_df serve per effettuare clusterizzazione micro_macro o no, Se disabilitato fa solo una copia del dataframe passato come parametro.
            Prevede altresì che l'utente passi un dataframe con la label macro_clusters e micro_cluster.
        Importante: nel caso in cui i flag di need_preprocess e process_df siano disabilitate, vanno comunque inseriti controlli in merito alla list_samples
        list_micro_clusters e n_macro_clusters ---> sono importanti per operazioni interne automatizzate
        '''
        print("Avvio della pipeline")
        if plotting_step:
            self.utility.plot_entire_dataset(dataframe=self.df, save=save_plots)

        if preprocess_df:
            print("Step 1: preprocessing")
            if need_preprocess:
                print("Step 1a: il dataframe necessita solo di preprocessing micro cluster")
            new_df, centers_ = self.micro_macro_clustering(self.df.copy(deep=True), n_macro_clusters=n_macro_clusters, list_micro_clusters=list_micro_clusters, need_preprocess=need_preprocess)
        else:
            print("Step 1b: il dataframe non necessita preprocessing")
            new_df = self.df.copy(deep=True)
        # formato del dataframe: [dim1, dim2, macro, micro]
        
        if plotting_step:
            self.utility.plot_dataset(new_df, 'macro_clusters', title='Scatter plot di macro clusters', save=save_plots, name_file='macroclusters_partition')
            self.utility.plot_dataset(new_df[new_df['macro_clusters'] == 1], 'micro_clusters', title='Scatter plot di malevoli', save=save_plots, name_file='microclusters_malevoli')
            print('Valori malevoli: ', new_df.loc[new_df['macro_clusters'] == 1, 'micro_clusters'].value_counts())
            self.utility.plot_dataset(new_df[new_df['macro_clusters'] == 0], 'micro_clusters', title='Scatter plot di benevoli', save=save_plots, name_file='microclusters_benevoli')
            print('Valori benevoli: ', new_df.loc[new_df['macro_clusters'] == 0, 'micro_clusters'].value_counts())
        
        if save_dataframe:
            self.save_df(df=new_df, filedf='prepared_df.csv')
        
        pos_mst, neg_mst = self.classifier_clusters(dataframe=new_df, plotting=plotting_step)
        print(f'MST benevoli: {pos_mst}')
        print(f'MST malevoli: {neg_mst}')
        print("Step 5: creazione del concept")
        concept_df = self.create_concept(dataframe=new_df.copy(deep=True), pos_mst=pos_mst, neg_mst=neg_mst, n_samples=n_samples, perc_neg=perc_neg, list_samples=list_samples, plotting=plotting_step)
        # formato del dataframe: [dim1, dim2, dimk, macro, micro, concept]
        
        if plotting_step:
            self.utility.plot_dataset(concept_df, 'concept', title="Concept", save=save_plots, name_file="concept_df")
            self.utility.plot_dataset(concept_df[concept_df['concept'] == 1], 'macro_clusters', title="Concept - partizione malevoli/benevoli", save=save_plots, name_file="concept_df_partition")
        
        # print(concept_df['concept'].value_counts())
        if save_dataframe:
            # salvaggio del dataframe del concept senza anomalie
            self.save_df(df=concept_df, filedf='concept_df_noabn.csv')
        
        print("Step 6: classificazione dei punti esterni al concept come outliers scores")
        abn_df = self.apply_abnomaly_scorer(dataframe=concept_df, plot_step=plotting_step) # formato [dim1, dim2, macro, micro, concept, anomaly]
        print("Anomalie: ", abn_df.loc[abn_df['concept'] == 0, 'anomaly'].value_counts())
        
        if plotting_step:
            self.utility.plot_dataset(abn_df[abn_df['concept'] == 0], 'anomaly', title='Anomaly scores', save=save_plots, name_file='anomaly_samples')
        
        if save_dataframe:
            self.save_df(df=abn_df, filedf=filename)
            print("Dataframe concept salvato con anomalie")
        
    
    def save_df(self, df, filedf):
        '''
        da modificare in termini di percorsi assoluti per poter effettivamente
        avere un percorso assoluto
        '''
        base_path = Path.cwd()
        cartella = base_path / "results_cdsg_directory"
        cartella.mkdir(parents=True, exist_ok=True)
        file_path = cartella / filedf
        df.to_csv(file_path, index=False)
        print(f"File salvato in: {file_path}")


