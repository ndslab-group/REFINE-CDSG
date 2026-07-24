from datetime import date
import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from UnionFind import UnionFind


class ClassifierClustersO2M:
    def __init__(self, dataframe, col_data, folder_save):
        self.dataframe = dataframe
        print(self.dataframe.head())
        self.col_data = col_data
        self.folder_save = folder_save
        os.makedirs(self.folder_save, exist_ok=True)

    # ------------------------------------------------------------------
    # training an isolation forest for each micro cluster
    # ------------------------------------------------------------------

    def train_isolation_forests(self, df, contamination='auto', n_estimators=100):
        """
        Train a separate Isolation Forest for each micro-cluster present in df.
        Parameters
        ----------
        df            : DataFrame with the data columns + 'macro_clusters' + 'micro_clusters'
        contamination : expected fraction of anomalies (default 0.1)
        n_estimators  : number of trees for each IF

        Returns
        -------
        models : dict  { micro_cluster_label -> trained IsolationForest }
        """
        micro_labels = sorted(df['micro_clusters'].unique())
        models = {}
        for label in micro_labels:
            cluster_data = df[df['micro_clusters'] == label]
            X = cluster_data.iloc[:, :self.col_data].to_numpy()
            print(f'Micro-cluster {label}: {len(X)} campioni')
            iso = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=17,
                n_jobs=10
            )
            iso.fit(X)
            models[label] = iso
        return models

    def classifiers_macro_clusters(self, dataframe, contamination='auto'):
        """
        For each macro cluster an ensemble of isolation forest is trained, one for each micro-cluster
       
        Returns
        -------
        all_models : list of dict
            [ models_macro0 , models_macro1 ]
            where models_macroX = { micro_label -> IsolationForest }
        """
        pos_df = dataframe[dataframe['macro_clusters'] == 0]
        neg_df = dataframe[dataframe['macro_clusters'] == 1]
        print(f'Size benignant: {len(pos_df)}\nSize malicious: {len(neg_df)}')

        all_models = []
        labels = ['benevoli', 'malevoli']
        for i, df in enumerate([pos_df, neg_df]):
            print(f'\n--- Training Isolation Forests [{labels[i]}] ---')
            models = self.train_isolation_forests(df, contamination=contamination)
            all_models.append(models)
            self._write_anomaly_report(df, models, label=labels[i])
        return all_models

    def _write_anomaly_report(self, df, models, label):
        """
        Scrive su file le anomaly-score medie di ogni micro-cluster
        valutate sul proprio Isolation Forest e su quelli degli altri cluster.
        """
        file_name = f'{self.folder_save}/report_clusters_{label}.txt'
        micro_labels = sorted(models.keys())
        lines = [f'=== Report anomaly scores [{label}] ===\n']
        for ml in micro_labels:
            cluster_data = df[df['micro_clusters'] == ml]
            X = cluster_data.iloc[:, :self.col_data].to_numpy()
            score_own = -models[ml].score_samples(X).mean()   # score anomalia medio (positivo = più anomalo)
            lines.append(f'Micro-cluster {ml}: anomaly_score_medio (proprio IF) = {score_own:.4f}')
        with open(file_name, 'w') as f:
            f.write('\n'.join(lines))
        print(f'Report salvato in {file_name}')

    def predict_anomaly_score(self, model, data):
        """
        Ritorna lo score di anomalia (valori più alti → più anomalo).
        Usiamo -score_samples per avere la stessa convenzione di sklearn.
        """
        return -model.score_samples(data)

    def predict_clf(self, model, data):
        """
        Predizione binaria: +1 (normale) / -1 (anomalia).
        Mappato a 0/1 per coerenza con il resto del codice.
        """
        raw = model.predict(data)          # +1 inlier, -1 outlier
        return np.where(raw == 1, 0, 1)    # 0 = normale, 1 = anomalia


    def compute_similarity_matrix(self, X, y, models):
        """
        Calcola la matrice di similarità tra micro-cluster usando gli anomaly scores.

        La cella [i, j] rappresenta quanto i punti del cluster i sono considerati
        "normali" (basso score di anomalia) dall'Isolation Forest del cluster j,
        normalizzato per il valore medio sul proprio modello.

        Parametri
        ---------
        X      : feature array
        y      : etichette micro-cluster (intere, 0-based)
        models : dict { micro_label -> IsolationForest }

        Ritorna
        -------
        similarity_matrix : np.ndarray (n_clusters x n_clusters)
        """
        micro_labels = sorted(models.keys())
        n_clusters = len(micro_labels)
        label_to_idx = {lbl: idx for idx, lbl in enumerate(micro_labels)}

        # score di anomalia dei punti del cluster i sul proprio IF (prior)
        own_scores = {}
        for lbl in micro_labels:
            mask = (y == lbl)
            own_scores[lbl] = self.predict_anomaly_score(models[lbl], X[mask]).mean()

        similarity_matrix = np.zeros((n_clusters, n_clusters))
        for lbl_i in micro_labels:
            i = label_to_idx[lbl_i]
            mask_i = (y == lbl_i)
            prior_i = own_scores[lbl_i]
            for lbl_j in micro_labels:
                j = label_to_idx[lbl_j]
                # score medio dei punti di i visti dall'IF di j
                score_ij = self.predict_anomaly_score(models[lbl_j], X[mask_i]).mean()
                # similarità: quanto i punti di i "sembrano normali" per j
                # valori bassi di score_ij rispetto a prior_i → alta similarità
                similarity_matrix[i, j] = prior_i / (score_ij + 1e-9)

        return similarity_matrix

    def plot_decision_boundary(self, models, grid_points, xx, yy):
        """
        Disegna i confini di decisione usando gli anomaly scores aggregati
        degli Isolation Forests.
        """
        micro_labels = sorted(models.keys())
        # per ogni punto della griglia calcola lo score per ogni micro-cluster
        all_scores = np.column_stack([
            self.predict_anomaly_score(models[lbl], grid_points)
            for lbl in micro_labels
        ])
        # assegna ogni punto al cluster con il minor score di anomalia (= più simile)
        Z = np.argmin(all_scores, axis=1).reshape(xx.shape)
        plt.contour(xx, yy, Z,
                    levels=np.arange(len(micro_labels) + 1) - 0.5,
                    colors='k', linestyles='--', linewidths=1)

    def plot_decision_data(self, df, models, save=False,
                           name_file='grafico',
                           title='Classificazione microclusters e Confini di Decisione'):
        """
        Plot dei confini di decisione per dati 2D con Isolation Forests.
        Formato atteso del dataframe: [dim1, dim2, macro_clusters, micro_clusters]

        Parametri
        ---------
        models : dict { micro_label -> IsolationForest }
        """
        X = df.iloc[:, :self.col_data].to_numpy()
        y = df['micro_clusters'].to_numpy()
        micro_labels = sorted(models.keys())

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        n_points = len(df)

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, n_points),
            np.linspace(y_min, y_max, n_points)
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        print(f'Griglia di punti: {grid_points.shape}')

        # heatmap: per ogni cluster mostra lo score di anomalia normalizzato
        all_scores = np.column_stack([
            self.predict_anomaly_score(models[lbl], grid_points)
            for lbl in micro_labels
        ])
        # normalizza 0-1 e inverti (così zone "normali" appaiono più intense)
        max_s = all_scores.max(axis=0, keepdims=True) + 1e-9
        probs_grid = 1.0 - (all_scores / max_s)

        for i in range(len(micro_labels)):
            plt.contourf(xx, yy, probs_grid[:, i].reshape(xx.shape),
                         alpha=0.3, cmap='coolwarm')

        labels = df['micro_clusters']
        unique_labels = labels.unique()
        colormap = plt.cm.get_cmap('tab10', len(unique_labels))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, s=50,
                              cmap=colormap, edgecolor='k', label='Dati originali')
        plt.colorbar(scatter, label='Cluster', ticks=range(len(unique_labels)))

        self.plot_decision_boundary(models, grid_points, xx, yy)

        plt.title(title)
        plt.xlabel('Dimensione 1')
        plt.ylabel('Dimensione 2')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        if save:
            current_datetime = date.today().strftime('%Y-%m-%d')
            cartella = f'results_{current_datetime}'
            if not os.path.exists(cartella):
                os.makedirs(cartella)
            file_path = os.path.join(cartella, f'{name_file}.png')
            plt.savefig(file_path)
            print(f'Grafico salvato in: {file_path}')
        plt.show()

    def train_test_loop(self, contamination='auto'):
        """
        Addestra gli Isolation Forests e (opzionalmente) genera i plot.
        Formato atteso: [dim1, dim2, ..., dimk, macro_clusters, micro_clusters]
        """
        df = self.dataframe.copy(deep=True)
        all_models = self.classifiers_macro_clusters(dataframe=df, contamination=contamination)
        # if plots:
        #     macro_names = ['benevoli', 'malevoli']
        #     for i in range(2):
        #         part_df = self.dataframe[self.dataframe['macro_clusters'] == i]
        #         name_file = f'clf_microclusters_{macro_names[i]}'
        #         self.plot_decision_data(df=part_df, models=all_models[i], save=True, name_file=name_file)

        return all_models

    def compute_graphs(self, dataframe, all_models):
        """
        Calcola le matrici di similarità e i dizionari di adiacenza per
        macro-cluster benevolo e malevolo.

        Parametri
        ---------
        all_models : list  [ models_macro0, models_macro1 ]
                     dove models_macroX = { micro_label -> IsolationForest }
        """
        results = {}
        names = ['Benigni', 'Maligni']
        for i, name in enumerate(names):
            print(f'\n{name}------------------')
            sub_df = dataframe[dataframe['macro_clusters'] == i]
            models = all_models[i]
            X = sub_df.iloc[:, :self.col_data].to_numpy()
            y = sub_df['micro_clusters'].to_numpy()
            sim_matrix = self.compute_similarity_matrix(X=X, y=y, models=models)
            sim_df = pd.DataFrame(sim_matrix)
            print(sim_df)
            adj_dict = self.extract_k_sim_neighbours(sim_df)
            print(adj_dict)
            results[i] = adj_dict

        return results[0], results[1]

    def extract_k_sim_neighbours(self, matrix_df, k=None):
        """
        Estrae per ogni cluster i k vicini più simili, normalizzando
        per il massimo globale (norm min-max).
        """
        neigh_cluster = {}
        for row in range(matrix_df.shape[0]):
            matrix_df.iat[row, row] = 0
        max_value_global = matrix_df.values.max()
        for row in range(matrix_df.shape[0]):
            vals = {}
            for col in range(matrix_df.shape[1]):
                if row != col:
                    vals[col] = matrix_df.iat[row, col] / max_value_global
            vals_ordered = dict(sorted(vals.items(), key=lambda item: item[1], reverse=True))
            neigh_cluster[row] = vals_ordered
        return neigh_cluster

    def plot_oriented_graph(self, dict_adj):
        G = nx.DiGraph()
        for node, neighbors in dict_adj.items():
            G.add_node(node)
            for node_neigh, weight in neighbors.items():
                G.add_edge(node, node_neigh, weight=weight)
        print('Archi del grafo orientato con i pesi:')
        for u, v, weight in G.edges(data='weight'):
            print(f'{u} -> {v} (weight: {weight})')
        plt.figure(figsize=(8, 6))
        nx.draw(G, with_labels=True, node_color='skyblue',
                node_size=700, font_size=15, font_weight='bold')
        plt.show()

    def create_oriented_graph(self, dict_adj):
        G = nx.DiGraph()
        for node, neighbors in dict_adj.items():
            G.add_node(node)
            for node_neigh, weight in neighbors.items():
                G.add_edge(node, node_neigh, weight=weight)
        return G

    def minimum_dict_wrapped(self, dict_adj):
        """
        Trasforma il dizionario di similarità in uno di distanze
        (archi più piccoli = nodi più simili) per Kruskal.
        """
        max_w = max(
            weight
            for neighbors in dict_adj.values()
            for weight in neighbors.values()
        )
        new_dict = {
            node: {neigh: max_w - w for neigh, w in neighbors.items()}
            for node, neighbors in dict_adj.items()
        }
        return new_dict

    def kruskal_mst(self, graph):
        edges = sorted(graph.edges(data=True), key=lambda edge: edge[2]['weight'])
        mst = nx.Graph()
        uf = UnionFind(len(graph.nodes))
        for u, v, data in edges:
            if uf.find(u) != uf.find(v):
                mst.add_edge(u, v, weight=data['weight'])
                uf.union(u, v)
            if len(mst.edges) == len(graph.nodes) - 1:
                break
        mst_dict = self.mst_to_dict(mst)
        return mst, mst_dict

    def mst_to_dict(self, mst):
        mst_dict = {}
        print(f'Archi: {mst.edges()}')
        for u, v, weight in mst.edges(data='weight'):
            mst_dict.setdefault(u, []).append((v, weight))
            mst_dict.setdefault(v, []).append((u, weight))
        return mst_dict

    def extract_minimum_spanning_tree(self, dict_adj):
        min_dict = self.minimum_dict_wrapped(dict_adj)
        min_graph = self.create_oriented_graph(min_dict)
        _, dict_tree = self.kruskal_mst(min_graph)
        return dict_tree