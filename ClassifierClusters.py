from datetime import date
import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from UnionFind import UnionFind

class ClassifierClusters:
    def __init__(self, dataframe, col_data):
        self.dataframe = dataframe
        print(self.dataframe.head())
        self.col_data = col_data

    def classifiers_macro_clusters(self, dataframe):
        # per prima cosa separo i cluster benevoli da malevoli
        pos_df = dataframe[dataframe['macro_clusters'] == 0]
        neg_df = dataframe[dataframe['macro_clusters'] == 1]
        print(f'Size benevoli: {len(pos_df)}\nSize malevoli: {len(neg_df)}')
        models = []
        param_grid = {
        'n_estimators': [50, 75, 100],
        'max_depth': [None, 10, 5],
        'min_samples_split': [2, 5, 10],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'class_weight': ['balanced', 'balanced_subsample', None]
        }
        dfs = [pos_df, neg_df] # dataframe da processare
        file_name = 'report_clusters_benevoli'
        for df in dfs:
            X = df.iloc[:,:self.col_data].to_numpy()
            y = df.iloc[:,-1].to_numpy() # l'ultima colonna è micro clusters
            # train test splitting
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=23)
            print(f'Train: X={X_train.shape}, y={y_train.shape}')
            print(f'Test: X={X_test.shape}, y={y_test.shape}')
            # addestramernto del modello scelto
            model_gs = self.train_classifier(X_train=X_train, y_train=y_train, param_grid=param_grid)
            # vale solo se usiamo random forest con gridsearch, o comunque se usiamo gridsearch nella funzione train_classifier
            best_rf = model_gs.best_estimator_
            best_params = model_gs.best_params_
            models.append((best_rf, best_params))
            print("Parameters: ", best_params)
            self.test_classifier(clf=best_rf, X_test=X_test, y_test=y_test, file_name=file_name)
            file_name = 'report_clusters_malevoli'
        return models

    def train_classifier(self, X_train, y_train, param_grid):
        # la scelta del modello potrebbe essere reimplementata dallo sviluppatore
        print("Procedo con l'addestramento")
        rf = RandomForestClassifier(random_state = 17)
        classes = len(np.unique(y_train))
        gridsearch = GridSearchCV(estimator=rf, param_grid=param_grid, cv=classes, n_jobs=-1, scoring='balanced_accuracy', verbose=4)
        gridsearch.fit(X_train, y_train)
        return gridsearch

    def test_classifier(self, clf, X_test, y_test, file_name='report'):
        y_pred = self.predict_clf(model=clf, data=X_test)
        print(f'Y-pred: {y_pred.shape}')
        report = classification_report(y_test, y_pred)
        print(f"Classification error: \n {report}")
        with open(f'{file_name}.txt', 'w') as f:
            f.write(report)
        

    def predict_proba_clf(self, model, data):
        '''
        da re-implementare nel caso in cui model non supporti predict_proba
        '''
        return model.predict_proba(data)
    
    def predict_clf(self, model, data):
        '''
        da re-implementare nel caso in cui model non supporti predicts
        '''
        return model.predict(data)

    def plot_decision_boundary(self, clf, grid_points, xx, yy):
        '''
        questa funzione permette di plottare i confini di decisione del nostro classificatore
        sulla base dei punti dello spazio
        '''
        probs = self.predict_proba_clf(model=clf, data=grid_points) # calcolo delle probabilità
        print(f'Dimensione probabilità: {probs.shape}')
        # Assegna a ciascun punto la classe con la probabilità massima
        Z = np.argmax(probs, axis=1) 
        print(Z.shape) # in questo momento ho fatto reshaper di tutti i punti, quindi ho un vettore di dimensione xx*yy
        Z = Z.reshape(xx.shape) # riadatta per ottenere una matrice di dimensione xx*yy, ovvero della griglia di partenza
        print(Z.shape)
        # Traccia i contorni per i confini decisionali
        plt.contour(xx, yy, Z, levels=np.arange(6) - 0.5, colors='k', linestyles='--', linewidths=1)
        # traccia contorno sulla base dei punti presi dalla griglia
        

    def plot_decision_data(self, df, clf, save=False, name_file='grafico', title='Classificazione microclusters e Confini di Decisione'):
        '''
        Plotting dei decision dati per dati bidimensionali. Formato atteso:
        [dim1, dim2, macro_clusters, micro_clusters]
        '''
        X = df.iloc[:,:self.col_data].to_numpy()
        y = df.iloc[:,-1].to_numpy()
        # estraggo i confini massimi per poter generare una mappa heatmap del nostro classificatore
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        n_points = len(df) # prendo un numero di campioni conforme al numero di dataframe per poi far un campionamento uniforme

        # genero quindi un numero uniforme di punti da min a max per ogni dimensione
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_points), np.linspace(y_min, y_max, n_points))
        
        # determino la griglia di punti
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        print(f'Griglia di punti: {grid_points.shape}')
        # calcolo delle probabilità di ogni punto sulla griglia
        probs_grid = self.predict_proba_clf(model=clf, data=grid_points)
        
        # plot della classificazione con le probabilità di appartenenza e i confini decisionali
        for i in range(len(np.unique(y))):
            # probabilità per ciascun cluster (canale di colore)
            plt.contourf(xx, yy, probs_grid[:, i].reshape(xx.shape), alpha=0.3, cmap="coolwarm")
        
        labels = df['micro_clusters']
        # creiamo una colormap con colori distinti per le etichette uniche
        unique_labels = labels.unique()
        colormap = plt.cm.get_cmap("tab10", len(unique_labels))
        # colors = labels.map({label: i for i, label in enumerate(unique_labels)})
        scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap=colormap, edgecolor="k", label='Dati originali')
        # scatter = plt.scatter(X[:, 0], X[:, 1], c=colors, s=50, cmap=colormap, edgecolor="k", label='Dati originali')
        plt.colorbar(scatter, label="Cluster", ticks=range(len(unique_labels)))

        # Chiamata alla funzione ausiliaria per disegnare i confini decisionali
        self.plot_decision_boundary(clf, grid_points, xx, yy)

        plt.title(title)
        plt.xlabel("Dimensione 1")
        plt.ylabel("Dimensione 2")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        if save:
            current_datetime = date.today().strftime("%Y-%m-%d")
            cartella = f"results_{current_datetime}"
            if not os.path.exists(cartella):
                os.makedirs(cartella)
            file_path = os.path.join(cartella, f"{name_file}.png")
            plt.savefig(file_path)
            print(f"Grafico salvato in: {file_path}")
        plt.show()

    def train_test_loop(self, plots=True):
        '''
        A partire dal modelli addestrati e trovati stabilisco metriche di similarità.
        Formato atteso:
        [dim1, dim2,..dimk, macro, micro]
        '''
        df = self.dataframe.copy(deep=True)
        models = self.classifiers_macro_clusters(dataframe=df)
        for model in models:
            print("params: ", model[1])
        if plots:
            for i in range(2):
                part_df = self.dataframe[self.dataframe['macro_clusters'] == i]
                if i == 0:
                    name_file='clf_microclusters_benevoli'
                else:
                    name_file='clf_microclusters_malevoli'
                self.plot_decision_data(df=part_df, clf=models[i][0], save=True, name_file=name_file)
        return models

    def compute_similarity_matrix(self, X, y, clf):
        probs = self.predict_proba_clf(model=clf, data=X)
        # ritorna le probabilità di appartenenza ad ogni cluster per ogni punto di training. guarda sopra
        n_clusters = len(np.unique(y))
        similarity_matrix = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            prior_i = np.mean(probs[y==i][:,i]) 
            # valore medio del prior p_i che un punto appartenga al cluster i-esimo
            for j in range(n_clusters):
                probs_j = np.mean(probs[y==i][:,j])
                # valore medio della probabilità che un punto appartenente alla classe i-esima venga
                # classificato come appartenente al cluster j-esimo 
                similarity_matrix[i, j] = probs_j / prior_i
        # similarity_clusters = pd.DataFrame(similarity_matrix)
        return similarity_matrix

    def extract_k_sim_neighbours(self, matrix_df, k=None):
        '''
        Permette di estrarre dalla matrice i k clusters più vicini,
        ritornando un dizionario tale per cui estraiamo per ogni cluster quali sono i vicini
        e quanto sono simili, andando a normalizzare per il valore massimo (norm min-max)
        '''
        neigh_cluster = {}
        for row in range(matrix_df.shape[0]):
            matrix_df.iat[row,row] = 0
        max_value_global = matrix_df.values.max()
        for row in range(matrix_df.shape[0]):
            vals = {}
            for col in range(matrix_df.shape[1]):
                if (row != col):
                    vals[col] = matrix_df.iat[row,col] /  max_value_global
            # Ordina il dizionario per valore
            vals_ordered = dict(sorted(vals.items(), key=lambda item: item[1], reverse=True))
            # aggiungi i vicini della chiave row come lista
            neigh_cluster[row] = vals_ordered
        return neigh_cluster
    
    def compute_graphs(self, dataframe, clfs):
        print('Benigni------------------')
        pos_df = dataframe[dataframe['macro_clusters'] == 0]
        models_pos = clfs[0][0]
        X_pos = pos_df.iloc[:,:self.col_data].to_numpy()
        y_pos = pos_df.iloc[:,-1].to_numpy()
        sim_matrix_df_b = pd.DataFrame(self.compute_similarity_matrix(X=X_pos, y=y_pos, clf=models_pos))
        print(sim_matrix_df_b) # dataframe benevoli
        pos_dict = self.extract_k_sim_neighbours(sim_matrix_df_b)
        print(pos_dict) # dizionario dei benevolis
        print('Maligni------------------')
        neg_df = dataframe[dataframe['macro_clusters'] == 1]
        models_neg = clfs[1][0]
        X_neg = neg_df.iloc[:,:self.col_data].to_numpy()
        y_neg = neg_df.iloc[:,-1].to_numpy()
        sim_matrix_df_m = pd.DataFrame(self.compute_similarity_matrix(X=X_neg, y=y_neg, clf=models_neg))
        print(sim_matrix_df_m) # dataframe malevoli
        neg_dict = self.extract_k_sim_neighbours(sim_matrix_df_m)
        print(neg_dict) # dizionario malevoli
        return pos_dict, neg_dict


    def plot_oriented_graph(self, dict_adj):
        '''
        Plot di un grafo orientato a partire da un dizionario di adiacenzes
        '''
        # creiamo un grafo orientato
        G = nx.DiGraph()
        for node, neighbors in dict_adj.items():
            G.add_node(node)
            for node_neigh, weight in neighbors.items():
                G.add_edge(node, node_neigh, weight=weight)
        # stampiamo le informazioni sugli archi
        print("Archi del grafo orientato con i pesi:")
        for u, v, weight in G.edges(data="weight"):
            print(f"{u} -> {v} (weight: {weight})")
        # visualizzazione del grafo
        plt.figure(figsize=(8, 6))
        nx.draw(G, with_labels=True, node_color="skyblue", node_size=700, font_size=15, font_weight="bold")
        plt.show()

    def create_oriented_graph(self, dict_adj):
        G = nx.DiGraph()
        for node, neighbors in dict_adj.items():
            G.add_node(node)
            for node_neigh, weight in neighbors.items():
                G.add_edge(node, node_neigh, weight=weight)
        return G

    def minimum_dict_wrapped(self, dict_adj):
        '''
        questo metodo permette di trasformare un dizionario di
        similarità in un dizionario di distanze.
        In questo modo archi più piccoli saranno associati a nodi più simili
        E questo ci va bene perchè stiamo trovando un albero minimo ricoprente
        '''
        # prima trova il massimo
        max = 0.0
        for node, neighbors in dict_adj.items():
            for node_neigh, weight in neighbors.items():
                if weight > max:
                    max = weight
        new_dict = {}
        # quindi inverte i valori sulla base del massimo
        for node, neighbors in dict_adj.items():
            new_dict[node] = {}
            for node_neigh, weight in neighbors.items():
                new_dict[node][node_neigh] = max - weight
        return new_dict


    def kruskal_mst(self, graph):
        # ordino gli archi per peso , in ordine crescente
        # in questo modo archi più piccoli corrispondono a nodi più simili
        edges = sorted(graph.edges(data=True), key=lambda edge: edge[2]['weight'])
        # print(edges)
        mst = nx.Graph()  # Resulting minimum spanning tree
        # aggiungo i nodi all'albero mst inizializzato prima
        uf = UnionFind(len(graph.nodes))
        for u, v, data in edges:
            if uf.find(u) != uf.find(v):
                mst.add_edge(u, v, weight=data['weight'])
                uf.union(u, v)
            # Stop if we have n-1 edges in the MST
            if len(mst.edges) == len(graph.nodes) - 1:
                break
                
        mst_dict = self.mst_to_dict(mst)
        return mst, mst_dict

    def mst_to_dict(self, mst):
        '''
        converte un albero in un dizionario
        '''
        mst_dict = {}
        print(f'Archi: {mst.edges()}')
        for u, v, weight in mst.edges(data="weight"):
            if u not in mst_dict:
                mst_dict[u] = []
            if v not in mst_dict:
                mst_dict[v] = []
            mst_dict[u].append((v, weight))
            mst_dict[v].append((u, weight))
        return mst_dict

    def extract_minimum_spanning_tree(self, dict_adj):
        # wrapping per avere i pesi minimi da adattare a Kruscal
        min_dict = self.minimum_dict_wrapped(dict_adj)
        # grafo orientato
        min_graph = self.create_oriented_graph(min_dict)
        _, dict_tree = self.kruskal_mst(min_graph)
        return dict_tree