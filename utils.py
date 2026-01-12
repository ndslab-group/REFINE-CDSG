import pandas as pd 
import numpy as np
import os
from datetime import date
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Slider

class Utility:
    def __init__(self, col_data, col_names):
        print("Utility")
        self.col_data = col_data
        self.col_names = col_names

    def plot_dataset(self, dataset, feature, title=None, save=False, name_file='grafico'):
        """
        Plotta un dataset 2D con due feature e una feature categoriale.
        Parameters:
        - dataset: DataFrame di Pandas contenente almeno due colonne per le feature e una colonna categoriale.
        - feature: stringa con il nome della colonna categoriale.
        """
        # Estraiamo le feature e le etichette
        x = dataset.iloc[:, 0]
        y = dataset.iloc[:, 1]
        labels = dataset[feature]

        unique_labels = np.unique(labels)
        colormap = plt.cm.get_cmap("viridis", len(unique_labels))

        # Plotting
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(x, y, c=labels, cmap=colormap, s=50, edgecolor="k", alpha=0.7)
        plt.colorbar(scatter, ticks=range(len(unique_labels)), label=feature)
        
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        if title is None:
            title = f"Scatter plot di {feature}"
        plt.title(title)
        plt.xlim(0,40)
        plt.ylim(0,30)
        if save:
            current_datetime = date.today().strftime("%Y-%m-%d")
        
            cartella = f"results_{current_datetime}" # Nome della cartella
           
            if not os.path.exists(cartella):
                os.makedirs(cartella)
            
            file_path = os.path.join(cartella, f"{name_file}.png")
            plt.savefig(file_path)
            print(f"Grafico salvato in: {file_path}")
        plt.show()

    def plot_entire_dataset(self, dataframe, save=False, name_file='dataset'):
        # Assumiamo che il dataframe 'df' abbia colonne 'DIM1' e 'DIM2'
        plt.figure(figsize=(8, 6))
        plt.scatter(dataframe['DIM1'], dataframe['DIM2'], color='blue', alpha=0.7)
        plt.xlabel('dim1')
        plt.ylabel('dim2')
        plt.title('Scatter Plot di DIM1 vs DIM2')
        plt.grid(True)
        if save:
            current_datetime = date.today().strftime("%Y-%m-%d")
            cartella = f"results_{current_datetime}"
            if not os.path.exists(cartella):
                os.makedirs(cartella)
            file_path = os.path.join(cartella, f"{name_file}.png")
            plt.savefig(file_path)
            print(f"Grafico salvato in: {file_path}")
        plt.show()


    def plot_contours_outlier(self, X, y, df, clf):
        # estraggo i confini massimi per poter generare una mappa heatmap del nostro classificatore
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        # genero quindi un numero uniforme di punti da min a max per ogni dimensione
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        # determino la griglia di punti
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        print(f'Griglia di punti: {grid_points.shape}')
        # calcolo delle probabilità di ogni punto sulla griglia
        scores_ = clf.decision_function(grid_points)
        scores_ = scores_.reshape(xx.shape)
        # Plot della classificazione con le probabilità di appartenenza e i confini decisionali
        plt.contourf(xx, yy, scores_, levels=np.linspace(scores_.min(), 0, 7), cmap=plt.cm.PuBu)
        plt.contourf(xx, yy, scores_, levels=[0, scores_.max()], colors="palevioletred")   
        labels = (df['anomaly'] * 10).astype(int)
        # Creiamo una colormap con colori distinti per le etichette uniche
        unique_labels = labels.unique()
        colormap = plt.cm.get_cmap("tab10", len(unique_labels))
        colors = labels.map({label: i for i, label in enumerate(unique_labels)})
        scatter = plt.scatter(X[:, 0], X[:, 1], c=colors, s=50, cmap=colormap, edgecolor="k", label='Dati originali')
        plt.colorbar(scatter, label="Cluster", ticks=range(len(unique_labels)))
        # Titolo e limiti
        plt.title("Decision boundaries Isolation Forest per punti esterni al concept")
        plt.xlabel("Dimensione 1")
        plt.ylabel("Dimensione 2")
        plt.xlim(0, 40)
        plt.ylim(0, 35)
        plt.show()



    def plot_with_anomaly_slider(self, df):
        '''
        Dato un dataframe bidimensionale questa funzione permette di plottare i campioni anomali sulla base di uno
        slider che varia i valori di soglia in maniera dinamica.
        '''
        x_min = df.iloc[:,0].min()
        x_max = df.iloc[:,0].max()
        y_min = df.iloc[:,1].min()
        y_max = df.iloc[:,1].max()

        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.1, bottom=0.25)  # Spazio per lo slider

        sc = ax.scatter(df['DIM1'], df['DIM2'], c=df['anomaly'], cmap="viridis", s=50, edgecolor="k", alpha=0.7)
        colorbar = plt.colorbar(sc, ax=ax)
        colorbar.set_label('Anomaly Score')

        ax.set_xlabel("Feature1")
        ax.set_ylabel("Feature2")
        ax.set_title("Scatter Plot con Filtro Anomaly")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor="lightgoldenrodyellow")
        slider = Slider(ax_slider, 'Anomaly Threshold', 0.0, 2.0, valinit=0.0)

        # funzione di aggiornamento per il filtro
        def update(val):
            threshold = slider.val
            # filtra i dati in base alla soglia dell'anomaly
            mask = df['anomaly'] >= threshold
            filtered_x = df['DIM1'].where(mask, np.nan)
            filtered_y = df['DIM2'].where(mask, np.nan)
            
            # aggiorna il grafico
            sc.set_offsets(np.c_[filtered_x, filtered_y])
            sc.set_array(df['anomaly'])  # manteniamo l'array intero per il color mapping
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show() 


    def plot_distances(self, n_windows, d_pos, d_neg, title, save=False, filename='distanze_wasserstein'):
        '''
        Realizza uno scatter plot delle distanze positive e negative
        su tutte le finestre nel range self.n_windows.
        
        Parametri:
        - d_pos: lista delle distanze positive
        - d_neg: lista delle distanze negative
        - title
        '''
        windows = range(n_windows)
        plt.figure(figsize=(10, 6))
        plt.plot(windows, d_pos, color='blue', label='Positive Distances', alpha=0.7)
        plt.plot(windows, d_neg, color='red', label='Negative Distances', alpha=0.7)
        plt.title(title)
        plt.xlabel("Window")
        plt.ylabel("Distance")
        plt.legend()
        plt.grid(True)
        if save:
            current_datetime = date.today().strftime("%Y-%m-%d")
            cartella = f"results_{current_datetime}"
            if not os.path.exists(cartella):
                os.makedirs(cartella)
            file_path = os.path.join(cartella, f"{filename}.png")
            plt.savefig(file_path)
            print(f"Grafico salvato in: {file_path}")
        plt.plot()


    def plot_performance(self, accuracies, save=False, filename='accuracy'):
        window_indices = list(range(1, len(accuracies) + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(window_indices, accuracies, marker='o', linestyle='-', color='b', label="Accuratezza per finestra")
        plt.xlabel("Finestra")
        plt.ylabel("F1-Score")
        plt.title("Andamento della f1-score per ciascuna finestra")
        plt.legend(loc="best")
        plt.grid()
        if save:
            current_datetime = date.today().strftime("%Y-%m-%d")
            cartella = f"results_{current_datetime}"
            if not os.path.exists(cartella):
                os.makedirs(cartella)
            file_path = os.path.join(cartella, f"{filename}.png")
            plt.savefig(file_path)
            print(f"Grafico salvato in: {file_path}")
        plt.show()