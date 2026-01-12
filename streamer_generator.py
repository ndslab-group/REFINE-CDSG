from datetime import date
import pandas as pd 
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Slider
from tqdm import tqdm
import time
from utils import Utility
from pathlib import Path
from joblib import Parallel, delayed


class StreamerGen:
    def __init__(self, dataframe_name, plotting, col_data):
        '''
        Il dataframe che andremo a leggere dovrà essere nel formato previsto:
        [dim1, dim2, ..., dimk, macro_cluster, micro_cluster, concept, anomaly]
        Dove:
        - dim1,...,dimk indicano le dimensioni effettive del dataframe di partenza
        - macro_cluster indica l'appartenenza di un campione a campioni benevoli o malevoli
        - micro_cluster indica l'appartenenza di un campione (benevolo o malevolo, in maniera univoca)
            ad un micro cluster (nel caso dei malevoli possiamo pensare ad una distinzione tra DDos o Dos etc)
        - concept indica l'appartenenza di un campione al concept o no
        - anomaly indica quanto un campione non appartenente al concept possa essere visto come outlier rispetto al concept stesso
        Nel caso in cui i dati siano bidimensionali allora è possibile plottare le prime due dimensioni.
        Eventualmente disabilitare plotting o prevedere un metodo interno che faccia copia delle utils ma riduca la dimensionalità
        '''
        self.df = pd.read_csv(os.path.join(os.getcwd(), dataframe_name))
        self.plotting = plotting
        self.col_data = col_data
        self.first_k_columns = None

    def getdataframe(self):
        return self.df.copy(deep=True)
    
    def save_stream(self, window_stream_df, filename):
        data_path = os.getcwd()
        data_path = Path(data_path)
        cartella = data_path / f"results_cdsg_directory"
        cartella.mkdir(parents=True, exist_ok=True)
        file_path = cartella / filename
        window_stream_df.to_csv(file_path, index=False)

    def save_params(self, perc_malignant_concept, perc_malignant_drift, intensity_ben, intensity_mal, win_size, start_drift, perc_drift_reached, nome_file):
        data_path = os.getcwd()
        data_path = Path(data_path)
        cartella = data_path / f"results_cdsg_directory"
        cartella.mkdir(parents=True, exist_ok=True)
        file_path = cartella / f'{nome_file}.txt'
        with open(file=file_path, mode='w') as f:
            print(f'Percentuale campioni maligni concetto richiesta: {perc_malignant_concept}', file=f)
            print(f'Percentuale campioni maligni drift richiesta: {perc_malignant_drift}', file=f)
            print(f'Intensità campioni benevoli per lo stream del drift: {intensity_ben}', file=f)
            print(f'Intensità campioni malevoli per lo stream del drift: {intensity_mal}', file=f)
            print(f'Dimensione finestra: {win_size}; Drift a partire dal campione: {start_drift}', file=f)
            print(f'Percentuale di campioni malevoli raggiunto nello stream: \n nel concept {perc_drift_reached[0]}, nel drift {perc_drift_reached[1]}', file=f)

    def plot_with_anomaly_slider(self, df):
        '''
        Dato un dataframe bidimensionale questa funzione permette di plottare i campioni anomali sulla base di uno
        slider che varia i valori di soglia in maniera dinamica.
        '''
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.1, bottom=0.25)  # Spazio per lo slider

        sc = ax.scatter(df['DIM1'], df['DIM2'], c=df['anomaly'], cmap="viridis", s=50, edgecolor="k", alpha=0.7)
        colorbar = plt.colorbar(sc, ax=ax)
        colorbar.set_label('Anomaly Score')

        ax.set_xlabel("Feature1")
        ax.set_ylabel("Feature2")
        ax.set_title("Scatter Plot con Filtro Anomaly")
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)

        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor="lightgoldenrodyellow")
        slider = Slider(ax_slider, 'Anomaly Threshold', 0.0, 2.0, valinit=0.0)

        # Funzione di aggiornamento per il filtro
        def update(val):
            threshold = slider.val
            # Filtra i dati in base alla soglia dell'anomaly
            mask = df['anomaly'] >= threshold
            filtered_x = df['DIM1'].where(mask, np.nan)
            filtered_y = df['DIM2'].where(mask, np.nan)
            
            # Aggiorna il grafico
            sc.set_offsets(np.c_[filtered_x, filtered_y])
            sc.set_array(df['anomaly'])  # manteniamo l'array intero per il color mapping
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()


    def plot_sliding_windows(self, window_df, start_win, end_win):
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.1, bottom=0.25)  # Spazio per lo slider

        sc = ax.scatter([], [], c=[], cmap="viridis", s=50, edgecolor="k", alpha=0.7)
        colorbar = plt.colorbar(sc, ax=ax)
        colorbar.set_label('Concept / Drift')

        ax.set_xlabel("Feature1")
        ax.set_ylabel("Feature2")
        ax.set_title("Scatter Plot della window")
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)

        # Crea l'asse per lo slider
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor="lightgoldenrodyellow")
        slider = Slider(ax_slider, 'Anomaly Threshold', start_win, end_win, valinit=start_win)
        last_point = ax.scatter([], [], color="red", s=150, edgecolor="darkred", label="Ultimo Campione", zorder=5)

        # Funzione di aggiornamento per il filtro
        def update(val):
            threshold = slider.val
            # Filtra i dati in base alla soglia dell'anomaly
            mask = window_df['WIN'] < threshold
            filtered_x = window_df['DIM1'].where(mask, np.nan)
            filtered_y = window_df['DIM2'].where(mask, np.nan)

            # Aggiorna il grafico
            sc.set_offsets(np.c_[filtered_x, filtered_y])
            sc.set_array(window_df['concept'])  # manteniamo l'array intero per il color mapping

            last_point_ = window_df[window_df['WIN'] == round(threshold)]
            # Solo se esiste un valore valido, aggiorna il punto rosso
            if not last_point_.empty:
                last_point.set_offsets(np.c_[last_point_['DIM1'], last_point_['DIM2']])
            fig.canvas.draw_idle()
            
        slider.on_changed(update)
        plt.show()
    
    
    def extract_drift_samples(self, drift_df, intensity_ben, intensity_mal, drift_win):
        '''
        Descrizione:
        Questa funzione permette di estrarre, sulla base di un dataframe di drift, campioni
        di una determinata intensità al fine di soddisfare una percentuale di campioni richiesta e di un drift_win
        Input:
        - drift df
        - intensità benevoli (in termini di outlier score)
        - intensità malevoli (in termini di outlier score)
        - drift_win: numero di campioni richiesti nella finestra
        Output:
        - restituisce le soglie da considerare per i campioni di drift per benevoli (indice 0) e benevoli (indice 1)
        '''
        intensity_ben = intensity_ben * 2
        intensity_mal = intensity_mal * 2
        intensities = [intensity_ben, intensity_mal]
        drift_samples_ = [] # raccoglie i campioni per il drift per benigni e malevoli
        values_ = [drift_df.loc[drift_df['macro_clusters'] == 0, 'anomaly'].value_counts(), 
                   drift_df.loc[drift_df['macro_clusters'] == 1, 'anomaly'].value_counts()]
        thresholds_ = [drift_df.loc[drift_df['macro_clusters'] == 0, 'anomaly'].unique(),
                       drift_df.loc[drift_df['macro_clusters'] == 1, 'anomaly'].unique()]
        for i in range(len(values_)):
            samples_ = []
            intensity = intensities[i]
            # aggiunge prima i campioni benevoli, poi quelli malevoli
            sum_ = 0.0
            thresholds = thresholds_[i]
            values = values_[i]
            for threshold in sorted(thresholds, reverse=True):
                if threshold >= intensity:
                    values_t = values[threshold]
                    samples_.append((threshold, values[threshold]))
                    sum_ += values_t
                    if sum_ >= drift_win:
                        break
            drift_samples_.append(samples_)
        return drift_samples_


    def check_proportion(self, dataframe, perc):
        '''
        Descrizione: 
        Questa funzione permette di controllare se un dataframe rispetta la percentale richiesta
        di campioni benevoli e malevoli.
        Input:
        - dataframe
        - percentuale malevoli
        Output:
        - flag: true se rispetta la percentuale di almeno quel X%, false altrimenti (quindi lancia un errore)
        '''
        tot = len(dataframe)
        tot_mal = len(dataframe[dataframe['macro_clusters'] == 1])
        print(f'Dimensione concept: {tot}, \nTotale campioni malevoli nel concept: {tot_mal}')
        flag = False
        if (tot_mal / tot) >= perc:
            flag = True
        else:
            # bisogna abbassare la percentuale come suggerimento.
            while perc >= 0.05:
                perc -= 0.05
                if (tot_mal / tot) >= perc:
                    break
        return perc, flag
    
    def check_drift_samples(self, drift_ben_samples, drift_mal_samples, perc_malignant_drift, drift_win):
        '''
        Questa funzione permette di verificare se:
        - ci siano abbastanza campioni per fare il drift (restituisce una flag in prima posizione)
        - venga controllato il rapporto di campioni maligni affinché corrisponda alla percentuale richiesta
            come parametro (deve essere almeno >= a quella richiesta)
        Restituisce:
        - prima flag che ci indica che abbiamo ancora pochi campioni, quindi dobbiamo diminuire la intensità dei benevoli
        - seconda flag che ci indica che non abbiamo la proporzione rispettata, quindi dobbiamo diminuire l'intensità dei malevoli
        '''
        ben_drift_samples = sum(item[1] for item in drift_ben_samples)
        mal_drift_samples = sum(item[1] for item in drift_mal_samples)
        print(f'Campioni benevoli: {ben_drift_samples}\nCampioni malevoli: {mal_drift_samples}')
        tot_samples = ben_drift_samples + mal_drift_samples
        flag1_, flag2_ = False, False
        if tot_samples < drift_win:
            flag1_ = True
        if (mal_drift_samples / tot_samples) < perc_malignant_drift:
            flag2_ = True
        return flag1_, flag2_

    
    def extract_sample(self, df_mal, df_ben, malignant_proba, perc_malignant):
        """
        Estrae un campione senza alterare il DataFrame originale.
        Attenzione: in questo modo il campione non verrà rimosso dal dataset e potrebbe avere equiprobabilità di essere ripescato
        """
        already_taken = True
        if malignant_proba < perc_malignant:
            # estrai un campione malevolo se la probabilità è bassa
            if not df_mal.empty:
                sample = df_mal.sample(n=1, replace=False)
            elif not df_ben.empty:
                sample = df_ben.sample(n=1, replace=False)
        else:
            # altrimenti estrai un benevolo
            if not df_ben.empty:
                sample = df_ben.sample(n=1, replace=False)
            elif not df_mal.empty:
                sample = df_mal.sample(n=1, replace=False)
        return sample


    def generate_samples(self, concept_df, drift_df, win_size, start_drift, perc_malignant_concept, perc_malignant_drift, recurrent=False, rec_drift=0):
        """
        Genera campioni ottimizzati con parallelizzazione ed estrazione batch.
        """
        # Pre-filtriamo i macro_clusters UNA SOLA VOLTA
        concept_mal = concept_df[concept_df['macro_clusters'] == 1].copy()
        concept_ben = concept_df[concept_df['macro_clusters'] == 0].copy()
        drift_mal = drift_df[drift_df['macro_clusters'] == 1].copy()
        drift_ben = drift_df[drift_df['macro_clusters'] == 0].copy()

        # Determiniamo in anticipo dove avviene il drift
        drift_mask = np.zeros(win_size, dtype=bool)
        if start_drift < win_size:
            drift_mask[start_drift:] = True
        if recurrent and rec_drift < win_size:
            drift_mask[rec_drift:] = ~drift_mask[rec_drift:]  # XOR per il drift ricorrente

        # Generiamo in batch le probabilità di malignità
        malign_probs = np.random.rand(win_size)

        # Fuori da process_sample:
        columns_to_select = list(concept_df.columns[:self.col_data]) + ['macro_clusters', 'concept']

        results = []
        used_indices_concept_mal = set()
        used_indices_concept_ben = set()
        used_indices_drift_mal = set()
        used_indices_drift_ben = set()
        for i in tqdm(range(win_size), desc="Elaborazione campioni"):
            use_concept = not drift_mask[i]
            if use_concept:
                sample = self.extract_sample(concept_mal, concept_ben, malign_probs[i], perc_malignant_concept)
                if malign_probs[i] < perc_malignant_concept:
                    concept_mal = concept_mal.loc[~concept_mal.index.isin(used_indices_concept_mal)]
                    used_indices_concept_mal.update(sample.index)
                else:
                    concept_ben = concept_ben.loc[~concept_ben.index.isin(used_indices_concept_ben)]
                    used_indices_concept_ben.update(sample.index)
            else:
                sample = self.extract_sample(drift_mal, drift_ben, malign_probs[i], perc_malignant_drift)
                if malign_probs[i] < perc_malignant_drift:
                    drift_mal = drift_mal.loc[~drift_mal.index.isin(used_indices_drift_mal)]
                    used_indices_drift_mal.update(sample.index)
                else:
                    drift_ben = drift_ben.loc[~drift_ben.index.isin(used_indices_drift_ben)]
                    used_indices_drift_ben.update(sample.index)
            sample_numeric = sample[columns_to_select].values.ravel()
            results.append(np.append(sample_numeric, i).tolist())
        
        window_formatted = results
        # Definiamo le colonne finali, includendo 'macro_clusters', 'concept', e 'WIN'

        self.first_k_columns = concept_df.columns[:self.col_data].tolist()
        new_columns = self.first_k_columns + ['macro_clusters', 'concept', 'WIN']
        
        # Restituisci la finestra formattata e le nuove colonne
        return window_formatted, new_columns
    
    def reduce_dataset_proportion(self, dataframe, perc):
        # campioni negativi in percentuale:
        perc_neg = perc * 100
        perc_pos = 100 - perc_neg
        tot = len(dataframe)
        tot_mal = len(dataframe[dataframe['macro_clusters'] == 1])
        tot_pos = len(dataframe[dataframe['macro_clusters'] == 0])
        num_pos_to_sample = int((tot_mal * perc_pos ) / perc_neg)

        # Adatta il numero di positivi in base ai dati disponibili
        num_pos_to_sample = min(num_pos_to_sample, tot_pos)
        print(f"Campioni negativi disponibili: {tot_mal}, Campioni positivi disponibili: {tot_pos}")
        print(f"Campioni positivi selezionati: {num_pos_to_sample}, Campioni negativi selezionati: {tot_mal}")
        
        # Campiona i dati senza sostituzione
        sampled_neg = dataframe[dataframe['macro_clusters'] == 1].sample(n=tot_mal, replace=False)
        sampled_pos = dataframe[dataframe['macro_clusters'] == 0].sample(n=num_pos_to_sample, replace=False)
        
        # Unisci i due dataset campionati
        sampled_df = pd.concat([sampled_neg, sampled_pos])
        print(f"Dataset ridotto con {len(sampled_df)} campioni")
        return sampled_df

    
    def recurrent_drift_generator(self, win_size, start_drift, rec_drift, perc_malignant_concept, perc_malignant_drift, save_stream=False, plot_window=False, filename='streaming.csv', intensity_mode='auto'):
        '''
        Descrizione:
            Questa funzione permette di generare un sudden drift recurrrent (del tipo A-B-A) a partire da un dataframe passato in input
            al generatore, atteso nel formato [dim1, dim2, ..., dimk, macro_clusters, micro_clusters, concept, anomaly].
            I campioni del concept verranno presi dal dataframe dove la label concept=1. I campioni del drift verranno presi dal dataframe
            risultante, sulla base di una percentuale di campioni malevoli da rispettare: se l'utente mi chiede di generare una finestra
            di drift da X campioni con percentuale dei campioni malevoli del x.x% rispetto al totale (in termini statistici),
            il generatore procederà a selezionare campioni di drift a partire da una soglia di intesità pari a 1.0, andando a diminuire
            incrementalmente per poter soddisfare la percentuale di ripartizione statistica richiesta (sia per benevoli che per malevoli)
            La generazione dello stream prevede poi estrazione randomica senza reinserimento dei campioni a partire dai due dataframe.
        Input:
            - win_size: dimensione della finestra dove far avvenire il drift
            - start_drift: quando far iniziare il drift
            - rec_drift: quando tornarre al concept
            - perc_malignant_concept: percentuale di campioni malevoli da rispettare nel concept in termini statistici. Serve anche
                come vincolo da rispettare in fase di creazione del concept_df; se non viene rispettata la partizione viene lanciato un
                errore con un suggerimento utile per poter scegliere successivamente il valore corretto; la stessa percentuale viene poi usata
                come probabilità di estrazione in fase di generazione.
            - perc_drift_concept: come prima, percentuale di campioni malevoli da rispettare nel drift in termini statistici. Serve
                anche come vincolo da rispettare in fase di creazione del drift_df; se non viene rispettata la proporzione, allora si procede con una
                ridimensione dei valori di intensità. Il modello cercherà di estrarre i valori di intensità corretti tramite una strategia Greedy
                da applicare al mio dataframe per estrarre il giusto numero di campioni malevoli e benevoli sulla base di percentuali di campioni e dimensione
                della finestra, Cercherà di dare il massimo nel senso che partità da valori di intensità elevati
            Output:
            - file riepilogativo dei campioni estratti, la finestra e l'intensità raggiunta
            - campioni estratti volta dopo volta
            - dataframe dei campioni estratti volta dopo volta
        =============================
        al momento l'intensità viene gestita in maniera automatica, per poterla impostare manualmente bisogna passare una tupla
        del tipo (intensità_benevoli, intensità_malevoli) come parametro intensity_mode
        =============================
        '''
        drift_win = win_size - rec_drift + start_drift
        concept_win = start_drift + (win_size - rec_drift)
        if win_size < start_drift:
            raise ValueError('Attenzione! Dimensione finestra e start drift invalida')
        if rec_drift < start_drift:
            raise ValueError('Attenzione! La finestra di recurrent dovrebbe venire dopo la finestra di drift')
        
        concept_df = self.df[self.df['concept'] == 1].copy(deep=True)
        concept_df = self.reduce_dataset_proportion(dataframe=concept_df, perc=perc_malignant_concept)
        
        print(f'Concept df size: {len(concept_df)}')
        if len(concept_df) < concept_win:
            raise ValueError('Attenzione! Dimensione concetto insufficiente per coprire la finestra richiesta')
        
        print(f'Drift windown: {drift_win}')
        print(f'Concept win: {concept_win}')
        drift_df = self.df[self.df['concept'] == 0].copy(deep=True)
        
        if len(drift_df) < drift_win:
            raise ValueError('Attenzione! Dimensione drift insufficiente per coprire la finestra richiesta')
        
        # il valore di intensità in questo caso viene impostato in maniera automatico, altrimenti andrebbe 
        # specificato qui sotto e commentare il restante pezzo di codice
        intensity_ben = 1.0
        intensity_mal = 1.0
        decrease_intensity = True
        
        if intensity_mode == 'auto':
            # commentare qui sotto se l'intensità deve essere regolata manualmente
            while decrease_intensity:
                print(f'Intensità benevoli: {intensity_ben}\nIntesità malevoli: {intensity_mal}') 
                drift_samples = self.extract_drift_samples(drift_df=drift_df, intensity_ben=intensity_ben, intensity_mal=intensity_mal, drift_win=drift_win)
                flag1_, flag2_ = self.check_drift_samples(drift_ben_samples=drift_samples[0], drift_mal_samples=drift_samples[1], perc_malignant_drift=perc_malignant_drift, drift_win=drift_win)
                if flag1_:
                    print(f'Warning! Campioni insufficienti per realizzare il drift. Diminuisco intensità benevoli')
                    intensity_ben -= 0.05
                    if intensity_ben <= 0.0:
                        intensity_ben = 0.0
                        flag1_ = False
                    decrease_intensity = True
                if flag2_:
                    print("Warning! Percentuale campioni malevoli non rispettata nel drift, diminuisco l'intensità dei malevoli")
                    intensity_mal -= 0.05
                    if intensity_mal <= 0.0:
                        raise ValueError("Errore....intensità negativa nei malevoli, numero di campioni insufficiente!....")
                    decrease_intensity = True
                if not (flag1_ or flag2_):
                    decrease_intensity = False
        else: 
            intensity_ben, intensity_mal = intensity_mode
            
        drift_df['in_win'] = 0
        for i in range(len(drift_samples)):
            thresholds_ = [item[0] for item in drift_samples[i]]
            for threshold in thresholds_:
                drift_df.loc[(drift_df['macro_clusters'] == i) & (drift_df['anomaly'] == threshold), 'in_win'] = 1
       
        # print(f'Intensità benevoli: {intensity_ben}\nIntesità malevoli: {intensity_mal}') 
        
        # generazione della finestra
        samples_win, new_cols = self.generate_samples(concept_df=concept_df, drift_df=drift_df[drift_df['in_win'] == 1], win_size=win_size, start_drift=start_drift, perc_malignant_concept=perc_malignant_concept, perc_malignant_drift=perc_malignant_drift, recurrent=True, rec_drift=rec_drift)
        win_df = pd.DataFrame(samples_win, columns=new_cols)
        perc_mal_reached_concept = len(win_df[(win_df['macro_clusters'] == 1) & (win_df['concept'] == 1)]) / len(win_df[win_df['concept'] == 1])
        perc_mal_reached_drift = len(win_df[(win_df['macro_clusters'] == 1) & (win_df['concept'] == 0)]) / len(win_df[win_df['concept'] == 0])
        perc_mal_reached = [perc_mal_reached_concept, perc_mal_reached_drift]
        
        if plot_window:
            self.plot_sliding_windows(window_df=win_df, start_win=0, end_win=win_size)
        
        if save_stream:
            self.save_stream(win_df, filename)
            self.save_params(
                perc_malignant_concept=perc_malignant_concept, 
                perc_malignant_drift=perc_malignant_drift, 
                intensity_ben=intensity_ben, 
                intensity_mal=intensity_mal,
                win_size=win_size,
                start_drift=start_drift,
                perc_drift_reached=perc_mal_reached
                )

        return samples_win

    
    def sudden_drift_generator(self, win_size, start_drift, perc_malignant_concept, perc_malignant_drift, nome_file_parametri, save_stream=False, plot_window=False, filename='streaming.csv', intensity_mode='auto'):
        '''
        Descrizione:
        Questa funzione permette di generare un sudden drift a partire da un dataframe passato in input
        al generatore, atteso nel formato [dim1, dim2, ..., dimk, macro_clusters, micro_clusters, concept, anomaly].
        I campioni del concept verranno presi dal dataframe dove la label concept=1. I campioni del drift verranno presi dal dataframe
        risultante, sulla base di una percentuale di campioni malevoli da rispettare: se l'utente mi chiede di generare una finestra
        di drift da X campioni con percentuale dei campioni malevoli del x.x% rispetto al totale (in termini statistici),
        il generatore procederà a selezionare campioni di drift a partire da una soglia di intesità pari a 1.0, andando a diminuire
        incrementalmente per poter soddisfare la percentuale di ripartizione statistica richiesta (sia per benevoli che per malevoli)
        La generazione dello stream prevede poi estrazione randomica senza reinserimento dei campioni a partire dai due dataframe.
        Input:
        - win_size: dimensione della finestra dove far avvenire il drift
        - start_drift: quando far iniziare il drift
        - perc_malignant_concept: percentuale di campioni malevoli da rispettare nel concept in termini statistici. Serve anche
            come vincolo da rispettare in fase di creazione del concept_df; se non viene rispettata la partizione viene lanciato un
            errore con un suggerimento utile per poter scegliere successivamente il valore corretto; la stessa percentuale viene poi usata
            come probabilità di estrazione in fase di generazione.
        - perc_drift_concept: come prima, percentuale di campioni malevoli da rispettare nel drift in termini statistici. Serve
            anche come vincolo da rispettare in fase di creazione del drift_df; se non viene rispettata la proporzione, allora si procede con una
            ridimensione dei valori di intensità. Il modello cercherà di estrarre i valori di intensità corretti tramite una strategia Greedy
            da applicare al mio dataframe per estrarre il giusto numero di campioni malevoli e benevoli sulla base di percentuali di campioni e dimensione
            della finestra, Cercherà di dare il massimo nel senso che partità da valori di intensità elevati
        Output:
        - file riepilogativo dei campioni estratti, la finestra e l'intensità raggiunta
        - campioni estratti volta dopo volta
        - dataframe dei campioni estratti volta dopo volta
        =============================
        al momento l'intensità viene gestita in maniera automatica, per poterla impostare manualmente bisogna passare una tupla
        del tipo (intensità_benevoli, intensità_malevoli) come parametro intensity_mode
        =============================
        '''
        if win_size < start_drift:
            raise ValueError('Attenzione! Dimensione finestra e start drift invalida')
        drift_win = win_size - start_drift
        concept_df = self.df[self.df['concept'] == 1].copy(deep=True)
        
        print(f'Concept df size: {len(concept_df)}')
        drift_df = self.df[self.df['concept'] == 0].copy(deep=True)
        
        print(f'Drift size: {len(drift_df)}')
        if start_drift > len(concept_df):
            raise ValueError('Attenzione! Dimensione concetto insufficiente per coprire la finestra richiesta')
        
        print(f'Drift windown: {drift_win}')
        intensity_ben = 1.0
        intensity_mal = 1.0
        
        if intensity_mode == 'auto':
            decrease_intensity = True
            while decrease_intensity:
                print(f'Intensità benevoli: {intensity_ben}\nIntesità malevoli: {intensity_mal}') 
                drift_samples = self.extract_drift_samples(drift_df=drift_df, intensity_ben=intensity_ben, intensity_mal=intensity_mal, drift_win=drift_win)
                flag1_, flag2_ = self.check_drift_samples(drift_ben_samples=drift_samples[0], drift_mal_samples=drift_samples[1], perc_malignant_drift=perc_malignant_drift, drift_win=drift_win)
                if flag1_:
                    print(f'Warning! Campioni insufficienti per realizzare il drift. Diminuisco intensità benevoli')
                    intensity_ben -= 0.05
                    if intensity_ben <= 0.0:
                        raise ValueError("Errore....intensità negativa dei benevoli, numero di campioni insufficiente!....")
                    decrease_intensity = True
                if flag2_:
                    print("Warning! Percentuale campioni malevoli non rispettata nel drift, diminuisco l'intensità dei malevoli")
                    intensity_mal -= 0.05
                    if intensity_mal <= 0.0:
                        raise ValueError("Errore....intensità negativa nei malevoli, numero di campioni insufficiente!....")
                    decrease_intensity = True
                if not (flag1_ or flag2_):
                    decrease_intensity = False
        else:
            intensity_ben, intensity_mal = intensity_mode

        drift_df['in_win'] = 0
        for i in range(len(drift_samples)):
            thresholds_ = [item[0] for item in drift_samples[i]]
            for threshold in thresholds_:
                drift_df.loc[(drift_df['macro_clusters'] == i) & (drift_df['anomaly'] == threshold), 'in_win'] = 1
        
        print(f'Intensità benevoli: {intensity_ben}\nIntesità malevoli: {intensity_mal}') 
       
        samples_win, new_cols = self.generate_samples(concept_df=concept_df, drift_df=drift_df[drift_df['in_win'] == 1], win_size=win_size, start_drift=start_drift, perc_malignant_concept=perc_malignant_concept, perc_malignant_drift=perc_malignant_drift, recurrent=False)
        win_df = pd.DataFrame(samples_win, columns=new_cols)
        perc_mal_reached_concept = len(win_df[(win_df['macro_clusters'] == 1) & (win_df['concept'] == 1)]) / len(win_df[win_df['concept'] == 1])
        perc_mal_reached_drift = len(win_df[(win_df['macro_clusters'] == 1) & (win_df['concept'] == 0)]) / len(win_df[win_df['concept'] == 0])
        perc_mal_reached = [perc_mal_reached_concept, perc_mal_reached_drift]
        
        if plot_window:
            self.plot_sliding_windows(window_df=win_df, start_win=0, end_win=win_size)
        
        if save_stream:
            self.save_stream(win_df, filename)
            self.save_params(
                perc_malignant_concept=perc_malignant_concept, 
                perc_malignant_drift=perc_malignant_drift, 
                intensity_ben=intensity_ben, 
                intensity_mal=intensity_mal,
                win_size=win_size,
                start_drift=start_drift,
                perc_drift_reached=perc_mal_reached,
                nome_file=nome_file_parametri
                )

        return samples_win