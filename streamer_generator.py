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
    def __init__(self, dataframe_name, plotting, col_data, clustering_technique, directory_stream):
        '''
        The dataframe we will be reading must be in the specified format:
        [dim1, dim2, ..., dimk, macro_cluster, micro_cluster, concept, anomaly]
        Where:
        - dim1,...,dimk indicate the actual dimensions of the source dataframe
        - macro_cluster indicates whether a sample belongs to benign or malicious samples
        - micro_cluster indicates to which micro cluster a sample (benign or malicious, unambiguously)
            to a micro-cluster (in the case of malicious samples, we can consider a distinction between DDoS or DoS, etc.)
        - concept indicates whether a sample belongs to the concept or not
        - anomaly indicates the extent to which a sample not belonging to the concept can be viewed as an outlier relative to the concept itself
        If the data is two-dimensional, then it is possible to plot the first two dimensions.
        If necessary, disable plotting or provide an internal method that copies the utilities but reduces the dimensionality

        This dataset is located within the folder created by the generator that contains the concept_df.csv file
        '''
        self.df = pd.read_csv(os.path.join(os.getcwd(), dataframe_name))
        self.plotting = plotting
        self.col_data = col_data
        self.first_k_columns = None
        self.clustering_technique = clustering_technique
        self.directory_stream = directory_stream

    def getdataframe(self):
        return self.df.copy(deep=True)
    
    def save_stream(self, window_stream_df, filename):
        data_path = Path(f'{self.directory_stream}/{self.clustering_technique}')
        data_path.mkdir(parents=True, exist_ok=True)
        file_path = data_path / filename
        window_stream_df.to_csv(file_path, index=False)

    def save_params(self, perc_malignant_concept, perc_malignant_drift, intensity_ben, intensity_mal, win_size, start_drift, perc_drift_reached, nome_file):
        data_path = Path(f'{self.directory_stream}/{self.clustering_technique}')
        data_path.mkdir(parents=True, exist_ok=True)
        file_path = data_path / f'{nome_file}.txt'
        with open(file=file_path, mode='w') as f:
            print(f'Percentuale campioni maligni concetto richiesta: {perc_malignant_concept}', file=f)
            print(f'Percentuale campioni maligni drift richiesta: {perc_malignant_drift}', file=f)
            print(f'Intensità campioni benevoli per lo stream del drift: {intensity_ben}', file=f)
            print(f'Intensità campioni malevoli per lo stream del drift: {intensity_mal}', file=f)
            print(f'Dimensione finestra: {win_size}; Drift a partire dal campione: {start_drift}', file=f)
            print(f'Percentuale di campioni malevoli raggiunto nello stream: \n nel concept {perc_drift_reached[0]}, nel drift {perc_drift_reached[1]}', file=f)

    def plot_with_anomaly_slider(self, df):
        '''
        Given a two-dimensional dataframe, this function allows you to plot outliers based on a
        slider that dynamically adjusts the threshold values.
        '''
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.1, bottom=0.25)  

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

        def update(val):
            threshold = slider.val
    
            mask = df['anomaly'] >= threshold
            filtered_x = df['DIM1'].where(mask, np.nan)
            filtered_y = df['DIM2'].where(mask, np.nan)
            
            sc.set_offsets(np.c_[filtered_x, filtered_y])
            sc.set_array(df['anomaly'])  
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()


    def plot_sliding_windows(self, window_df, start_win, end_win):
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.1, bottom=0.25)  

        sc = ax.scatter([], [], c=[], cmap="viridis", s=50, edgecolor="k", alpha=0.7)
        colorbar = plt.colorbar(sc, ax=ax)
        colorbar.set_label('Concept / Drift')

        ax.set_xlabel("Feature1")
        ax.set_ylabel("Feature2")
        ax.set_title("Scatter Plot della window")
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)

        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor="lightgoldenrodyellow")
        slider = Slider(ax_slider, 'Anomaly Threshold', start_win, end_win, valinit=start_win)
        last_point = ax.scatter([], [], color="red", s=150, edgecolor="darkred", label="Ultimo Campione", zorder=5)

        def update(val):
            threshold = slider.val
        
            mask = window_df['WIN'] < threshold
            filtered_x = window_df['DIM1'].where(mask, np.nan)
            filtered_y = window_df['DIM2'].where(mask, np.nan)

            sc.set_offsets(np.c_[filtered_x, filtered_y])
            sc.set_array(window_df['concept'])  

            last_point_ = window_df[window_df['WIN'] == round(threshold)]
            
            if not last_point_.empty:
                last_point.set_offsets(np.c_[last_point_['DIM1'], last_point_['DIM2']])
            fig.canvas.draw_idle()
            
        slider.on_changed(update)
        plt.show()
    
    
    def extract_drift_samples(self, drift_df, intensity_ben, intensity_mal, drift_win):
        '''
        Description:
            This function allows you to extract, based on a drift dataframe, samples
            of a specific intensity in order to meet a required percentage of samples and a drift_win
            Input:
            - drift df
            - benign intensity (in terms of outlier score)
            - malicious intensity (in terms of outlier score)
            - drift_win: number of samples required in the window
        Output:
            - returns the thresholds to be considered for drift samples for benign (index 0) and benign (index 1)
        '''
        intensity_ben = intensity_ben * 2
        intensity_mal = intensity_mal * 2
        intensities = [intensity_ben, intensity_mal]
        drift_samples_ = [] 
        values_ = [drift_df.loc[drift_df['macro_clusters'] == 0, 'anomaly'].value_counts(), 
                   drift_df.loc[drift_df['macro_clusters'] == 1, 'anomaly'].value_counts()]
        thresholds_ = [drift_df.loc[drift_df['macro_clusters'] == 0, 'anomaly'].unique(),
                       drift_df.loc[drift_df['macro_clusters'] == 1, 'anomaly'].unique()]
        for i in range(len(values_)):
            samples_ = []
            intensity = intensities[i]
            
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
        Description:
        This function allows you to check whether a dataframe meets the required percentage
        of benign and malicious samples.
        Input:
        - dataframe
        - percentage of malicious samples
        Output:
        - flag: true if it meets the requirement of at least X%, false otherwise (and thus raises an error)
        '''
        tot = len(dataframe)
        tot_mal = len(dataframe[dataframe['macro_clusters'] == 1])
        print(f'Dimensione concept: {tot}, \nTotale campioni malevoli nel concept: {tot_mal}')
        flag = False
        if (tot_mal / tot) >= perc:
            flag = True
        else:
            while perc >= 0.05:
                perc -= 0.05
                if (tot_mal / tot) >= perc:
                    break
        return perc, flag
    
    def check_drift_samples(self, drift_ben_samples, drift_mal_samples, perc_malignant_drift, drift_win):
        '''
       This function allows you to verify whether:
        - there are enough samples to perform the drift (returns a flag in the first position)
        - the ratio of malignant samples is checked to ensure it matches the required percentage
            as a parameter (it must be at least >= the required value)
        Returns:
        - a first flag indicating that we still have too few samples, so we must decrease the intensity of the benign samples
        - a second flag indicating that the required proportion is not met, so we must decrease the intensity of the malignant samples
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
        concept_mal = concept_df[concept_df['macro_clusters'] == 1].copy()
        concept_ben = concept_df[concept_df['macro_clusters'] == 0].copy()
        drift_mal = drift_df[drift_df['macro_clusters'] == 1].copy()
        drift_ben = drift_df[drift_df['macro_clusters'] == 0].copy()

        drift_mask = np.zeros(win_size, dtype=bool)
        if start_drift < win_size:
            drift_mask[start_drift:] = True
        if recurrent and rec_drift < win_size:
            drift_mask[rec_drift:] = ~drift_mask[rec_drift:]  # XOR per il drift ricorrente

        malign_probs = np.random.rand(win_size)

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
                    used_indices_concept_mal.update(sample.index)
                    concept_mal = concept_mal.loc[~concept_mal.index.isin(used_indices_concept_mal)]
                else:
                    used_indices_concept_ben.update(sample.index)
                    concept_ben = concept_ben.loc[~concept_ben.index.isin(used_indices_concept_ben)]   
            else:
                sample = self.extract_sample(drift_mal, drift_ben, malign_probs[i], perc_malignant_drift)
                if malign_probs[i] < perc_malignant_drift:
                    used_indices_drift_mal.update(sample.index)
                    drift_mal = drift_mal.loc[~drift_mal.index.isin(used_indices_drift_mal)]             
                else:
                    used_indices_drift_ben.update(sample.index)
                    drift_ben = drift_ben.loc[~drift_ben.index.isin(used_indices_drift_ben)]       
            sample_numeric = sample[columns_to_select].values.ravel()
            results.append(np.append(sample_numeric, i).tolist())
        
        window_formatted = results

        # Definiamo le colonne finali, includendo 'macro_clusters', 'concept', e 'WIN'
        self.first_k_columns = concept_df.columns[:self.col_data].tolist()
        new_columns = self.first_k_columns + ['macro_clusters', 'concept', 'WIN']
        
        return window_formatted, new_columns
    
    def reduce_dataset_proportion(self, dataframe, perc):
        perc_neg = perc * 100
        perc_pos = 100 - perc_neg
        tot = len(dataframe)
        tot_mal = len(dataframe[dataframe['macro_clusters'] == 1])
        tot_pos = len(dataframe[dataframe['macro_clusters'] == 0])
        num_pos_to_sample = int((tot_mal * perc_pos ) / perc_neg)

        num_pos_to_sample = min(num_pos_to_sample, tot_pos)
        print(f"Campioni negativi disponibili: {tot_mal}, Campioni positivi disponibili: {tot_pos}")
        print(f"Campioni positivi selezionati: {num_pos_to_sample}, Campioni negativi selezionati: {tot_mal}")
        
        sampled_neg = dataframe[dataframe['macro_clusters'] == 1].sample(n=tot_mal, replace=False)
        sampled_pos = dataframe[dataframe['macro_clusters'] == 0].sample(n=num_pos_to_sample, replace=False)
        
        sampled_df = pd.concat([sampled_neg, sampled_pos])
        print(f"Dataset ridotto con {len(sampled_df)} campioni")
        return sampled_df

    
    def recurrent_drift_generator(self, win_size, start_drift, rec_drift, perc_malignant_concept, perc_malignant_drift, nome_file_parametri, save_stream=False, plot_window=False, filename='streaming.csv', intensity_mode='auto'):
        '''
        Description:
            This function allows you to generate a recurrent sudden drift (of the A-B-A type) from a dataframe passed as input
            to the generator, which must be in the format [dim1, dim2, ..., dimk, macro_clusters, micro_clusters, concept, anomaly].
            Concept samples will be taken from the dataframe where the label is concept=1. Drift samples will be taken from the
            resulting dataframe, based on a specified percentage of malicious samples: if the user asks me to generate a drift window
            drift window of X samples with a percentage of malicious samples of x.x% relative to the total (in statistical terms),
            the generator will select drift samples starting from an intensity threshold of 1.0, decreasing
            incrementally to satisfy the required statistical distribution percentage (for both benign and malicious samples)
            Stream generation then involves random sampling without replacement from the two dataframes.
        Input:
            - win_size: size of the window in which the drift occurs
            - start_drift: when to start the drift
            - rec_drift: when to return to the concept
            - perc_malignant_concept: percentage of malicious samples to be maintained in the concept in statistical terms. It also serves
                as a constraint to be met during the creation of the concept_df; if this constraint is not met, an
                error is thrown with a helpful suggestion for selecting the correct value later; the same percentage is then used
                as the sampling probability during the generation phase.
            - perc_drift_concept: As before, the percentage of malicious samples to be maintained in the drift in statistical terms. It also serves
                as a constraint to be met during the creation of the drift_df; if the proportion is not met, the intensity values are
                resized. The model will attempt to extract the correct intensity values using a greedy strategy
                applied to my dataframe to extract the right number of malicious and benign samples based on sample percentages and window size
                . It will aim to maximize the results by starting with high intensity values
        Output:
            - a summary file of the extracted samples, the window, and the intensity achieved
            - samples extracted in each run
            - a dataframe of the samples extracted in each run
        =============================
        Currently, the intensity is managed automatically. To set it manually, you must pass a tuple
        of the form (benign_intensity, malignant_intensity) as the intensity_mode parameter
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
                perc_drift_reached=perc_mal_reached,
                nome_file=nome_file_parametri
                )

        return samples_win

    
    def sudden_drift_generator(self, win_size, start_drift, perc_malignant_concept, perc_malignant_drift, nome_file_parametri, save_stream=False, plot_window=False, filename='streaming.csv', intensity_mode='auto'):
        '''
        Description:
        This function allows you to generate a sudden drift based on a dataframe passed as input
        to the generator, which must be in the format [dim1, dim2, ..., dimk, macro_clusters, micro_clusters, concept, anomaly].
        The concept samples will be taken from the dataframe where the label is concept=1. The drift samples will be taken from the
        resulting dataframe, based on a specified percentage of malicious samples: if the user asks me to generate a drift window
        drift window of X samples with a percentage of malicious samples of x.x% relative to the total (in statistical terms),
        the generator will select drift samples starting from an intensity threshold of 1.0, decreasing
        incrementally to satisfy the required statistical distribution percentage (for both benign and malicious samples)
        Stream generation then involves random sampling without replacement from the two dataframes.
        Input:
        - win_size: size of the window in which the drift is to occur
        - start_drift: when to start the drift
        - perc_malignant_concept: percentage of malicious samples to be maintained in the concept in statistical terms. It also serves
            as a constraint to be met during the creation of the concept_df; if this constraint is not met, an
            error is thrown with a helpful suggestion for selecting the correct value later; the same percentage is then used
            as the sampling probability during generation.
        - perc_drift_concept: As before, the percentage of malicious samples to be maintained in the drift in statistical terms. It also serves
            as a constraint to be met during the creation of the drift_df; if the proportion is not met, the intensity values are
            resized. The model will attempt to extract the correct intensity values using a greedy strategy
            applied to my dataframe to extract the right number of malicious and benign samples based on sample percentages and window size
            . It will aim to maximize the results by starting with high intensity values
        Output:
        - a summary file of the extracted samples, the window, and the intensity achieved
        - samples extracted in each run
        - a dataframe of the samples extracted in each run
        =============================
        Currently, the intensity is managed automatically. To set it manually, you must pass a tuple
        of the form (benign_intensity, malignant_intensity) as the intensity_mode parameter
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
    
    def simulate_n_samples(self, width_drift, perc_drift, slope):
        '''
        A utility function that allows you to calculate, based on the generated random distribution of samples, how many samples of the concept and how many of the drift to consider
        - count_a: samples of the concept
        - count_b: samples of the drift
        '''
        count_a = 0
        count_b = 0
        for i in range(width_drift):
            perc_a = 1 - (i / width_drift) * slope
            if perc_a < 0.0:
                # If the slope is too steep, I might end up with negative values; in that case, I'll treat 0.0 as the percentage of samples for that concept
                perc_a = 0.0
            if perc_drift[i] < perc_a:
                count_a += 1
            else:
                count_b += 1
        return count_a, count_b
    
    def gradual_drift_generator(self, win_size, start_drift, width_drift, slope, perc_malignant_concept, perc_malignant_drift, save_stream=False, filename='streaming.csv', nome_file_parametri='params_gradual', intensity_mode='auto'):
        '''
        Description:
        This function allows you to generate a gradual drift from a dataframe passed as input
        to the generator, which must be in the format [dim1, dim2, ..., dimk, macro_clusters, micro_clusters, concept, anomaly].
        Samples for the concept will be taken from the dataframe where the label is concept=1.
            Win_size: total size of the dataset
            Start_drift: starting position of the drift
            Width_drift: width of the drift
            Slope: slope of the drift, i.e., how much the distribution must change over its duration
            Perc_malignant_concept: percentage of malignant samples in the concept
            Perc_malignant_drift: percentage of malignant samples in the drift
            Save_stream: flag to save the generated stream
            Filename: name of the output file
            Parameter_file_name: name of the file to save the parameters
            Intensity_mode: intensity setting mode
        '''
        proba_gradual = np.random.rand(width_drift)
        
        gradual_concept, gradual_drift = self.simulate_n_samples(width_drift=width_drift, perc_drift=proba_gradual, slope=slope)

        ###  First, I make sure to adjust the dataset to the required proportion of malicious samples; otherwise, I risk not having enough samples to generate the drift
        concept_df = self.df[self.df['concept'] == 1].copy(deep=True)
        concept_df = self.reduce_dataset_proportion(dataframe=concept_df, perc=perc_malignant_concept)
        print(f'Concept df size: {len(concept_df)}')
        
        if len(concept_df) < (start_drift + gradual_concept):
            raise ValueError('Attenzione! Dimensione concetto insufficiente per coprire la finestra richiesta')
        
        drift_df = self.df[self.df['concept'] == 0].copy(deep=True)
        print(f'Drift size: {len(drift_df)}')
        
        if len(drift_df) < (win_size - (start_drift + width_drift) + gradual_drift):
            raise ValueError('Attenzione! Dimensione drift insufficiente per coprire la finestra richiesta')
        
        intensity_ben = 1.0
        intensity_mal = 1.0
        decrease_intensity = True
        
        ### This is the window that can be reached, since I'm going to add the gradual portion to the small portion of actual drift
        drift_win = win_size - (start_drift + width_drift) + gradual_drift
        
        if intensity_mode == 'auto':
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
            
        # print("For debugging only")
        # print(f'Intensità benevoli: {intensity_ben}\nIntesità malevoli: {intensity_mal}')
        # print(f'Gradual concept samples: {gradual_concept}, gradual drift samples: {gradual_drift}')
        
        samples_win, new_cols = self.generate_samples_gradual(concept_df=concept_df, drift_df=drift_df[drift_df['in_win'] == 1], win_size=win_size, start_drift=start_drift, width_drift=width_drift, perc_malignant_concept=perc_malignant_concept, perc_malignant_drift=perc_malignant_drift, proba_gradual=proba_gradual, slope=slope)
        win_df = pd.DataFrame(samples_win, columns=new_cols)
        perc_mal_reached_concept = len(win_df[(win_df['macro_clusters'] == 1) & (win_df['concept'] == 1)]) / len(win_df[win_df['concept'] == 1])
        perc_mal_reached_drift = len(win_df[(win_df['macro_clusters'] == 1) & (win_df['concept'] == 0)]) / len(win_df[win_df['concept'] == 0])
        perc_mal_reached = [perc_mal_reached_concept, perc_mal_reached_drift]
        
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
    
    def generate_samples_gradual(self, concept_df, drift_df, win_size, start_drift, width_drift, perc_malignant_concept, perc_malignant_drift, proba_gradual, slope):
        """
        Generates optimized samples using parallelization and batch extraction for gradual drift.
        - Start drift indicate the sample index where the gradual must start
        - Width determine for how long drift must gradually shift
        - Slope determine the rapidity of shifting
        """
        print(f'campioni nel concetto: {len(concept_df)}, campioni nel drift: {len(drift_df)}')
        
        concept_mal = concept_df[concept_df['macro_clusters'] == 1].copy()
        concept_ben = concept_df[concept_df['macro_clusters'] == 0].copy()
        drift_mal = drift_df[drift_df['macro_clusters'] == 1].copy()
        drift_ben = drift_df[drift_df['macro_clusters'] == 0].copy()
        
        print(f'campioni maligni nel concetto: {len(concept_mal)}, campioni benevoli nel concetto: {len(concept_ben)}')
        print(f'campioni maligni nel drift: {len(drift_mal)}, campioni benevoli nel drift: {len(drift_ben)}')
        
        # We determine in advance where the drift occurs, and then mask the samples that should be extracted from the concept and those that should be extracted from the drift based on the generated random distribution and the chosen slope
        drift_mask = np.zeros(win_size, dtype=bool)
        for i in range(width_drift):
            gradual_perc = 1 - (i / width_drift) * slope
            if proba_gradual[i] < gradual_perc:
                drift_mask[start_drift + i] = False
            else:
                drift_mask[start_drift + i] = True
       
        drift_mask[start_drift + width_drift:] = True
    
        malign_probs = np.random.rand(win_size)

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
                    used_indices_concept_mal.update(sample.index)
                    concept_mal = concept_mal.loc[~concept_mal.index.isin(used_indices_concept_mal)]             
                else:
                    used_indices_concept_ben.update(sample.index)
                    concept_ben = concept_ben.loc[~concept_ben.index.isin(used_indices_concept_ben)]                 
            else:
                sample = self.extract_sample(drift_mal, drift_ben, malign_probs[i], perc_malignant_drift)
                if malign_probs[i] < perc_malignant_drift:
                    used_indices_drift_mal.update(sample.index)
                    drift_mal = drift_mal.loc[~drift_mal.index.isin(used_indices_drift_mal)]             
                else:
                    used_indices_drift_ben.update(sample.index)
                    drift_ben = drift_ben.loc[~drift_ben.index.isin(used_indices_drift_ben)]      
            sample_numeric = sample[columns_to_select].values.ravel()
            results.append(np.append(sample_numeric, i).tolist())
            
        window_formatted = results
        self.first_k_columns = concept_df.columns[:self.col_data].tolist()
        new_columns = self.first_k_columns + ['macro_clusters', 'concept', 'WIN']
        return window_formatted, new_columns
    
    def load_df_incremental(self, filename):
        data_path = Path(f'{self.directory_stream}/{self.clustering_technique}')
        file_path = data_path / filename
        df = pd.read_csv(file_path)
        return df
    
    def incremental_drift_generator(self, win_size, list_starts, concept_df_incremental_filename, spatial_biases,
                                    nome_file_parametri='params_incremental', save_stream=False, filename='streaming_incremental.csv', intensity_mode='auto'):
        '''
        Description:
            Generates an incremental drift with N concepts (A, B, C, ...) that alternate
            in sequence according to a start list.
            Each concept has its own spatial bias, that is, the proportion of
            malicious/benign samples to be maintained during its active period.
            Transitions between adjacent concepts constitute drift windows,
            which are handled using the same intensity logic as sudden/recurrent drift.

        Input:
            - win_size: total window size
            - list_starts: list of N start points [s_A, s_B, s_C, ...], one for each concept.
                           Concept i is active in the interval [list_starts[i], list_starts[i+1]).
                           The last concept is active up to win_size. Ideally, s_A starts at 0.
            - spatial_biases: a list of N floats in [0,1], one for each concept,
                              indicating the desired percentage of malicious samples
                              for that concept during its active period.
            - concept_df_incremental_filename: name of the dataframe with a ‘concept’ column in {A, B, C, ...}
                                      generated by incremental_generator
            - intensity_mode: ‘auto’ or a tuple (intensity_ben, intensity_mal) applied
                              to all drift transitions

        Output:
            - results: list of generated samples
            - saves the stream and parameters if save_stream=True
        '''
        n_concepts = len(list_starts)
        concept_df_incremental = self.load_df_incremental(filename=concept_df_incremental_filename)
        concept_labels = np.unique(concept_df_incremental.loc[:,'concept']) # ['A','B','C',...]

        if len(spatial_biases) != n_concepts:
            raise ValueError('spatial_biases deve avere tanti elementi quanti sono i concetti')
        if any(b < 0.0 or b > 1.0 for b in spatial_biases):
            raise ValueError('spatial_biases deve contenere valori in [0, 1]')
        if any(list_starts[i] >= list_starts[i+1] for i in range(n_concepts - 1)):
            raise ValueError('list_starts deve essere strettamente crescente')
        if list_starts[0] < 0 or list_starts[-1] >= win_size:
            raise ValueError('list_starts fuori dai limiti della finestra')

        concept_lengths = {}
        for idx in range(n_concepts):
            start = list_starts[idx]
            end = list_starts[idx + 1] if idx + 1 < n_concepts else win_size
            concept_lengths[concept_labels[idx]] = end - start

        ## creation of sub dataframes with spatial bias specified (per_mal) for each concept
        concept_dfs = {}
        for i, label in enumerate(concept_labels):
            perc_mal = spatial_biases[i]
            sub = concept_df_incremental[concept_df_incremental['concept'] == label].copy(deep=True)
            sub = self.reduce_dataset_proportion(dataframe=sub, perc=perc_mal)
            concept_dfs[label] = {
                'mal': sub[sub['macro_clusters'] == 1].copy(),
                'ben': sub[sub['macro_clusters'] == 0].copy(),
                'perc_mal': perc_mal
            }
            print(f'Concept {label} — size dopo riduzione: {len(sub)}')

        ### check if there are enough samples for each concept
        for label in concept_labels:
            n_timesteps  = concept_lengths[label]
            perc_mal     = concept_dfs[label]['perc_mal']
            n_mal_needed = int(np.ceil(n_timesteps * perc_mal))
            n_ben_needed = n_timesteps - n_mal_needed
            n_mal_avail  = len(concept_dfs[label]['mal'])
            n_ben_avail  = len(concept_dfs[label]['ben'])

            print(f'\nConcetto {label}  (timestep: {n_timesteps}, perc_mal={perc_mal:.2f}):')
            print(f'  Malevoli — richiesti: {n_mal_needed:>5} | disponibili: {n_mal_avail:>5}  {"OK" if n_mal_avail >= n_mal_needed else "WARN"}')
            print(f'  Benigni  — richiesti: {n_ben_needed:>5} | disponibili: {n_ben_avail:>5}  {"OK" if n_ben_avail >= n_ben_needed else "WARN"}')

            if n_mal_avail < n_mal_needed:
                raise ValueError(
                    f'Campioni malevoli insufficienti per il concetto {label}: '
                    f'richiesti {n_mal_needed}, disponibili {n_mal_avail}. '
                    f'Riduci win_size, modifica list_starts o aumenta il dataset sorgente.'
                )
            if n_ben_avail < n_ben_needed:
                raise ValueError(
                    f'Campioni benigni insufficienti per il concetto {label}: '
                    f'richiesti {n_ben_needed}, disponibili {n_ben_avail}. '
                    f'Riduci win_size, modifica list_starts o aumenta il dataset sorgente.'
                )
            
        concept_mask = np.zeros(win_size, dtype=int)
        for idx in range(n_concepts):
            start = list_starts[idx]
            end = list_starts[idx + 1] if idx + 1 < n_concepts else win_size
            concept_mask[start:end] = idx

        columns_to_select = list(concept_df_incremental.columns[:self.col_data]) + ['macro_clusters', 'concept']
        used_indices = {label: {'mal': set(), 'ben': set()} for label in concept_labels}
    
        malign_probs = np.random.rand(win_size)

        results = []
        for i in tqdm(range(win_size), desc='Generazione stream incrementale'):
            active_idx   = concept_mask[i]
            active_label = concept_labels[active_idx]
            perc_mal     = concept_dfs[active_label]['perc_mal']

            df_mal = concept_dfs[active_label]['mal']
            df_ben = concept_dfs[active_label]['ben']

            sample = self.extract_sample(df_mal, df_ben, malign_probs[i], perc_mal) # estrazione dei campioni specificando i dataset da utilizzare

            # aggiorna pool del concetto attivo senza reinserimento
            if malign_probs[i] < perc_mal:
                used_indices[active_label]['mal'].update(sample.index)
                concept_dfs[active_label]['mal'] = df_mal.loc[
                    ~df_mal.index.isin(used_indices[active_label]['mal'])
                ]
            else:
                used_indices[active_label]['ben'].update(sample.index)
                concept_dfs[active_label]['ben'] = df_ben.loc[
                    ~df_ben.index.isin(used_indices[active_label]['ben'])
                ]

            sample_values = sample[columns_to_select].values.ravel()
            results.append(np.append(sample_values, i).tolist())

        self.first_k_columns = concept_df_incremental.columns[:self.col_data].tolist()
        new_columns = self.first_k_columns + ['macro_clusters', 'concept', 'WIN']
        win_df = pd.DataFrame(results, columns=new_columns)

        perc_mal_reached = []
        for label in concept_labels:
            sub = win_df[win_df['concept'] == label]
            p = len(sub[sub['macro_clusters'] == 1]) / len(sub) if len(sub) > 0 else 0.0
            perc_mal_reached.append(p)
            target = concept_dfs[label]['perc_mal']
            print(f'Concetto {label}: perc_mal target={target:.3f} | raggiunta={p:.3f} | Δ={abs(p - target):.3f}')

        if save_stream:
            self.save_stream(win_df, filename)
            data_path   = Path(f'{self.directory_stream}/{self.clustering_technique}')
            data_path.mkdir(parents=True, exist_ok=True)
            params_path = data_path / f'{nome_file_parametri}.txt'
            with open(params_path, 'w') as f:
                print(f'Tipo drift: incrementale con {n_concepts} concetti', file=f)
                print(f'Win size: {win_size}', file=f)
                print(f'List starts: {list_starts}', file=f)
                print(f'Concept labels: {concept_labels}', file=f)
                for label, p, bias in zip(concept_labels, perc_mal_reached, spatial_biases):
                    print(f'Concetto {label}:', file=f)
                    print(f'  spatial_bias (perc_mal target) : {bias:.3f}', file=f)
                    print(f'  perc_mal raggiunta             : {p:.3f}', file=f)

        return results
        