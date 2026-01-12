#### fare un esempio di main funzionamente da cdsg.
from cdsg import CDSG

if __name__ == "__main__":
    '''
    In this example, we demonstrate how to use the CDSG module from REFINE framework.
    In particular, we suppose to have already a source dataset in the datasets/ folder.

    We will generate a concept dataset and then a data stream with both sudden and recurrent drifts,
    starting from the same concept dataset.

    Intensity will be set to 'auto', so that the framework will manage it automatically. 
    Moreover, spatial bias is set to 0.7, that means that 70% of benign samples will be present in the stream. 
    Feel free to change these parameters as you wish, but remember to adapt also the drift_temporal_annotations accordingly.

    The generated datasets will be saved in the results_cdsg_directory/ folder.
    ##############################
    '''
    refine_cdsg_sudden = CDSG(directory_name='datasets', filename='source_dataset.csv', directory_stream='results_cdsg_directory')
    refine_cdsg_sudden.run_cdsg(
        _runcg=True,
        _run_ds=True,
        n_macro_clusters=2,
        list_micro_clusters=[6,6],
        perc_neg=0.4,
        list_samples=[2,2],
        k=0.7,
        drift_type='sudden',
        drift_temporal_annotations=(200_000, 120_000), ### drift starts at 120k_th sample, window size is 200k samples
        intensity='auto'
    )
    ##############################
    refine_cdsg_recurrent = CDSG(directory_name='datasets', filename='source_dataset.csv', directory_stream='results_cdsg_directory')
    refine_cdsg_recurrent.run_cdsg(
        _runcg=False,
        _run_ds=True,
        n_macro_clusters=2,
        list_micro_clusters=[6,6],
        perc_neg=0.4,
        list_samples=[2,2],
        k=0.7,
        drift_type='recurrent_sudden',
        drift_temporal_annotations=(200_000, 80_000, 120_000), ### drift starts at 80k_th sample, window size is 200k samples, recurrence starts at 120k_th sample
        intensity='auto'
    )