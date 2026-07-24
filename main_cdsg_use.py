"""
Complete example of how to use CDSG.

This main file performs:

1. Concept dataset generation
2. Recurrent sudden drift generation
3. Sudden drift generation
4. Gradual drift generation

IMPORTANT
---------
The concept dataset is generated only once.

Afterwards, the three streams are generated separately
starting from the same already-generated concept dataset.

Expected project structure:

main_folder/
│
├── main.py
├── cdsg.py
├── concept_generator.py
├── streamer_generator.py
├── utils.py
│
├── datasets/
│   └── source_dataset.csv
│
└── results_cdsg_directory/
"""

from cdsg import CDSG


if __name__ == "__main__":

    # Directory containing the source dataset.
    #
    # Example:
    #   datasets/source_dataset.csv
    directory_name = "datasets"

    # The file must be located at: datasets/source_dataset.csv
    #
    # The dataset must contain columns compatible with  the preprocessing implemented in the CDSG class.
    filename = "source_dataset.csv"

    # Main directory where CDSG will save:
    #
    # - the generated concept dataset
    # - drift parameters
    # - the final streams
    directory_stream = "results_cdsg_directory"

    # Clustering technique used to generate the concepts.
    # Examples supported by the project: kmeans, dec, agglomerative
    # The technique must be compatible with both
    # ConceptGenerator and StreamerGen.
    clustering_technique = "kmeans"

    ##### define now the concept creation parameters
    # Number of macro-clusters: Each macro-cluster represents a large region/concept of the dataset.
    #   n_macro_clusters = 2 means that the dataset is organized into 2 macro-concepts.
    n_macro_clusters = 2

    # Number of micro-clusters associated with each macro-cluster.
    # With: [6, 6] the first macro-cluster is divided into 6 micro-clusters and the second macro-cluster is divided into 6 micro-clusters.
    # IMPORTANT:
    # the length of the list must be consistent with n_macro_clusters.
    list_micro_clusters = [6, 6]

    # Percentage of negative/anomalous samples used during concept generation.
    # Example: perc_neg = 0.4 means 40% negative/anomalous samples.
    perc_neg = 0.4

    # Number of samples extracted from each partition/concept.
    # Example: [2, 2] specifies the number of samples used for the two components associated with the macro-clusters.
    list_samples = [2, 2]


    ### streaming parameters
    # Spatial bias: k represents the proportion of benign samples in the stream.
    # The proportion of malicious/anomalous samples is internally computed as: malignant_k = 1 - k
    # Example:
    #   k = 0.7
    # means:
    #   70% benign samples
    #   30% malicious samples
    k = 0.7
    intensity = "auto"



    cdsg = CDSG(
        directory_name=directory_name,
        filename=filename,
        directory_stream=directory_stream
    )

    cdsg.run_cdsg(
        _runcg=True,
        _run_ds=True,
        n_macro_clusters=n_macro_clusters,
        list_micro_clusters=list_micro_clusters,
        perc_neg=perc_neg,
        list_samples=list_samples,
        k=k,
        drift_type="sudden",
        drift_temporal_annotations=(200_000, 120_000), ### for sudden drift: [Size, Start]
        clustering_technique=clustering_technique,
        intensity=intensity
    )


    #### if instead the one2many approach for creating concepts, the mode_classifier parameter must be specified.
    # 2 values are considered: m2m or o2m
    cdsg.run_cdsg(
        _runcg=True,
        _run_ds=True,
        n_macro_clusters=n_macro_clusters,
        list_micro_clusters=list_micro_clusters,
        perc_neg=perc_neg,
        list_samples=list_samples,
        k=k,
        drift_type="sudden",
        drift_temporal_annotations=(200_000, 120_000), ### for sudden drift: [Size, Start]
        clustering_technique=clustering_technique,
        intensity=intensity,
        mode_classifier='o2m'
    )



    # A recurrent sudden drift represents an abrupt change that can subsequently reappear over time.
    # FORMAT: [win_size, start_drift, rec_drift]
    # win_size: Total stream length.
    # start_drift: Sample index at which the drift starts.
    # rec_drift: Indicates the point/interval associated with the recurrence of the drift.
    #
    # EXAMPLE: [200_000, 80_000, 120_000]
    # means:
    #   total stream: 200,000 samples
    #   initial drift: sample 80,000
    #   recurrence: sample 120,000

    cdsg.run_cdsg(
        _runcg=False,
        _run_ds=True,
        n_macro_clusters=n_macro_clusters,
        list_micro_clusters=list_micro_clusters,
        perc_neg=perc_neg,
        list_samples=list_samples,
        k=k,
        drift_type="recurrent_sudden",
        drift_temporal_annotations=[200_000, 80_000, 120_000],
        clustering_technique=clustering_technique,
        intensity=intensity
    )



    # A gradual drift represents a progressive change.
    # FORMAT: [win_size, start_drift, width_drift]
    # win_size: Total stream length.
    # start_drift: Point at which the transition starts.
    # width_drift: Duration of the gradual transition.
    # EXAMPLE: [200_000, 80_000, 40_000]
    # means:
    #
    #   total stream: 200,000 samples
    #   drift starts: sample 80,000
    #   transition duration: 40,000 samples
    #
    # Therefore, the transition occurs progressively
    # over the interval:
    #
    #   80,000 -> 120,000

    # slope controls the steepness of the gradual drift.
    # slope = 1.0: Standard transition.
    # slope > 1.0: Steeper transition. 
    # slope < 1.0: Slower transition.
    slope = 1.0

    cdsg.run_cdsg(
        _runcg=False,
        _run_ds=True,
        n_macro_clusters=n_macro_clusters,
        list_micro_clusters=list_micro_clusters,
        perc_neg=perc_neg,
        list_samples=list_samples,
        k=k,
        drift_type="gradual",
        drift_temporal_annotations=[200_000,80_000,40_000],
        clustering_technique=clustering_technique,
        intensity=intensity,        
        slope=slope 
    )
## in the case of icnremental drift, must be specified the partitions AFTER the first concept creation and partition between macro and micro clusters
cdsg.incremental_concept_generator(
    list_micro_clusters_incremental=[[2,2], [2,2], [2,2]],  # [x,y], where x are POS partitions (Benignant) and y are NEG partitions (Malicious) 
    clustering_technique='kmeans',
)