from cdsg import CDSG


if __name__ == "__main__":

    # ============================================================
    # RECURRENT SUDDEN DRIFT WITH CICIDS
    # ============================================================

    refine_cdsg_cicids = CDSG(
        directory_name='datasets',
        filename='cleaned_cicids.csv',
        directory_stream='results_cdsg_directory_cicids'
    )

    refine_cdsg_cicids.run_only_cg(
        n_macro_clusters=2,
        list_micro_clusters=[6, 6],
        perc_neg=0.4,
        list_samples=[2, 2],
        clustering_technique='kmeans'
    )

    refine_cdsg_cicids = CDSG(
        directory_name='datasets',
        filename='cleaned_cicids.csv',
        directory_stream='results_cdsg_directory_cicids'
    )
    
    refine_cdsg_cicids.run_only_ds(
        k=0.7,
        drift_type='recurrent_sudden',
        drift_temporal_annotations=[200_000, 100_000, 150_000], ### [win_size, start_drift, rec_drift]
        clustering_technique='kmeans'
    )

    ### example using o2m clsutering technique

    refine_cdsg_o2m_kmeans = CDSG(
        directory_name='datasets',
        filename='cleaned_cicids.csv',
        directory_stream='results_cdsg_directory_cicids'
    )

    refine_cdsg_o2m_kmeans.run_only_cg(
        n_macro_clusters=2,
        list_micro_clusters=[6, 6],
        perc_neg=0.4,
        list_samples=[2, 2],
        clustering_technique='kmeans',
        mode_classifier='o2m'
    )

    refine_cdsg_o2m_dec = CDSG(
        directory_name='datasets',
        filename='cleaned_cicids.csv',
        directory_stream='results_cdsg_directory_cicids'
    )

    refine_cdsg_o2m_dec.run_only_cg(
        n_macro_clusters=2,
        list_micro_clusters=[6, 6],
        perc_neg=0.4,
        list_samples=[2, 2],
        clustering_technique='dec',
        mode_classifier='o2m'
    )


    # ============================================================
    # SUDDEN DRIFT WITH BCCC-CPACKET
    # ============================================================

    refine_cdsg_cpacket_kmeans = CDSG(
        directory_name='datasets',
        filename='cleaned_bccc_cpacket.csv',
        directory_stream='results_cdsg_directory_bccc_cpacket'
    )

    refine_cdsg_cpacket_kmeans.run_only_cg(
        n_macro_clusters=2,
        list_micro_clusters=[6, 6],
        perc_neg=0.4,
        list_samples=[3, 3],
        clustering_technique='kmeans',
        _skip_micro=True
    )

    refine_cdsg_cpacket_kmeans.run_only_ds(
        k=0.5,
        drift_type='sudden',
        drift_temporal_annotations=[100_000, 50_000], ### [win_size, start_drift]
        clustering_technique='kmeans'
    )

    refine_cdsg_cpacket_dec = CDSG(
        directory_name='datasets',
        filename='cleaned_bccc_cpacket.csv',
        directory_stream='results_cdsg_directory_bccc_cpacket'
    )

    refine_cdsg_cpacket_dec.run_only_cg(
        n_macro_clusters=2,
        list_micro_clusters=[6, 6],
        perc_neg=0.4,
        list_samples=[4, 3],
        clustering_technique='dec',
        _skip_micro=True
    )

    refine_cdsg_cpacket_dec.run_only_ds(
        k=0.5,
        drift_type='sudden',
        drift_temporal_annotations=[100_000, 50_000], ### [win_size, start_drift]
        clustering_technique='dec'
    )


    # ============================================================
    # INCREMENTAL DRIFT
    # ============================================================

    refine_cdsg_incremental = CDSG(
        directory_name='datasets',
        filename='cleaned_bccc_cpacket.csv',
        directory_stream='results_cdsg_directory_bccc_cpacket'
    )

    refine_cdsg_incremental.run_only_ds(
        k=0.7,
        drift_type='incremental',
        drift_temporal_annotations=[160_000, [0, 80_000, 120_000]], ## [win_size, list_starts]
        clustering_technique='kmeans',
        spatial_biases_list=[0.8, 0.7, 0.3]
    )


    # ============================================================
    # GRADUAL DRIFT WITH X-IIoTID
    # ============================================================

    refine_cdsg_xiiot = CDSG(
        directory_name='datasets',
        filename='X-IIoTID-cleaned.csv',
        directory_stream='results_cdsg_directory_xiiot'
    )

    refine_cdsg_xiiot.run_only_cg(
        n_macro_clusters=2,
        list_micro_clusters=[3, 6],
        perc_neg=0.4,
        list_samples=[2, 2],
        clustering_technique='dec'
    )


    refine_cdsg_xiiot.run_only_ds(
        k=0.5,
        drift_type='gradual',
        drift_temporal_annotations=[120_000, 60_000, 30_000], ## [win_size, start_drift, width_drift]
        clustering_technique='dec',
        slope=2.0
    )