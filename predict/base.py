from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from model_saver.save import save_model


def base_predict_and_save_results(df_train, save_name, kmeans_clusters=82, db_eps=0.5, db_min_samples=5):
    df_train_num = df_train.select_dtypes('number')
    #DBSCAN
    dbscan = DBSCAN(eps=db_eps, min_samples=db_min_samples).fit(df_train_num)
    dbscan_predictions = dbscan.fit_predict(df_train_num)

    df_results = df_train_num
    df_results['cluster_label_dbscan'] = dbscan_predictions
    #KMeans
    kmeans = KMeans(n_clusters=kmeans_clusters, n_init="auto").fit(df_train_num)
    kmeans_predictions = kmeans.predict(df_train_num)
    df_results['cluster_label_kmeans'] = kmeans_predictions

    #Spectral Clustering
    spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors')
    spectral_predictions = spectral.fit_predict(df_train_num)
    df_results['cluster_label_spectral'] = spectral_predictions

    #Hierarchical Clustering
    connectivity = kneighbors_graph(df_train_num, n_neighbors=10, include_self=False)
    agg_clustering = AgglomerativeClustering(n_clusters=3, connectivity=connectivity)
    hierarchical_predictions = agg_clustering.fit_predict(df_train_num)
    df_results['cluster_label_hierarchical'] = hierarchical_predictions

    models_df = {
        'df': df_results,
        'kmeans_model': kmeans,
        'dbscan_model': dbscan,
        'spectral_model': spectral,
        'hierarchical_model': agg_clustering
    }

    save_model(models_df, save_name)