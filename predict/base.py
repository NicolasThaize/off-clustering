from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from model_saver.save import save_model
from metrics.metrics import plot_evaluation_metrics, save_metrics_to_excel
from predict.kmeans import kmeans


def base_predict_and_save_results(df_train, save_name, iteration_nb, kmeans_clusters=82, db_eps=0.5, db_min_samples=5):
    df_train_num = df_train.select_dtypes('number')
    #DBSCAN
    dbscan = DBSCAN(eps=db_eps, min_samples=db_min_samples).fit(df_train_num)
    dbscan_predictions = dbscan.fit_predict(df_train_num)

    #KMeans
    kmeans = KMeans(n_clusters=kmeans_clusters, n_init="auto").fit(df_train_num)
    kmeans_predictions = kmeans.predict(df_train_num)

    models_df = {
        'df': df_train_num,
        'kmeans_model': kmeans,
     #   'dbscan_model': dbscan
    }

    save_metrics_to_excel(iteration_name=f'kmeans_{iteration_nb}', metrics=plot_evaluation_metrics(df_train_num, kmeans))
    #save_metrics_to_excel(iteration_name=f'dbscan_{iteration_nb}', metrics=plot_evaluation_metrics(df_train_num, dbscan))

    save_model(models_df, save_name)

def base_tuning_kmeans(df, save_name, iteration_nb):
    model = kmeans(df, 21)
    value = {
        "df": df,
        "kmeans_model": model
    }
    save_metrics_to_excel(iteration_name=f'kmeans_tuning_{iteration_nb}', metrics=plot_evaluation_metrics(df, model))
    save_model(value, save_name)

def base_spec_hierarchical(df, save_name, iteration_nb):
    # Spectral Clustering
    spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors')
    spectral_predictions = spectral.fit_predict(df)

    # Hierarchical Clustering
    connectivity = kneighbors_graph(df, n_neighbors=10, include_self=False)
    agg_clustering = AgglomerativeClustering(n_clusters=3, connectivity=connectivity)
    hierarchical_predictions = agg_clustering.fit_predict(df)

    save_metrics_to_excel(iteration_name=f'spectral_{iteration_nb}', metrics=plot_evaluation_metrics(df, spectral))
    save_metrics_to_excel(iteration_name=f'hierarchical_{iteration_nb}', metrics=plot_evaluation_metrics(df, agg_clustering))

    models_df = {
        'df': df,
        'spectral_model': spectral,
        'hierarchical_model': agg_clustering
    }

    save_model(models_df, save_name)
