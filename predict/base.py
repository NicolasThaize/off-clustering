from sklearn.cluster import KMeans, DBSCAN
from model_saver.save import save_model


def base_predict_and_save_results(df_train, save_name, kmeans_clusters=82, db_eps=0.5, db_min_samples=5):
    df_train_num = df_train.select_dtypes('number')

    dbscan = DBSCAN(eps=db_eps, min_samples=db_min_samples).fit(df_train_num)
    dbscan_predictions = dbscan.fit_predict(df_train_num)

    df_results = df_train_num
    df_results['cluster_label_dbscan'] = dbscan_predictions

    kmeans = KMeans(n_clusters=kmeans_clusters, n_init="auto").fit(df_train_num)
    kmeans_predictions = kmeans.predict(df_train_num)
    df_results['cluster_label_kmeans'] = kmeans_predictions

    models_df = {
        'df': df_results,
        'kmeans_model': kmeans,
        'dbscan_model': dbscan
    }

    save_model(models_df, save_name)
