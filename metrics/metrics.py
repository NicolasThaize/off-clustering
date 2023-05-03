import os
import pandas as pd
import sys
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def plot_evaluation_metrics(df, model):
    """
    Calcule et affiche les métriques d'évaluation pour un modèle de clustering donné.

    Parameters:
    df (pd.DataFrame): DataFrame contenant les données d'entrée
    model: Un modèle de clustering entraîné avec des étiquettes (par exemple, KMeans, AgglomerativeClustering)

    Returns:
    dict: Un dictionnaire contenant les métriques d'évaluation calculées

    Examples:
    >>> metrics = plot_evaluation_metrics(data, model)
    Coefficient de silhouette : KMeans(n_clusters=2) 0.5
    Indice Calinski-Harabasz : KMeans(n_clusters=2) 7.5
    Indice Davies-Bouldin : KMeans(n_clusters=2) 1.2
    """
    
    # Calculer les métriques d'évaluation
    silhouette_avg = silhouette_score(df, model.labels_)
    calinski_harabasz = calinski_harabasz_score(df, model.labels_)
    davies_bouldin = davies_bouldin_score(df, model.labels_)

    metrics = {
    'silhouette': silhouette_avg,
    'calinski_harabasz': calinski_harabasz,
    'davies_bouldin': davies_bouldin
}

    return metrics

def save_metrics_to_excel(iteration_name, metrics):
    """
    Écrit les métriques dans un fichier Excel nommé "metrics.xlsx". Si le fichier existe déjà, les nouvelles données
    sont ajoutées à la suite. Si le fichier n'existe pas, il est créé.

    Parameters:
    iteration_name (str): Le nom de l'itération à ajouter
    metrics (dict): Un dictionnaire contenant les métriques à ajouter

    Returns:
    None

    Examples:
    >>> save_metrics_to_excel(iteration_name='', metrics=plot_evaluation_metrics(df, model)
    """
    file_name = 'metrics.xlsx'
    iteration_name = input("Entrez le nom de l'itération ou 'esc' pour annuler : ")
    if iteration_name.lower() == "esc":
        return

    new_metrics = pd.DataFrame.from_dict(metrics, orient='index', columns=[iteration_name]).T

    if os.path.exists(file_name):
        df_metrics = pd.read_excel(file_name, index_col=0, engine='openpyxl')
        df_metrics = df_metrics._append(new_metrics, ignore_index=False)
    else:
        df_metrics = new_metrics

    # Écrire "Iteration name" en A1
    df_metrics.index.name = "Iteration name"

    # Écrire les données dans un fichier Excel avec xlsxwriter
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    df_metrics.to_excel(writer, index=True, header=True)
    worksheet = writer.sheets['Sheet1']

    # Ajuster la largeur des colonnes A, B, C et D
    for col_idx, col_letter in enumerate(['A', 'B', 'C', 'D'], 1):
        worksheet.set_column(f'{col_letter}:{col_letter}', 20)  # 20 est la nouvelle largeur de la colonne

    # Enregistrer les modifications
    writer._save()



