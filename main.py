from teaching_ml_2023.scripts.data_loader import *
from teaching_ml_2023.scripts.Filtering_TP import Filtering # Tâche 0.0
from teaching_ml_2023.scripts.Preprocessing_TP import Preprocessing # Tâche 1.0
from teaching_ml_2023.scripts.Scaling_TP import Scaling # Tâche 2.0
from teaching_ml_2023.scripts.OutliersManaging_TP import OutliersManaging # Tâche 3.0
from encoding.binary_encoding import binary_encode
from predict.base import base_predict_and_save_results, base_tuning_kmeans, base_spec_hierarchical

def filtering(object, verbose=False):
    """Filter dataframe
    Returns:
        df (DataFrame): return filtered pandas dataframe
    @Author: Thomas PAYAN
    """
    if verbose:    
        print("\nFiltering dataframe ...")

    ft_endswith = object.get_features_endswith(endswith=object.endswith)
    object.drop_features(ft_endswith)

    ft_wendswith = object.get_features_endswith(endswith=object.wendswith, invert=True)
    object.drop_features(ft_wendswith)

    return object.df

def preprocessing(object, verbose=False):
    """Preprocess dataframe
    Returns:
        df (DataFrame): return preprocessed dataframe
    @Author: Thomas PAYAN
    """
    if verbose:    
        print("\nPreprocessing dataframe ...")
        
    object.drop_duplicated_values()

    object.drop_missing_values()

    object.impute_missing_values()

    # object.categorical_features_encoding()

    return object.df

def scaling(object, verbose=False):
    """Scale dataframe
    Returns:
        df (DataFrame): return scaled dataframe
    @Author: Thomas PAYAN
    """
    if verbose:    
        print("\nScaling dataframe ...")

    object.convert_categorical_features_to_numeric()

    object.scaling_features()

    return object.df

def outliers_managing(object, ft_exclude=[], endswith=None, verbose=False):
        """Managing outliers in pandas dataframe
        Returns:
            df (DataFrame): return managed dataframe
        @Author: Thomas PAYAN
        """
        if verbose:    
            print("\nOutliers managing dataframe ...")

        object.df = object.correct_features_100g(ft_exclude)

        # df_100g = object.get_features_endswith(endswith, ft_exclude) # Select features list
        # tukey_outliers = object.tukey_outliers(df_100g.columns.tolist()) # Detect outliers

        # object.df.loc[tukey_outliers] # Show the ouliers rows
        # object.df.drop(tukey_outliers, inplace=True) # Drop outliers
    
        return object.df

if __name__ == "__main__":
    # Load OpenFoodFact dataset
    file_path = r"teaching_ml_2023/data/en.openfoodfacts.org.products.csv"
    nrows     = 10000
    df_train  = get_data(file_path, nrows, True)

    # Execute filtering
    endswith      = ["_t","_datetime","_url","_en"]
    wendswith     = ["_tags"]
    obj_filtering = Filtering(df_train, endswith, wendswith)
    df_train      = filtering(obj_filtering, verbose=True)
    print(df_train.head())

    ft_delete = ["code","url","creator","last_modified_by"]
    obj_filtering.drop_features(ft_delete)
    print(df_train.head())

    # Excute preprocessing
    percent             = 70
    num_imput           = 'knn'  # Numerical features imputation method
    cat_imput           = 'mode' # Categorical features imputation method
    label_encode_method = 'code' # Label encoding method
    obj_preprocessing   = Preprocessing(df_train, percent, num_imput, cat_imput, label_encode_method)
    df_train            = preprocessing(obj_preprocessing, verbose=True)
    print(df_train.head())

    # Execute outliers managing
    endswith   = "_100g"
    ft_exclude = [
                    'energy-kj_100g',
                    'energy-kcal_100g',
                    'ph_100g',
                    'carbon-footprint_100g',
                    'nutrition-score-fr_100g',
                    'nutrition-score-uk_100g'
                ]
    obj_outliers_managing = OutliersManaging(df_train)

    df_train = outliers_managing(obj_outliers_managing, ft_exclude, endswith, verbose=True)
    print(df_train.head())

    # Execute scaling
    #method      = 'min_max'
    #obj_scaling = Scaling(df_train, method)

    #df_train = scaling(obj_scaling, verbose=True)
    print(df_train.head())

    df_train = binary_encode(df_train, df_train.select_dtypes('object').columns.to_list())
    print(df_train.head())

    base_predict_and_save_results(df_train, "dbscan_kmeans_basic_test", "test")
    base_tuning_kmeans(df_train, "kmeans_k_tuning_test", "test")
    base_spec_hierarchical(df_train, "spectral_test", "test")
