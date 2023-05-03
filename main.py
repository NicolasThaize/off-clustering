from model_saver.save import load_model
from metrics.metrics import plot_evaluation_metrics
from metrics.metrics import save_metrics_to_excel

# exmample implementation
my_object = load_model('dbscan_kmeans_basic_1')
df = my_object['df']
kmeans_model = my_object['kmeans_model']
dbscan_model = my_object['dbscan_model']

metrics1 = plot_evaluation_metrics(df, kmeans_model)
metrics2 = plot_evaluation_metrics(df, dbscan_model)

save_metrics_to_excel(iteration_name='', metrics=metrics1)