from src.preprocessing import load_and_preprocess
from src.clustering import kmeans_clustering
from src.evaluation import evaluate_clustering
from src.visualization import plot_clusters

X, raw_df = load_and_preprocess("data/customers.csv")

labels, model = kmeans_clustering(X, n_clusters=4)
score = evaluate_clustering(X, labels)

print(f"Silhouette Score: {score:.3f}")

plot_clusters(X, labels)
