from sklearn.cluster import KMeans, AgglomerativeClustering

def kmeans_clustering(X, n_clusters=4):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    return labels, model

def hierarchical_clustering(X, n_clusters=4):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    return labels
