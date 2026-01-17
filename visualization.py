import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_clusters(X, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10")
    plt.title("Customer Segments Visualization")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()
