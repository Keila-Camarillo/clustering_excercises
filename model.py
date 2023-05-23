import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

def plot_inertia(X):
    """
    Plots the change in inertia as the number of clusters (k) increases using the K-means algorithm.

    Parameters:
    X (array-like or dataframe): The input dataset for clustering.

    Returns:
    None (displays a plot)

    """

    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(9, 6))
        inertia_values = {k: KMeans(k).fit(X).inertia_ for k in range(2, 12)}
        pd.Series(inertia_values).plot(marker='x')
        plt.xticks(range(2, 12))
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.title('Change in inertia as k increases')
        plt.show()


def plot_kmeans_clusters(X, col_1, col_2):
    """
    Plots the scatter plots of clusters for different values of k using the K-means algorithm.

    Parameters:
    X (array-like or dataframe): The input dataset for clustering.

    Returns:
    None (displays a plot)

    """

    fig, axs = plt.subplots(2, 2, figsize=(13, 13), sharex=True, sharey=True)

    for ax, k in zip(axs.ravel(), range(2, 6)):
        clusters = KMeans(k).fit(X).predict(X)
        ax.scatter(X[col_1], X[col_2], c=clusters)
        ax.set(title='k = {}'.format(k), xlabel=col_1, ylabel=col_2)

    plt.tight_layout()
    plt.show()

def visualize_cluster_centers(df, x, y):
    """
    Visualizes the clusters along with the centers on unscaled data.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data and cluster labels.
    x (str): Name of the column for the x-axis in the scatter plot.
    y (str): Name of the column for the y-axis in the scatter plot.

    Returns:
    None (displays a plot)

    """
    plt.figure(figsize=(14, 9))

    # Scatter plot of data with hue for cluster
    sns.scatterplot(x=df[x], y=df[y], data=df, hue='cluster')

    # Plot cluster centers (centroids)
    centroids = df.groupby('cluster')[[x, y]].mean()
    centroids.plot.scatter(x=x, y=y, ax=plt.gca(), color='k', alpha=0.3, s=800, marker=(8, 1, 0), label='centroids')

    plt.title('Visualizing Cluster Centers')

    # Get unique cluster labels
    unique_clusters = df['cluster'].unique()

    # Create legend labels for clusters
    cluster_labels = [f'Cluster {cluster}' for cluster in unique_clusters]

    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()
