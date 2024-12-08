import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from IPython.display import display, Image as IPImage
import matplotlib as mpl

mpl.rcParams['figure.max_open_warning'] = 100 



def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u, s, v = np.linalg.svd(np.dot(Phi.T, np.asarray(Lambda) ** 3 - (gamma / p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda))))))
        R = np.dot(u, v)
        d = np.sum(s)
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return np.dot(Phi, R)


def initialize_centroids(X, k):
    n_samples = X.shape[0]
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices]
    return centroids

def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    cluster_labels = np.argmin(distances, axis=1)
    return cluster_labels

def update_centroids(X, cluster_labels, k):
    new_centroids = np.array([X[cluster_labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def k_means(X, k, max_iters=100, tol=1e-4,duration=500):
    # Step 1: Initialize centroids
    centroids = initialize_centroids(X, k)
    plot_objects = []  

    for i in range(max_iters):
        # Step 2: Assign clusters
        cluster_labels = assign_clusters(X, centroids)

        # Step 3: Update centroids
        new_centroids = update_centroids(X, cluster_labels, k)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='tab20', s=50, edgecolors='k', alpha=0.7)
        ax.scatter(new_centroids[:, 0], new_centroids[:, 1], c='red', s=100, marker='X', edgecolors='white', linewidth=2, alpha=0.8, zorder=5)
        ax.set_title(f"Iteration {i+1}", fontsize=16)
        ax.set_xlabel('Feature 1', fontsize=14)
        ax.set_ylabel('Feature 2', fontsize=14)
        ax.set_facecolor('whitesmoke')
        ax.grid(True, alpha=0.3)
        plot_objects.append(fig)

            # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    if plot_objects:
        images = []
        for fig in plot_objects:
            buf = BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf)
            images.append(img)

        gif_filename = "kmeans_animation.gif" 
        gif_buf = BytesIO()
        images[0].save(gif_buf, format="GIF", save_all=True, append_images=images[1:], duration=duration, loop=0)
        with open(gif_filename, 'wb') as f:
            f.write(gif_buf.getvalue())
        gif_buf.seek(0)
        display(IPImage(data=gif_buf.read(), format="gif"))

    return centroids, cluster_labels, plot_objects  


