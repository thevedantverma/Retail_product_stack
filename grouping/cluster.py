from sklearn.cluster import DBSCAN
import numpy as np

def cluster_products(embeddings):

    embeddings = np.array(embeddings)

    clustering = DBSCAN(
        eps=0.1 ,
        min_samples=1,
        metric='cosine'
    )

    labels = clustering.fit_predict(
        embeddings
    )

    return labels.tolist()