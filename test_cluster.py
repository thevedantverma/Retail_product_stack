import numpy as np

from grouping.cluster import cluster_products

embeddings = np.array([
    [1, 2, 3],
    [1, 2, 3.1],
    [10, 11, 12]
])

labels = cluster_products(embeddings)

print(labels)