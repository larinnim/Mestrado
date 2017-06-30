import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

broken_df = pd.read_csv('dados.csv')
mat = broken_df.as_matrix()


kmeans = KMeans(n_clusters=4,random_state=3425).fit(mat)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_


