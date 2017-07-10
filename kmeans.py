import numpy as np
from sklearn.cluster import KMeans
import pandas as pd


broken_df = pd.read_csv('/home/pi/dados.csv')
mat = broken_df.as_matrix()
kmeans = KMeans(n_clusters=3,random_state=3425).fit(mat)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

def k_means_function(x):

	#Mean = x.mean(axis=0)
	#print(Mean)
	Y = kmeans.predict(x)

	return Y
