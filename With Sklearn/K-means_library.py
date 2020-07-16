import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
plt.style.use('ggplot')

# Import dataset
data = pd.read_csv('data_clustering.csv')
print("Data dan atribut")
print(data.shape)
data.head()

# plotting data
f1 = data['x'].values
f2 = data['y'].values
X = np.array(list(zip(f1, f2)))

#banyak cluster
k = 3

kmeans = KMeans(n_clusters=3)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
C = kmeans.cluster_centers_

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig,ax = plt.subplots()
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if labels[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='black')

fig.canvas.set_window_title('k-Means Clustering(dengan library) - Machine Learning - 1301170345 - 1301174311')

plt.show()