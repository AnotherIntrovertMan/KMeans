from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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
plt.scatter(f1, f2, c='black', s=7)

# fungsi utk menghitung Jarak Euclidean
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# banyak klaster
k = 3
# random centroids koordinat X
C_x = np.random.randint(0, np.max(X)-20, size=k)
# random centroids koordinat Y
C_y = np.random.randint(0, np.max(X)-20, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)

# Plotting centroidnya
plt.scatter(f1, f2, c='black', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')

# menyimpan centroid lama
Cen_old = np.zeros(C.shape)
# label cluster(0, 1, 2)
clusters = np.zeros(len(X))
# error = jarak antara centroid lama dan baru
error = dist(C, Cen_old, None)
# looping sampai error = 0, atau centroid sudah tidak pindah
while error != 0:
    # value cluster terdekat
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # store centroid lama
    Cen_old = deepcopy(C)
    # rata2
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, Cen_old, None)

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='black')

print(clusters)
print(clusters[1])
fig.canvas.set_window_title('k-Means Clustering(tanpa library) - Machine Learning - 1301170345 - 1301174311')

plt.show()