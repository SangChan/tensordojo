import tensorflow as tf
import numpy as np
# 그림 9.8 출력 코드
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, Y = make_blobs(random_state=6)
pred_Y = np.zeros_like(Y)

fig = plt.figure(figsize=(6,12))

centers = []
for c in range(3):
    center_X = random.random() * (max(X[:,0]) - min(X[:,0])) + min(X[:,0])
    center_Y = random.random() * (max(X[:,1]) - min(X[:,1])) + min(X[:,1])
    centers.append([center_X, center_Y])
centers = np.array(centers)
prev_centers = []

for t in range(3):
    for i in range(len(X)):
        min_dist = 9999
        center = -1
        for c in range(3):
            dist = ((X[i,0] - centers[c,0]) ** 2 + (X[i,1] - centers[c,1]) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                center = c
        pred_Y[i] = center
    
    ax = fig.add_subplot(3, 1, t+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(X[:,0], X[:,1], marker='.', c=pred_Y, cmap='rainbow')
    ax.scatter(centers[:,0], centers[:,1], marker='D', c=range(3), cmap='rainbow', edgecolors=(0,0,0,1))
    
    if len(prev_centers) != 0:
        for c in range(3):
            ax.arrow(prev_centers[c,0], prev_centers[c,1], (centers[c,0]-prev_centers[c,0])*0.95, (centers[c,1]-prev_centers[c,1])*0.95, head_width=0.25, head_length=0.2, fc='k', ec='k')
    
    # update center
    prev_centers = np.copy(centers)
    for c in range(3):
        count = len(pred_Y[pred_Y == c])
        centers[c,0] = sum(X[pred_Y == c, 0]) / (count+1e-6)
        centers[c,1] = sum(X[pred_Y == c, 1]) / (count+1e-6)

plt.show()