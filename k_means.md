```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Use only Annual Income and Spending Score
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# -------------------------------
# K-MEANS WITH K = 5
# -------------------------------

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

# Train the model on the full dataset
kmeans.fit(X)

# -------------------------------
# SILHOUETTE SCORE
# -------------------------------

labels = kmeans.labels_

score = silhouette_score(X, labels)

print("Silhouette Score:", score)

# -------------------------------
# CLUSTER CENTERS
# -------------------------------

print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# -------------------------------
# VISUALIZATION
# -------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    X['Annual Income (k$)'],
    X['Spending Score (1-100)'],
    c=labels,
    cmap='viridis',
    s=50
)

# Plot centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c='red',
    marker='X',
    s=200,
    label='Centroids'
)

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('K-Means Clustering (K=5)')
plt.legend()
plt.show()

```
<img width="396" height="195" alt="image" src="https://github.com/user-attachments/assets/9e605b0f-d85d-4441-9c2d-28e9fcc6d8ca" />

<img width="746" height="545" alt="image" src="https://github.com/user-attachments/assets/7628bb13-2110-47e4-916d-55f8e1782141" />

