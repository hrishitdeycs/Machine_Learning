```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Use only Annual Income and Spending Score
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# 80% Training, 20% Testing
X_train, X_test = train_test_split(
    X,
    test_size=0.20,
    random_state=42
)

# K-Means with k = 5
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

# Train the model
kmeans.fit(X_train)

# -------------------------------
# TRAINING SCORE
# -------------------------------

# Cluster labels for training data
train_labels = kmeans.labels_

# Training silhouette score
training_score = silhouette_score(X_train, train_labels)

print("Training Silhouette Score:", training_score)

# -------------------------------
# TESTING SCORE
# -------------------------------

# Predict clusters for test data
test_labels = kmeans.predict(X_test)

# Calculate silhouette score for test data
testing_score = silhouette_score(X_test, test_labels)

print("Testing Silhouette Score:", testing_score)

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
    X_train['Annual Income (k$)'],
    X_train['Spending Score (1-100)'],
    c=train_labels,
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
<img width="425" height="229" alt="image" src="https://github.com/user-attachments/assets/5eeff43a-b7a9-4b87-a94a-c5a1c9ae0c03" />

<img width="762" height="575" alt="image" src="https://github.com/user-attachments/assets/2f7382bc-6ee1-4523-92ea-cf749ceaf23c" />
