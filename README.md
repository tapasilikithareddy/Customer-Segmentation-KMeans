# Customer Segmentation using K-Means Clustering

## Overview

Customer segmentation is a critical process for businesses to understand their customer base. It involves dividing customers into distinct groups (segments) based on certain characteristics such as age, spending habits, gender, etc. This project uses **K-Means Clustering**, an unsupervised machine learning algorithm, to cluster customers based on their spending behavior and demographic data.

In this project, the goal is to segment customers visiting a mall into groups based on factors such as annual income, age, and spending score. The K-Means algorithm is applied to identify patterns and group customers accordingly. These insights can help businesses tailor their marketing strategies, products, and services to specific customer segments.

### K-Means Clustering

K-Means is an iterative clustering algorithm that tries to partition the dataset into **K** pre-defined clusters. It works by finding the centroid of each cluster and assigning data points to the nearest centroid. The algorithm minimizes the **intra-cluster distance** (distance between points within a cluster) and maximizes the **inter-cluster distance** (distance between centroids of clusters).

### Steps in the Project

1. **Data Exploration**:
   - Import necessary libraries (Pandas, Matplotlib, etc.).
   - Load and inspect the dataset.
   - Clean the data by handling missing values and removing irrelevant features.

2. **Data Preprocessing**:
   - Standardize the data using **StandardScaler** to ensure all features contribute equally to the model.
   
3. **K-Means Clustering**:
   - Use the **Elbow Method** to determine the optimal number of clusters (`K`).
   - Apply the **K-Means algorithm** to partition the customers into different segments.

4. **Visualization**:
   - Visualize the customer segments using scatter plots and other graphical tools.
   - Analyze the customer segments to gain actionable insights into spending behavior and demographics.

### Code Example

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load and clean the dataset
customer_data = pd.read_csv('customer_data.csv')
customer_data = customer_data.drop('Gender', axis=1)  # Drop irrelevant column

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data)

# Elbow Method to determine the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-Means with the optimal number of clusters
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans.fit(scaled_data)

# Add the cluster labels to the original DataFrame
customer_data['Cluster'] = kmeans.labels_

# Visualize the clusters
plt.scatter(customer_data['Age'], customer_data['Annual Income'], c=customer_data['Cluster'])
plt.title('Customer Segmentation')
plt.xlabel('Age')
plt.ylabel('Annual Income')
plt.show()
