# Network Intrusion Detection Datasets and Anomaly Detection
This project demonstrates the use of the nids-datasets library for handling network intrusion detection datasets. It includes analysis, visualization, and anomaly detection using various machine learning techniques.

# Project Overview
The project involves the following key steps:

# Dataset Handling: Use the nids-datasets library to load and explore network intrusion detection datasets.
Data Visualization: Visualize the distribution of different types of network events across various files.
Anomaly Detection: Apply machine learning techniques such as Isolation Forest, Local Outlier Factor (LOF), and k-NN to detect anomalies in the dataset.
Model Evaluation: Evaluate the performance of the anomaly detection models using precision and recall.

# Prerequisites
To run the project, ensure you have the following Python packages installed:

Python 3.6 or higher
nids-datasets
matplotlib
seaborn
scikit-learn
You can install the required packages using the following command:


pip install nids-datasets matplotlib seaborn scikit-learn
Dataset Handling
The project uses the nids-datasets library to load and explore the UNSW-NB15 dataset. The dataset contains various types of network events, including normal traffic and different types of attacks.

Example of loading the dataset:

from nids_datasets import Dataset, DatasetInfo

df = DatasetInfo(dataset='UNSW-NB15')
df.head()
Data Visualization
The distribution of different types of network events is visualized using bar charts and stacked bar charts. This helps in understanding the prevalence of different event types in the dataset.

Example of plotting event distributions:


import matplotlib.pyplot as plt

df.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Distribution of Network Events by File')
plt.xlabel('File')
plt.ylabel('Count')
plt.show()
Anomaly Detection
Isolation Forest
Isolation Forest is used to detect anomalies in the dataset. The anomaly scores are visualized to identify files with unusual patterns of network events.

Example of applying Isolation Forest:

from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.1)
df_filtered['anomaly_score'] = model.fit_predict(df_filtered)
Local Outlier Factor (LOF)
LOF is applied to identify outliers based on local density deviations. The results are compared with those from the Isolation Forest model.

k-Nearest Neighbors (k-NN)
A k-NN based anomaly detection method is implemented to identify anomalies based on the average distance to the nearest neighbors.

Example of applying k-NN anomaly detection:

from scipy.spatial import distance_matrix

def knn_anomaly_detection(X, k=5, threshold=1.5):
    # Implement k-NN anomaly detection
    return anomalies, avg_distances
Model Evaluation
The performance of the anomaly detection models is evaluated using precision and recall metrics.

Example of calculating precision and recall:

python
Copy code
from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
Visualization of Results
The results of the anomaly detection models are visualized to compare the distribution of events in anomalous files versus normal files.

Example of visualization:

plt.scatter(X_data[:, 0], X_data[:, 1], color='blue', label='Normal')
plt.scatter(X_data[anomalies, 0], X_data[anomalies, 1], color='red', label='Anomalies')
plt.legend()
plt.title("Anomaly Detection using k-NN")
plt.show()
License
This project is licensed under the MIT License. 
