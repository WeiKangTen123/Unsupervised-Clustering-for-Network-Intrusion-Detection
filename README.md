# **Network Anomaly Detection Using Clustering and Machine Learning**

## **Project Overview**
This project applies **unsupervised clustering and anomaly detection** techniques to structured network traffic data for **intrusion detection**. It uses various **machine learning algorithms**, **dimensionality reduction techniques**, and **data visualization methods** to identify unusual patterns in network activity.

## **Files and Directories**
- **`G1_finalized_main_coding.ipynb`** - The main Jupyter Notebook for data processing, clustering, and anomaly detection.
- **`G1_Dashboard.py`** - A Streamlit-based interactive dashboard for visualizing clustering results and anomalies.
- **`G1_Dashboard_command.txt`** - Instructions to run the Streamlit dashboard and install necessary dependencies.
- **`networkDetect.csv`** - The processed network traffic dataset used in the project.
- **`Network Anomaly Detection.docx`** - A document detailing the dataset, features, and anomaly detection problem.
- **`G1_ppt.pptx`** - A PowerPoint presentation summarizing the research, methodology, and findings.

---

## **Dataset Description**
The dataset is based on **KDDCUP’99**, a widely used dataset for network-based anomaly detection. The dataset contains structured data representing network traffic features such as:
- **Basic Features**: Duration, Protocol Type, Service, Flag, Source/Destination Bytes, etc.
- **Content Features**: Failed Logins, Root Access Attempts, Shell Usage, File Creations, etc.
- **Traffic Features**: Connection Count, Error Rates, Same/Different Service Rates, etc.
- **Target Variable**: Attack categories such as **Normal, DOS, PROBE, R2L, U2R**

---

## **Methods and Techniques Used**
### **1. Data Preprocessing**
- **Handling Missing Values**: Checked for null values and ensured data integrity.
- **Duplicate Removal**: Identified and removed duplicate records.
- **Encoding Categorical Data**: Applied **One-Hot Encoding** to categorical features.
- **Feature Engineering**: Created new features such as `total_bytes = src_bytes + dst_bytes`.

### **2. Sampling Techniques**
- **Random Sampling**
- **Systematic Sampling**
- **Incremental Sampling**
- **Cluster-Based Sampling**

### **3. Unsupervised Learning Techniques**
#### **Clustering Algorithms:**
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: Detects clusters based on density and marks outliers.
- **HDBSCAN (Hierarchical DBSCAN)**: Improves DBSCAN by detecting clusters with varying densities.
- **Gaussian Mixture Models (GMM)**: Uses a probabilistic approach to model clusters in network traffic.
- **K-Means Clustering**: Groups data points into predefined clusters based on similarity.

#### **Anomaly Detection Models:**
- **Isolation Forest**: Detects anomalies by isolating outliers in high-dimensional data.
- **Autoencoder Neural Network**: Identifies anomalies using reconstruction error.

### **4. Dimensionality Reduction**
- **PCA (Principal Component Analysis)**: Reduces high-dimensional data for visualization and clustering.
- **UMAP (Uniform Manifold Approximation and Projection)**: Captures nonlinear structures for better clustering insights.
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Helps visualize high-dimensional data in 2D space.

### **5. Clustering Evaluation Metrics**
- **Silhouette Score**: Measures how well-separated clusters are.
- **Davies-Bouldin Score**: Evaluates intra-cluster and inter-cluster similarity.
- **Calinski-Harabasz Score**: Assesses cluster dispersion and separation.

---

## **Interactive Dashboard (Streamlit)**
A **Streamlit dashboard** is included to interactively visualize clustering results and anomalies:
- **Upload datasets** (MinMax Scaled & Standard Scaled).
- **View Feature Importance** via Random Forest.
- **Explore Correlation Heatmaps**.
- **Apply Clustering Algorithms** and view results with PCA, UMAP, and t-SNE.
- **Evaluate Clustering Performance** with metrics.
- **Identify Anomalies** with Isolation Forest and Autoencoder.

To run the dashboard:
```bash
streamlit run G1_Dashboard.py
```

---

## **Installation and Dependencies**
### **1. Create a Virtual Environment (Optional)**
```bash
conda create -n myenv python=3.9
conda activate myenv
```

### **2. Install Required Libraries**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow streamlit umap-learn hdbscan plotly
```

---

## **Results & Conclusion**
- **MinMax Scaled Dataset**: PCA revealed clear cluster separation, while UMAP showed slight overlap in clusters.
- **Autoencoder-based Anomaly Detection**: Detected high-risk anomalies that traditional clustering missed.
- **DBSCAN & HDBSCAN**: Identified potential attack patterns more effectively than K-Means.
- **Overall**: The combination of clustering, feature selection, and anomaly detection provided a **robust method for detecting unusual network traffic behavior**.

---

## **Future Improvements**
- **Use Real-Time Streaming Data** instead of a static dataset.
- **Apply Deep Learning Models** like LSTM for anomaly detection.
- **Integrate with Security Systems** for automated network threat detection.
