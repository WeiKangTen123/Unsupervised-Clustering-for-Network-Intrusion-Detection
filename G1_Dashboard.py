import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import hdbscan
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.cluster import KMeans
import umap

# Set page title and layout
st.title('Interactive Clustering and Anomaly Detection Dashboard')
st.sidebar.title('Options')

# Define the selected variables for clustering
selected_vars = [
    'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count', 
    'num_compromised', 'num_root', 'num_file_creations', 'num_shells', 
    'num_access_files', 'num_outbound_cmds', 'total_bytes', 'serror_rate', 
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 
    'srv_diff_host_rate', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

# File upload
uploaded_files = st.sidebar.file_uploader("Upload your datasets (MinMax Scaled and Standard Scaled)", type=["csv", "xlsx"], accept_multiple_files=True)

# Check if two datasets are uploaded
if uploaded_files and len(uploaded_files) == 2:
    # Assume first dataset is MinMax scaled and second dataset is Standard scaled
    minmax_data = pd.read_csv(uploaded_files[0]) if uploaded_files[0].name.endswith('csv') else pd.read_excel(uploaded_files[0])
    standard_data = pd.read_csv(uploaded_files[1]) if uploaded_files[1].name.endswith('csv') else pd.read_excel(uploaded_files[1])
    
    # Select only the chosen columns
    X_minmax = minmax_data[selected_vars].dropna()
    X_standard = standard_data[selected_vars].dropna()
    
    st.write("Data Overviews:")
    st.write("MinMax Scaled Data Overview:")
    st.write(X_minmax.head())
    st.write("Standard Scaled Data Overview:")
    st.write(X_standard.head())

    # Function to calculate and plot feature importance
    def plot_feature_importance(data, title):
        # Apply a Random Forest Classifier to determine feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(data, np.random.randint(0, 2, size=(data.shape[0],)))  # Use random target variable for feature importance

        # Get feature importance
        importance = rf.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': data.columns, 'Importance': importance})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title(title)
        st.pyplot(plt.gcf())

    # Plot feature importance for both datasets
    st.subheader("Feature Importance Visualization")
    plot_feature_importance(X_minmax, "Feature Importance for MinMax Scaled Data")
    plot_feature_importance(X_standard, "Feature Importance for Standard Scaled Data")

    # Display correlation heatmaps for both datasets
    st.subheader("Correlation Heatmap Visualization")
    fig_minmax, ax_minmax = plt.subplots(figsize=(10, 8))
    sns.heatmap(X_minmax.corr(), annot=True, cmap='coolwarm', ax=ax_minmax)
    st.write("Correlation Heatmap for MinMax Scaled Data")
    st.pyplot(fig_minmax)

    fig_standard, ax_standard = plt.subplots(figsize=(10, 8))
    sns.heatmap(X_standard.corr(), annot=True, cmap='coolwarm', ax=ax_standard)
    st.write("Correlation Heatmap for Standard Scaled Data")
    st.pyplot(fig_standard)

else:
    st.info("Please upload exactly two datasets: one MinMax scaled and one Standard scaled.")

# Function for interactive data exploration
def explore_data(data):
    st.subheader("Data Exploration")
    st.write(data)
    st.dataframe(data.describe())
    st.write("Select columns to visualize:")
    columns = st.multiselect("Columns", data.columns)
    if columns:
        fig = px.scatter_matrix(data[columns], title="Scatter Matrix Plot")
        st.plotly_chart(fig)

# Function to plot clusters
def plot_clusters(X, labels, title):
    fig = px.scatter(x=X[:, 0], y=X[:, 1], color=labels.astype(str), title=title, labels={'x': 'Feature 1', 'y': 'Feature 2'})
    st.plotly_chart(fig)

# Function to apply PCA after clustering
def apply_pca_after_clustering(data, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data)
    plot_clusters(X_pca, labels, "PCA Visualization with Clusters")

# Function to apply UMAP for visualization
def apply_umap_and_plot(data, labels):
    reducer = umap.UMAP(n_components=2)
    X_umap = reducer.fit_transform(data)
    fig = px.scatter(x=X_umap[:, 0], y=X_umap[:, 1], color=labels.astype(str), title="UMAP Visualization with Clusters", labels={'x': 'UMAP1', 'y': 'UMAP2'})
    st.plotly_chart(fig)

# Function to apply t-SNE for visualization with optimizations
def apply_tsne_and_plot(data, labels):
    # Determine the maximum number of components for PCA
    max_components = min(data.shape[0], data.shape[1]) - 1
    n_components = min(50, max_components)  # Choose a reasonable number for PCA
    
    # Reduce dimensionality with PCA first
    pca = PCA(n_components=n_components)
    data_reduced = pca.fit_transform(data)
    
    # Apply t-SNE with optimized parameters
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, learning_rate=200, random_state=42)
    X_tsne = tsne.fit_transform(data_reduced)
    fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=labels.astype(str), title="t-SNE Visualization with Clusters", labels={'x': 't-SNE1', 'y': 't-SNE2'})
    st.plotly_chart(fig)

# Function to evaluate clustering performance
def evaluate_clustering(X, labels):
    # Check if there are at least two clusters
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        st.warning("Clustering resulted in a single cluster or all noise points. Cannot compute clustering metrics.")
        return None, None, None

    # Compute clustering metrics if there are valid clusters
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    return silhouette, davies_bouldin, calinski_harabasz

# Function to evaluate Isolation Forest performance
def evaluate_isolation_forest(X, labels):
    # Labels from Isolation Forest: -1 for outliers, 1 for inliers
    labels_binary = (labels == -1).astype(int)  # Convert to binary: 1 for outliers, 0 for inliers
    precision, recall, f1, _ = precision_recall_fscore_support(labels_binary, labels, average='binary')

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels_binary, labels)
    roc_auc = auc(fpr, tpr)

    # Display evaluation results
    st.write(f"Isolation Forest Performance: Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {roc_auc:.4f}")
    fig_roc = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC = {roc_auc:.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig_roc.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    st.plotly_chart(fig_roc)

# Initialize session state to store results
if 'comparative_results' not in st.session_state:
        st.session_state['comparative_results'] = pd.DataFrame(columns=['Model', 'Dataset', 'Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score'])

# Select Clustering Algorithm
algorithm = st.sidebar.selectbox(
    "Select Clustering Algorithm",
    ["Gaussian Mixture Model (GMM)", "HDBSCAN", "DBSCAN", "Autoencoder", "Isolation Forest"]
)

# Sidebar options for algorithm-specific parameters
if algorithm == "DBSCAN" or algorithm == "Autoencoder":
    eps = st.sidebar.slider('Epsilon (eps)', 0.1, 2.0, 0.5)
    min_samples = st.sidebar.slider('Min Samples', 1, 20, 5)

elif algorithm == "Gaussian Mixture Model (GMM)":
    n_components = st.sidebar.slider('Number of Components', 1, 10, 3)
    covariance_type = st.sidebar.selectbox('Covariance Type', ['full', 'tied', 'diag', 'spherical'])

elif algorithm == "HDBSCAN":
    min_cluster_size = st.sidebar.slider('Min Cluster Size', 2, 50, 5)
    min_samples = st.sidebar.slider('Min Samples', 1, 20, 5)

elif algorithm == "Isolation Forest":
    contamination = st.sidebar.slider('Contamination', 0.01, 0.5, 0.1)
    n_estimators = st.sidebar.slider('Number of Trees (n_estimators)', 50, 300, 100)

if uploaded_files and len(uploaded_files) == 2:
    # Define clustering models
    def gmm_clustering(X):
        gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=0)
        return gmm.fit_predict(X)

    def hdbscan_clustering(X):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        return clusterer.fit_predict(X)

    def dbscan_clustering(X, eps, min_samples):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        return dbscan.fit_predict(X)

    def autoencoder_clustering(X):
        input_dim = X.shape[1]
        encoding_dim = 4
        input_layer = Input(shape=(input_dim,))
    
        encoded = Dense(encoding_dim * 2, activation='relu')(input_layer)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)
    
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        autoencoder.fit(X, X, epochs=100, batch_size=128, shuffle=True, validation_split=0.2, verbose=0)
    
        encoder = Model(input_layer, encoded)
        encoded_data = encoder.predict(X)
    
    # Apply KMeans clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        return kmeans.fit_predict(encoded_data)

    def isolation_forest_clustering(X):
        iso_forest = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=0)
        return iso_forest.fit_predict(X)

    # Apply selected algorithm to both datasets
    for dataset_name, X in [("MinMax Scaled", X_minmax), ("Standard Scaled", X_standard)]:
        st.write(f"Processing {dataset_name} Dataset...")

        # Perform clustering
        if algorithm == "Gaussian Mixture Model (GMM)":
            labels = gmm_clustering(X)
        elif algorithm == "HDBSCAN":
            labels = hdbscan_clustering(X)
        elif algorithm == "DBSCAN":
            labels = dbscan_clustering(X, eps, min_samples)
        elif algorithm == "Autoencoder":
            labels = autoencoder_clustering(X)
        elif algorithm == "Isolation Forest":
            labels = isolation_forest_clustering(X)
            # Evaluate the Isolation Forest separately
            evaluate_isolation_forest(X, labels)
            continue  # Skip the regular evaluation for Isolation Forest

        # Evaluate clustering performance
        silhouette, db, ch = evaluate_clustering(X, labels)

        # Update or add results to session state
        if silhouette is not None:
            if not st.session_state['comparative_results'].empty:
                # Check if there is an existing entry for the same model and dataset
                existing_index = st.session_state['comparative_results'][
                    (st.session_state['comparative_results']['Model'] == algorithm) &
                    (st.session_state['comparative_results']['Dataset'] == dataset_name)
                ].index

                # If there is an existing entry, update it
                if len(existing_index) > 0:
                    st.session_state['comparative_results'].loc[existing_index, ['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score']] = [silhouette, db, ch]
                else:
                    # If no existing entry, append new result
                    results_df = pd.DataFrame({
                        'Model': [algorithm],
                        'Dataset': [dataset_name],
                        'Silhouette Score': [silhouette],
                        'Davies-Bouldin Score': [db],
                        'Calinski-Harabasz Score': [ch]
                    })
                    st.session_state['comparative_results'] = pd.concat([st.session_state['comparative_results'], results_df], ignore_index=True)
            else:
                # If dataframe is empty, simply add the new results
                results_df = pd.DataFrame({
                    'Model': [algorithm],
                    'Dataset': [dataset_name],
                    'Silhouette Score': [silhouette],
                    'Davies-Bouldin Score': [db],
                    'Calinski-Harabasz Score': [ch]
                })
                st.session_state['comparative_results'] = pd.concat([st.session_state['comparative_results'], results_df], ignore_index=True)

        # Display evaluation results
        if silhouette is not None:
            st.write(f"**{dataset_name} Data**: Silhouette Score: {silhouette:.4f}, Davies-Bouldin Score: {db:.4f}, Calinski-Harabasz Score: {ch:.4f}")

        # Apply PCA after clustering and plot
        st.subheader(f'PCA Visualization with Clustering for {dataset_name} Data')
        apply_pca_after_clustering(X, labels)

        # Apply UMAP after clustering and plot
        st.subheader(f'UMAP Visualization with Clustering for {dataset_name} Data')
        apply_umap_and_plot(X, labels)

        # Apply t-SNE after clustering and plot
        st.subheader(f't-SNE Visualization with Clustering for {dataset_name} Data')
        apply_tsne_and_plot(X, labels)

# Display all stored results
st.subheader("Comparative Analysis of Clustering Models")
st.dataframe(st.session_state['comparative_results'])

# Plot comparative analysis across different scores
fig = px.bar(
    st.session_state['comparative_results'].melt(id_vars=['Model', 'Dataset'], 
                                                 value_vars=['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score'], 
                                                 var_name='Score Type', 
                                                 value_name='Score Value'),
    x='Model',
    y='Score Value',
    color='Dataset',
    barmode='group',
    facet_col='Score Type',
    title="Comparison of Clustering Models Across Different Scores"
)
st.plotly_chart(fig)

# Button to update clustering in real-time
if st.sidebar.button('Update Clustering'):
    st.experimental_rerun()

# Customization options
st.sidebar.title('Customization')
theme = st.sidebar.selectbox('Choose Theme', ['Light', 'Dark'])
if theme == 'Dark':
    st.write('<style>body { background-color: #2E2E2E; color: white; }</style>', unsafe_allow_html=True)