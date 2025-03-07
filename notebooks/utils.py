# utils.py
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def clean_data(df):
    """Clean the TikTok influencer dataset"""
    df_clean = df.copy()
    
    # Check for duplicates
    duplicates = df_clean.duplicated(subset='account_id').sum()
    print(f"Duplicates found: {duplicates}")
    if duplicates > 0:
        df_clean.drop_duplicates(subset='account_id', inplace=True)
    
    # Handle missing values
    df_clean['biography'] = df_clean['biography'].fillna('')
    df_clean.drop('create_time', axis=1, inplace=True)
    
    return df_clean

def remove_outliers_iqr(df, columns):
    """Remove outliers using the IQR method on specified columns"""
    df_no_outliers = df.copy()
    
    for col in columns:
        Q1 = df_no_outliers[col].quantile(0.25)
        Q3 = df_no_outliers[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Create a mask for rows to keep
        mask = (df_no_outliers[col] >= lower_bound) & (df_no_outliers[col] <= upper_bound)
        df_no_outliers = df_no_outliers[mask]
    
    return df_no_outliers

def find_optimal_clusters(scaled_features, max_clusters=10):
    """Find the optimal number of clusters using multiple methods"""
    wcss = []
    
    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)
        
    return wcss

def plot_cluster_metrics(wcss, max_clusters=10):
    """Plot the WCSS scores"""
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), wcss, marker='o', linestyle='-', color='blue')
    plt.title('Elbow Method for Optimal k', fontsize=14)
    plt.xlabel('Number of Clusters', fontsize=12)
    plt.ylabel('WCSS', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()