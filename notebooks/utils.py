# utils.py
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

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
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Convert IDs to string (since they're identifiers)
    id_columns = ['account_id', 'id']
    df[id_columns] = df[id_columns].astype(str)

    # Convert URLs to category (since they're repeated strings)
    url_columns = ['bio_link', 'url', 'profile_pic_url']
    df[url_columns] = df[url_columns].astype('category')

    # Convert text fields to category
    text_columns = ['nickname']
    df[text_columns] = df[text_columns].astype('category')

    # Keep biography as string (since it's likely unique free text)
    # Keep engagement rates as float64 (for precision)
    # Keep numeric counts as int64
    # Keep is_verified as boolean
    # Drop create_time since it's all null
    df = df.drop('create_time', axis=1)

    # Convert top_videos to appropriate type (assuming it's JSON-like string data)
    df['top_videos'] = df['top_videos'].astype('category')
    
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

def find_optimal_clusters(scaled_features):
    """Find the optimal number of clusters using multiple methods"""
    wcss = []
    
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)
        
    return wcss

def plot_cluster_metrics(wcss):
    """Plot the WCSS scores"""
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(2, 11), wcss, marker='o', linestyle='-', color='blue')
    plt.title('Elbow Method for Optimal k', fontsize=14)
    plt.xlabel('Number of Clusters', fontsize=12)
    plt.ylabel('WCSS', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()