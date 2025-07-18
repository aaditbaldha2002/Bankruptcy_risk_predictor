import logging
import os
from typing import List, Tuple
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

def clustering(data_path: str) -> Tuple[str,List[str]]:
    try:
        df = pd.read_csv(data_path)
        bankrupt=df['Bankrupt?']
        df=df.drop(columns=['Bankrupt?'])
        logger.info(f"Loaded data from {data_path} with shape {df.shape}")
    except FileNotFoundError as fnf_err:
        logger.error(f"File not found: {data_path} - {fnf_err}")
        raise
    except Exception as e:
        logger.exception(f"Error loading data from {data_path}: {e}")
        raise

    try:
        logger.info("Scaling the dataset...")
        scl = StandardScaler()
        scaled_df = pd.DataFrame(scl.fit_transform(df), columns=df.columns)
        min_max_scaler = MinMaxScaler()
        final_scaled_df = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)
        logger.info("Scaling completed successfully")
    except Exception as e:
        logger.exception(f"Error during scaling: {e}")
        raise

    try:
        logger.info("Performing KMeans clustering...")
        kmeans = KMeans(n_clusters=5, random_state=2)
        kmeans.fit(final_scaled_df)
        final_scaled_df['Cluster'] = kmeans.labels_
        final_scaled_df['Bankrupt?']=bankrupt
        logger.info("KMeans clustering completed")
    except Exception as e:
        logger.exception(f"Error during KMeans clustering: {e}")
        raise

    try:
        logger.info("Storing the artifacts...")
        cluster_dfs = []
        cluster_file_paths = []
        base_dir = 'artifacts/clustering'
        os.makedirs(base_dir, exist_ok=True)

        clustering_file_path = os.path.join(base_dir, 'clustering_intermediate_train_data.csv')
        final_scaled_df.to_csv(clustering_file_path, index=False)
        logger.info(f"Saved clustered dataframe with labels at {clustering_file_path}")

        for cluster_num in range(5):
            cluster_df = final_scaled_df[final_scaled_df['Cluster'] == cluster_num].drop(columns='Cluster')
            cluster_file_path = os.path.join(base_dir, f'cluster_{cluster_num}_intermediate_train_data.csv')
            cluster_df.to_csv(cluster_file_path, index=False)
            logger.info(f"Saved cluster {cluster_num} data at {cluster_file_path}")
            cluster_dfs.append(cluster_df)
            cluster_file_paths.append(cluster_file_path)
        
        logger.info("Artifacts stored successfully")
    except Exception as e:
        logger.exception(f"Error during saving clustered data files: {e}")
        raise

    return clustering_file_path,cluster_file_paths
