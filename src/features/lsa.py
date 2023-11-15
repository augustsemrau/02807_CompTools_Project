import numpy as np
import pandas as pd
import time

from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

from pathlib import Path

start = time.time()

DATA_DIR = Path('../../data/processed/FakeNewsNet/')
# DATA_DIR = Path("")

print("Loading large TF-IDF matrix from disk")
tf_idf_info = np.load(DATA_DIR / 'large_tf-idf.npz', allow_pickle=True)
large_tf_idf_matrix, large_tf_idf_news_id, all_unique_words = tf_idf_info['tf_idf'], tf_idf_info['news_id'], tf_idf_info['all_unique_words']
print("Loaded large TF-IDF matrix of shape {} from disk!".format(large_tf_idf_matrix.shape))

print("Loading BuzzFeed TF-IDF matrix from disk")
tf_idf_info = np.load(DATA_DIR / 'BuzzFeed_tf-idf.npz', allow_pickle=True)
buzzfeed_tf_idf_matrix, buzzfeed_tf_idf_news_id = tf_idf_info['tf_idf'], tf_idf_info['news_id']
print("Loaded BuzzFeed TF-IDF matrix of shape {} from disk!".format(buzzfeed_tf_idf_matrix.shape))

del tf_idf_info

### LARGE MATRIX ###
print("Computing LSA matrix")
svd                 = TruncatedSVD(n_components=300, random_state=42)
large_lsa_matrix    = svd.fit_transform(large_tf_idf_matrix)

print("Computing t-SNE")
tsne            = TSNE(n_components=3, random_state=42)
tsne_features   = tsne.fit_transform(large_lsa_matrix)

print("Saving LSA matrix to disk")
np.savez_compressed(DATA_DIR / 'large_lsa-matrix.npz', lsa_matrix=large_lsa_matrix)
print("Saving t-SNE features to disk")
np.savez_compressed(DATA_DIR / 'large_tsne-features.npz', tsne_features=tsne_features)
del large_lsa_matrix, tsne_features

### BUZZFEED MATRIX ###
buzzfeed_lsa_matrix     = svd.transform(buzzfeed_tf_idf_matrix)
tsne                    = TSNE(n_components=3, random_state=42)
buzzfeed_tsne_features  = tsne.fit_transform(buzzfeed_lsa_matrix)
print("Saving LSA matrix to disk")
np.savez_compressed(DATA_DIR / 'BuzzFeed_lsa-matrix.npz', lsa_matrix=buzzfeed_lsa_matrix)
print("Saving t-SNE features to disk")
np.savez_compressed(DATA_DIR / 'BuzzFeed_tsne-features.npz', tsne_features=tsne_features)

print(f"Script running took: {time.time() - start:.2f} seconds")
