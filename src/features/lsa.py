import numpy as np
import pandas as pd
import time

from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

from pathlib import Path

start = time.time()

DATA_DIR = Path('../../data/processed/FakeNewsNet/')
# DATA_DIR = Path("")
dataset = 'BuzzFeed' # or 'large'

print("Loading TF-IDF matrix from disk")
tf_idf_info = np.load(DATA_DIR / f'{dataset}_tf-idf.npz', allow_pickle=True)
tf_idf_matrix, tf_idf_news_id, all_unique_words = tf_idf_info['tf_idf'], tf_idf_info['news_id'], tf_idf_info['all_unique_words']
print("Loaded TF-IDF matrix of shape {} from disk!".format(tf_idf_matrix.shape))
del tf_idf_info

### LARGE MATRIX ###
print("Computing LSA matrix")
svd         = TruncatedSVD(n_components=300, random_state=42)
lsa_matrix  = svd.fit_transform(tf_idf_matrix)

print("Computing t-SNE")
tsne            = TSNE(n_components=3, random_state=42)
tsne_features   = tsne.fit_transform(lsa_matrix)

print("Saving LSA matrix to disk")
np.savez_compressed(DATA_DIR / f'{dataset}_lsa-matrix.npz', lsa_matrix=lsa_matrix)
print("Saving t-SNE features to disk")
np.savez_compressed(DATA_DIR / f'{dataset}_tsne-features.npz', tsne_features=tsne_features)

print(f"Script running took: {time.time() - start:.2f} seconds")
