import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import DistanceMetric

df_resume = pd.read_csv("resumes/resumes_train.csv")

df_resume["role"][df_resume["role"].iloc[-1] == df_resume["role"]] = "Other"

model = SentenceTransformer("all-MiniLM-L6-v2")
# encode text
embedding_arr = model.encode(df_resume["resume"])
# define query
query = "I need someone to build my data infrastructure"
# encode query
query_embedding = model.encode(query)
# define distance metric
dist = DistanceMetric.get_metric("euclidean")
# compute pairwise distances between query embeddings and resume embeddings
dist_arr = dist.pairwise(query_embedding, embedding_arr.reshape(-1, 1)).flatten()
# sort results
idist_arr_sorted = np.argsort(dist_arr)
# print roles of top 10 closet resumes to query in embedding space
print(df_resume["role"].iloc[idist_arr_sorted[:10]])
