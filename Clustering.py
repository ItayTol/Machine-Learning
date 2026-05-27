'''
318720604
Itay Toledo
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering as ac, KMeans as km
from sklearn.metrics import silhouette_score as sil
from scipy.cluster import hierarchy as hie

csv="https://ocw.mit.edu/courses/sloan-school-of-management/15-071-the-analytics-edge-spring-2017/clustering/assignment-6/dailykos.csv"
df=pd.read_csv(csv)

# Find labels of each algorithm
def get_labels_km(num_clusters):
    model=km(num_clusters, random_state=3)
    model.fit(df)
    return model.labels_

def get_labels_ac(num_clusters):
    model=ac(num_clusters)
    model.fit(df)
    return model.labels_

# Find smallest and largest clusters of each amount of clusters in each algorithm
def find_min_max_km(num_clusters):
    cn=pd.Series(get_labels_km(num_clusters)).value_counts()
    return('Biggest', cn.index[0], cn.iloc[0], 'Smallest', cn.index[-1], cn.iloc[-1])

def find_min_max_ac(num_clusters):
    cn=pd.Series(get_labels_ac(num_clusters)).value_counts()
    return('Biggest', cn.index[0], cn.iloc[0], 'Smallest', cn.index[-1], cn.iloc[-1])

# Export to csv
def to_csv_km(num_clusters, cluster_index):
    con=(pd.concat([pd.DataFrame(get_labels_km(num_clusters)), df], axis=1))
    con.rename(columns={0:'cluster'}, inplace=True)
    cluster=con.query('cluster == @cluster_index').drop('cluster', axis=1)
    cluster.to_csv('C:/Users/dror_toledo/Desktop/Y4S1/Data Mining/Ex2/KMeans {} clusters, number{}.csv'.format(num_clusters, cluster_index))

def to_csv_ac(num_clusters, cluster_index):
    con=(pd.concat([pd.DataFrame(get_labels_ac(num_clusters)), df], axis=1))
    con.rename(columns={0:'cluster'}, inplace=True)
    cluster=con.query('cluster == @cluster_index').drop('cluster', axis=1)
    cluster.to_csv('C:/Users/dror_toledo/Desktop/Y4S1/Data Mining/Ex2/Hierarchical {} clusters, number{}.csv'.format(num_clusters, cluster_index))

# Find most frequent word in each cluster according to amount of clusters and algorithm
def find_words_km(num_clusters, cluster_index, num_words):
    con=pd.concat([pd.DataFrame(get_labels_km(num_clusters)), df], axis=1)
    grouped=con.groupby(con.columns[0]).sum().transpose()
    ord_=grouped.sort_values(by=grouped.columns[cluster_index], ascending=False).index[:num_words].values
    return ord_

def find_words_ac(num_clusters, cluster_index, num_words):
    con=pd.concat([pd.DataFrame(get_labels_ac(num_clusters)), df], axis=1)
    grouped=con.groupby(con.columns[0]).sum().transpose()
    ord_=grouped.sort_values(by=grouped.columns[cluster_index], ascending=False).index[:num_words].values
    return ord_

# Number of wanted amounts of clusters and their indices
samps=[2, 3, 7, 8]
clusters={2: [0, 1], 3: [0, 1, 2], 7: [i for i in range(7)], 8: [i for i in range(8)]}

# # K-Means
# Getting the 2 criterions sse, silhouette
sse_list=list()
sil_list=list()
for k in samps:
    kmeans = km(n_clusters=k, random_state=3)
    kmeans.fit(df)
    sse_list.append(kmeans.inertia_)
    sil_list.append(sil(df, kmeans.labels_))

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle("Comparing different amounts of clusters in K-Means")

ax1.plot(samps, sse_list)
ax1.set_xticks(samps)
ax1.set_title("SSE")
ax1.set_xlabel("Number of clusters", fontsize=10)
ax1.grid()

ax2.plot(samps, sil_list)
ax2.set_xticks(samps)
ax2.set_title("Silhouette score")
ax2.set_xlabel("Number of clusters", fontsize=10)
ax2.grid()
plt.show()

# Find minimum and maximum sizes of clusters
for k in samps:
    print(k, find_min_max_km(k))

# Find common words in each cluster
for index, lis in clusters.items():
    for val in lis:
        print(index, val, find_words_km(index, val, 15))

# Export each cluster to csv
for index, lis in clusters.items():
    for val in lis:
        to_csv_km(index, val)

# # Agglomerative clustering
# Getting the 2 criterions
# SSE is taken analogicly from the denrdogram of "Ward" method
dend_ward = hie.dendrogram(hie.linkage(df, method='ward'))
plt.title('"Ward" method')
sse_dict={2: 220, 3: 180, 7: 120, 8: 80}
sil_dict=dict()
for k in samps:
    agglo=ac(n_clusters=k).fit(df)
    sil_dict.update({k: (sil(df, agglo.labels_))})

# Vizualisations
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle("Comparing different amounts of clusters in agglomerative clustering")

ax1.plot(list(sse_dict.keys()), list(sse_dict.values()))
ax1.set_xticks(samps)
ax1.set_title("SSE")
ax1.set_xlabel("Number of clusters", fontsize=10)
ax1.grid()

ax2.plot(list(sil_dict.keys()), list(sil_dict.values()))
ax2.set_xticks(samps)
ax2.set_title("Silhouette score")
ax2.set_xlabel("Number of clusters", fontsize=10)
ax2.grid()
plt.show()

# Find minimum and maximum sizes of clusters
for samp in samps:
    print(samp, find_min_max_ac(samp))

# Find common words in each cluster
for index, lis in clusters.items():
    for val in lis:
        print(index, val, find_words_ac(index, val, 15))

# Export each cluster to csv
for index, lis in clusters.items():
    for val in lis:
        to_csv_ac(index, val)

dend_complete = hie.dendrogram(hie.linkage(df, method='complete'))
plt.title('"Complete" method')
dend_average = hie.dendrogram(hie.linkage(df, method='average'))
plt.title('"Average" method')
dend_ward = hie.dendrogram(hie.linkage(df, method='ward'))
plt.title('"Ward" method')
dend_centroid = hie.dendrogram(hie.linkage(df, method='centroid'))
plt.title('"Centroid" method')
