#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Aryan Apte
# B20186
# 8770083396
from numpy.lib import r_
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import scipy 
from sklearn import metrics 
from scipy.optimize import linear_sum_assignment

data=pd.read_csv("C:\\Users\\amay\\Desktop\\DS3_Assn\\Assn7\\Iris.csv")
df=data.iloc[:,[i for i in range(4)]]


# finding eigonvectors and eigonvalues
corr_matrix = df.corr()
evalues, evectors = np.linalg.eig(corr_matrix)

#---------------------------------------------Q1------------------------------------------------
# Ploting eigon values 
plt.bar([0,1,2,3], evalues,width=0.5)
plt.xlabel('Position -->')
plt.title('Q1\nEigon value for PCA')
plt.ylabel("Eigon value -->")
plt.show()

#print(evectors)
#Finding 2 eigonvectors with highest eigonvalues
max_indices = []
for i in range(2):
    max = evalues[0]
    maxi = 0
    for j in range(len(evalues)):
        if evalues[j] > max:
            max = evalues[j]
            maxi = j
    max_indices.append(maxi)
    evalues[maxi]=0

eigonvector_1 = evectors[:, max_indices[0]]  #eigon vector with highest eigonvalue
eigonvector_2 = evectors[:, max_indices[1]]  #eigon vector with second highest eigonvalue

# Now reducing data into 2 dimension
pca_into_2 = PCA(n_components=2)
pca_into_2.fit(df)
reduced_data_into_2 = pca_into_2.fit_transform(df)
reduced_data_dataframe = pd.DataFrame(reduced_data_into_2, columns=['A', 'B'])

def purity_score(y_true, y_pred): 
    contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred) 
    #print(contingency_matrix)
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)

# Now ploting 2 dimensional data on scatter plot
plt.scatter(reduced_data_dataframe['A'], reduced_data_dataframe['B'])
plt.title('Q1\nData projected along 2 dimensions')
plt.xlabel('X - axis')
plt.ylabel('Y - axis')
plt.show()


# In[2]:


# Q2

kmeans = KMeans(n_clusters=3) 
kmeans.fit(reduced_data_dataframe)
kmeans_prediction = kmeans.predict(reduced_data_dataframe)
kmeans_center_x=[]
kmeans_center_y=[]
for i in kmeans.cluster_centers_:
    kmeans_center_x.append(i[0])
    kmeans_center_y.append(i[1])

data_with_kmeans_clusters=reduced_data_dataframe.copy()
data_with_kmeans_clusters['Clusters']=kmeans_prediction

# a
plt.scatter(data_with_kmeans_clusters['A'], data_with_kmeans_clusters['B'],c=data_with_kmeans_clusters['Clusters'],
            cmap='rainbow')
plt.scatter(kmeans_center_x,kmeans_center_y, marker='X', color='black', label='Cluster center')
plt.title('Q2 part 1\nKmeans cluster for K=3')
plt.xlabel('X - axis')
plt.ylabel('Y - axis')
plt.legend()
plt.show()

# b
kmeans_features_value=reduced_data_dataframe.values
#kmeans_distortion=sum(np.min(cdist(kmeans_features_value,kmeans.cluster_centers_, 'euclidean'),axis=1)) 
kmeans_distortion=kmeans.inertia_
print('Q2 part b--> Distortion measure :',kmeans_distortion)
print()

# c

print('Q2 part c--> Purity score :',purity_score(y_true=data['Species'], y_pred=kmeans_prediction))
print()


# In[3]:


# Q3

K=range(2,8)
kmeans_dis=[]
kmeans_pu_score=[]
for k in K:
    kmeans = KMeans(n_clusters=k) 
    kmeans.fit(reduced_data_dataframe)
    kmeans_prediction = kmeans.predict(reduced_data_dataframe)
    kmeans_features_value=reduced_data_dataframe.values
    kmeans_distortion=kmeans.inertia_
    kmeans_dis.append(kmeans_distortion)
    kmeans_pu_score.append(purity_score(y_true=data['Species'], y_pred=kmeans_prediction))

plt.plot(K, kmeans_dis, marker='o',color='black')
plt.xlabel('Number of cluster -->')
plt.ylabel('Distortion measure -->')
plt.title('Q3\n Number of clusters(K) vs. distortion measure')  
plt.show()

print('Q3 --> List of purity score :',kmeans_pu_score) 


# In[4]:


# Q4

gmm = GaussianMixture(n_components = 3)
gmm.fit(reduced_data_dataframe)
GMM_prediction = gmm.predict(reduced_data_dataframe)
data_with_gmm_clusters=reduced_data_dataframe.copy()
data_with_gmm_clusters['Clusters']=GMM_prediction

# a
gmm_features_value=reduced_data_dataframe.values
gmm_centers = np.empty(shape=(gmm.n_components, gmm_features_value.shape[1]))
for i in range(gmm.n_components):
    density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(reduced_data_dataframe)
    gmm_centers[i, :] = gmm_features_value[np.argmax(density)]
    
plt.scatter(data_with_gmm_clusters['A'], data_with_gmm_clusters['B'],c=data_with_gmm_clusters['Clusters'], cmap='rainbow')
plt.scatter(gmm_centers[:, 0], gmm_centers[:, 1], s=20, marker='x', color='black', label='Cluster center')
plt.title('Q4 part 1\nGMM for K=3')
plt.legend()
plt.xlabel('X - axis')
plt.ylabel('Y - axis')
plt.show()

# b

gmm_distortion=sum(gmm.score_samples(reduced_data_dataframe))
print('Q4 part b--> Distortion measure :',gmm_distortion)
print()

# c
print('Q4 part c--> Purity score :',purity_score(y_true=data['Species'], y_pred=GMM_prediction))
print()


# In[5]:


K=range(2,8)
gmm_dis=[]
gmm_pu_score=[]
for k in K:
    gmm = GaussianMixture(n_components=k) 
    gmm.fit(reduced_data_dataframe)
    GMM_prediction = gmm.predict(reduced_data_dataframe)
    gmm_features_value=reduced_data_dataframe.values
    gmm_centers = np.empty(shape=(gmm.n_components, gmm_features_value.shape[1]))
    for i in range(gmm.n_components):
        density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(reduced_data_dataframe)
        gmm_centers[i, :] = gmm_features_value[np.argmax(density)]
    gmm_distortion=sum(gmm.score_samples(reduced_data_dataframe))
    gmm_dis.append(gmm_distortion)
    gmm_pu_score.append(purity_score(y_true=data['Species'], y_pred=GMM_prediction))
plt.plot(K, gmm_dis,marker='o',color='black')
plt.xlabel('Number of cluster -->')
plt.ylabel('Distortion measure -->')
plt.title('Q5\n Number of clusters(K) vs. distortion measure')  
plt.show()

print('Q5 --> List of purity score :',gmm_pu_score) 


# In[6]:


# Q6

from sklearn.cluster import DBSCAN 
def dbscan_model(DATA,r_d, e, s):
    dbscan_model=DBSCAN(eps=e, min_samples=s).fit(r_d)
    DBSCAN_predictions = dbscan_model.labels_
    
    data_with_dbscan_clusters=r_d.copy()
    data_with_dbscan_clusters['Clusters']=DBSCAN_predictions
    
    plt.scatter(data_with_dbscan_clusters['A'], data_with_dbscan_clusters['B'],c=data_with_dbscan_clusters['Clusters'], cmap='rainbow')
    plt.title(f'Scatter plot for eps= {e} and min_samples= {s}')
    plt.xlabel('X - axis')
    plt.ylabel('Y - axis')
    plt.show()

    print(f'Purity score for eps= {e} and min_samples= {s} is :',purity_score(y_true=DATA['Species'],
                                                                              y_pred=DBSCAN_predictions))
    
dbscan_model(data,reduced_data_dataframe,1,4)
dbscan_model(data,reduced_data_dataframe,1,10)
dbscan_model(data,reduced_data_dataframe,5,4)
dbscan_model(data,reduced_data_dataframe,5,10)


# In[ ]:




