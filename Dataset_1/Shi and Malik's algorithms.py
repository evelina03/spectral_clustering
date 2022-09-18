#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
from sklearn.metrics import DistanceMetric
import networkx as nx
from scipy import linalg



# In[2]:


# coverting the input csv file into a graph
def getGraph(csv,measure) :
    
    # Uploading the files
    df = pd.read_csv(csv)
    df.head()
    df_cluster = df.iloc[:,1:13]
    
    # Converting the file into Graph
    dist = DistanceMetric.get_metric(measure)
    matrix_dist = dist.pairwise(df_cluster.to_numpy())
    Graph=nx.from_numpy_matrix(matrix_dist)
    draw_graph(Graph)
    
    return matrix_dist, Graph
    


# In[3]:


#Function to draw graph
def draw_graph(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)


# In[4]:


#equation fro adjacency matrix
def equation(x):
    n = 0.4
    function = np.exp(-((matrix_dist ** 2) / (2. * n ** 2)))
    return function 
    


# In[5]:


#Getting adjacency matrix for all the points in csv
def getAjacency(matrix):
    adacency_matrix = []
    for row in matrix_dist:
        x = []
        for point in row:
            x.append(equation(point))
            adacency_matrix.append(x)
   
    return adacency_matrix 


# In[6]:


def getLaplacian(Graph,matrix_dist):
    D = Graph.degree()
    degrees = [val for (node, val) in Graph.degree()]
    D = np.zeros((len(degrees), len(degrees)))
    np.fill_diagonal(D, degrees)
#     print('degree matrix:')
    print(D)
    L = D - matrix_dist
    
#Normalized laplacian matrix
    D_half = linalg.fractional_matrix_power(D, -0.5)
    LN = np.matmul(np.matmul(D_half, L), D_half)
    
#  print('laplacian matrix:')
    print(LN)
    return D,L,LN


# In[7]:


def getEigen(LN):
    e, v = np.linalg.eig(LN)
        # eigenvalues
    print('eigenvalues:')
    print(e)
        # eigenvectors
    print('eigenvectors:')
    print(v)
    return e,v
    


# In[11]:


matrix_dist, Graph = getGraph("HeartFailure.csv","euclidean")


# In[12]:


print(matrix_dist)


# In[ ]:


adjacency_matrix= getAjacency(matrix_dist)
print(adjacency_matrix)


# In[ ]:


D,L,LN = getLaplacian(Graph,matrix_dist)


# In[ ]:


e,v = getEigen(LN)


# In[ ]:


# fig = plt.figure()
# ax1 = plt.subplot(121)
# plt.plot(e)
# ax1.title.set_text('eigenvalues')
# i = np.where(e < 10e-6)[0]
# ax2 = plt.subplot(122)
# plt.plot(v[:, i[0]])
# fig.tight_layout()
# plt.show()


# # In[ ]:


# U = np.array(v[:, i[0]])
# km = KMeans(init='k-means++', n_clusters=3)
# df['clusters'] = km.fit_predict(U)
# df

