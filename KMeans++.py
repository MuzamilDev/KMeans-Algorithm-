import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("/content/Mall_Customers.csv")
df  #it will read and show csv file we have 

x = df[['Annual Income (k$)','Spending Score (1-100)']] #just two colums

x  #x variable data shown 

from sklearn.cluster import KMeans

wcss=[]
for i in range (1,20):
  
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=30) #function
  
  kmeans.fit(x) #train the model
  
  wcss.append(kmeans.inertia_)
  
wcss #for data show

plt.plot(range(1,20),wcss) #plot

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=30) #kmean function

kmeans.fit(x)

x['cluster number'] = kmeans.fit_predict(x) #predict function with fit function

x  #data shown for 5 cluster 
