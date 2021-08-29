import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


ProjectData = pd.read_csv("Project_Data_1.csv", index_col=0, decimal=",")
ProjectData.head()


ProjectData.isna()


ProjectData.dropna()


decomposition = ProjectData.iloc[:, 0:]
decomposition.head(6)


modelPCA = PCA(n_components=2)
pcaData = modelPCA.fit(decomposition).transform(
    decomposition)

newData = pd.DataFrame(pcaData, columns=["pca_1", "pca_2"])
newData.index = ProjectData.index
newData.head(6)


wcss = []  

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300, n_init=10, random_state=10)
    kmeans.fit(newData)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("wcss")
plt.show()


kmeans = KMeans(n_clusters=3)

kmeans.fit(newData)

print(kmeans.cluster_centers_)


newData["cluster"] = kmeans.labels_
newData.head()


sns.lmplot('pca_1', 'pca_2', data=newData, hue='cluster',
           palette='coolwarm', height=6, aspect=1, fit_reg=False)


ProjectData["cluster"] = kmeans.labels_


newData.sort_values(["cluster", "pca_1", "pca_2"])
newData.to_csv("output.csv", index=True)


ProjectData.loc["Sierra Leone"]
X = ProjectData.loc["Sierra Leone"].index[0:18]
Y = ProjectData.loc["Sierra Leone"].values[0:18]

plt.bar(X, Y)
plt.setp(plt.gca().get_xticklabels(), rotation=90,
         horizontalalignment='right')  

plt.show()


ProjectData.loc["Monaco"]
X = ProjectData.loc["Monaco"].index[0:18]
Y = ProjectData.loc["Monaco"].values[0:18]

plt.bar(X, Y)
plt.setp(plt.gca().get_xticklabels(), rotation=90,
         horizontalalignment='right')  

plt.show()
