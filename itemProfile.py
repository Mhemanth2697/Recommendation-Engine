import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import numpy as np


class ItemProfile:

    def __init__(self, path):
        self.data = pd.read_csv(path)
        self.data.drop(columns = ['ID', 'ISBN', 'userRating'], inplace = True)
        self.vectorizer = TfidfVectorizer(min_df = 55)

    def deleteDuplicates(self):
        self.data = self.data.sort_values('averageRating').drop_duplicates(subset=['Name'], keep='last', ignore_index = True)
        self.data = self.data.sample(frac = 1)
        self.data = self.data.reset_index()
        self.data.drop(columns = ['index'], inplace = True)

    def modifyAuthors(self):
        def helper(row, map):
            return map[row['Authors']]
        authors = pd.DataFrame(self.data['Authors'].unique())
        self.map = dict(authors[0])
        self.inverseMap = dict((v, k) for k, v in self.map.items())
        self.data.Authors = self.data.apply(helper, map = self.inverseMap, axis = 1)

    def modifyGenres(self):
        genresList = list(self.data.genres)
        genresList = [' '.join(list(set(genre.lower().split(' ')))) for genre in genresList]
        vectorized = self.vectorizer.fit_transform(genresList)
        genreFrame = pd.DataFrame(vectorized.toarray(), columns = self.vectorizer.get_feature_names())
        self.data = self.data.concat([self.data, genreFrame], axis = 1)
        self.data.drop(columns = ['genres'], inplace = True)

    def getProfileMatrix(self):
        return self.data

    def clusterTrainData(self, k, model):
        if model == 'kmeans':
            kmeans = KMeans(n_clusters=k, max_iter=500)
            kmeans.fit(self.data)
            labels = kmeans.predict(self.data)
            return labels, kmeans

    def getSilhouetteScore(self, labels):
        silScore = silhouette_score(self.data, labels)
        return silScore

    def clusterAnalysis(self, iterFlag, k, model):
        if iterFlag:
            for i in range(15, k):
                labels = self.clusterData(i, model)
                silScore = self.getSilhouetteScore(labels)
                print(i, silScore)
        else:
            labels, model = self.clusterData(k, model)
            #silScore = self.getSilhouetteScore(labels, trainData)
            #print(silScore)
            return labels, model

    def getMembers(self, model, cluster):
        labels = model.labels_
        clusterMembers = self.data[labels == cluster]
        clusterMembersOriginal = self.data.loc[clusterMembers.index]
        return clusterMembersOriginal

    def getWeightedSum(self, similarity, clusterMembers):
        return np.matmul(similarity, clusterMembers)

    def getSimilarity(self, clusterMembers, iThDataVector):
        # clusterMembers = np.array(clusterMembers.fillna(0))
        iThDataVector = np.array(iThDataVector.fillna(0)).reshape(1, -1)
        similarityList = cosine_similarity(clusterMembers, iThDataVector)
        return np.transpose(similarityList)

    def getBookScore(self, iThDataVector, clusterMembers, similarityList):
        clusterMembers = clusterMembers.iloc[:, np.isnan(iThDataVector)]
        weightedSum = self.getWeightedSum(similarityList, clusterMembers)
        denominator = len(clusterMembers) - np.transpose(np.array(clusterMembers.isnull().sum()))
        for i in range(len(denominator)):
            if denominator[i] == 0:
                weightedSum[0][i] = -100
            else:
                weightedSum[i] = weightedSum[0][i] / denominator[i]
        return weightedSum

    def getRecommendationPerUser(self, i, model):
        cluster = self.predictLabel(i, model)
        clusterMembers = self.getMembers(model, cluster)
        iThDataVector = self.data.loc[self.testData.iloc[i].index]
        similarityList = self.getSimilarity(clusterMembers, iThDataVector)
        bookScore = self.weightedSm(iThDataVector, clusterMembers, similarityList)
        columnSortedIndices = np.argsort(-bookScore)
        recommendations = clusterMembers.columns[columnSortedIndices[:10]]
        return recommendations

    def getRecommendations(self, labels, model):  # IMPLEMENT FOR ALL USERS
        user = self.testData.iloc[5]
        print(self.getRecommendationPerUser(user, model))