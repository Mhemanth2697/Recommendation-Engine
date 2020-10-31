import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class UserProfile:

    def __init__(self, path):
        self.data = pd.read_csv(path)

        def tooFewHelper(col):
            if col == 'too few ratings':
                return 'tooFewRatings'
            else:
                return col
        self.data['genres'] = self.data.genres.apply(tooFewHelper)
        self.data.dropna(axis=0, inplace=True)
        self.data.drop(columns=['Authors', 'ISBN', 'pagesNumber', 'averageRating', 'genres'], inplace=True)

    def pivotData(self):             
        '''
        Pivots the rows and columns of the original data so that the books are the columns and users are rows 
        
        '''
        self.data['ncol'] = self.data.index
        self.data = self.data.sort_values('ncol').drop_duplicates(subset = ['ID', 'Name', 'userRating'], keep = 'last')
        self.data.drop(columns = ['ncol'], inplace = True)
        self.data = self.data.pivot(index = 'ID', columns = 'Name', values = 'userRating')

    def normalizeData(self):                  
        '''
        Normalizes the data with the mean of the columns for the entire row
        
        '''
        meanCol = self.data.mean(axis = 1)
        for column in list(self.data.columns):
            self.data[column] = self.data[column] - meanCol
        self.data['mean'] = meanCol

    def fixNa(self):                  
        '''
        Fills any NA rows for the ratings here with the mean obtained from the train data for that row(user)
        
        '''
        self.trainData = self.trainData.apply(lambda row: row.fillna(-row['mean'] -1), axis=1)
        self.testData = self.testData.apply(lambda row: row.fillna(-row['mean'] -1), axis=1)
        self.trainData.drop(columns=['mean'], inplace = True)
        self.testData.drop(columns=['mean'], inplace = True)
        self.data.drop(columns = ['mean'], inplace = True)
        print("FIX NA DONE")

    def splitTrainAndTest(self):
        self.data = self.data.sample(frac = 1)
        self.trainData = self.data[:2000]
        self.testData = self.data[2000:]

    def clusterTrainData(self, k, model):
        if model == 'kmeans':
            kmeans = KMeans(n_clusters=k, max_iter=500)
            kmeans.fit(self.trainData)
            labels = kmeans.predict(self.trainData)
            return labels, kmeans

    def getSilhouetteScore(self, labels):
        silScore = silhouette_score(self.trainData, labels)
        return silScore

    def clusterAnalysis(self, iterFlag, k, model):
        if iterFlag:
            for i in range(15, k):
                labels = self.clusterTrainData(i, model)
                silScore = self.getSilhouetteScore(labels)
                print(i, silScore)
        else:
            labels, model = self.clusterTrainData(k, model)
            #silScore = self.getSilhouetteScore(labels, trainData)
            #print(silScore)
            print("------CLUSTERING DONE-----")
            return labels, model

    def predictLabel(self, i, model):         
        '''
        Predicts the cluster label of the user in question
        
        '''
        row = np.array(self.testData.iloc[i, :]).reshape(1, -1)
        cluster = model.predict(row)            
        return cluster

    def getMembers(self, model, cluster):     
        '''
        Gets the other members of the cluster to calculate similarity
        
        '''
        labels = model.labels_
        clusterMembers = self.trainData[labels == cluster]
        clusterMembersOriginal = self.data.loc[clusterMembers.index]
        return clusterMembersOriginal

    def getWeightedSum(self, similarity, clusterMembers):   
        '''
        Weighted sum of the Cosine Similarity
        
        '''
        print(similarity.shape)
        print(clusterMembers.shape)
        score = np.matmul(similarity, clusterMembers)
        return score

    def getSimilarity(self, clusterMembers, iThDataVector):   
        '''
        Calculates the Cosine Similarity between the users of the particular cluster
        
        '''
        clusterMembers = np.array(clusterMembers.fillna(0))
        iThDataVector = np.array(iThDataVector.fillna(0)).reshape(1, -1)
        similarityList = cosine_similarity(clusterMembers, iThDataVector) 
        return np.transpose(similarityList)

    def getBookScore(self, iThDataVector, clusterMembers, similarityList):  
        '''
        
        Calculates the scores of the books that have been read by other members of the cluster of which the ith user is a part of.  
        
        '''
        clusterMembers = clusterMembers.iloc[:, np.isnan(iThDataVector)]  
        #Gets the columns which are null, that is the books that the user did not read
        
        weightedSum = self.getWeightedSum(similarityList, clusterMembers) 
        #Calculates the weighted sum after the calculation of similarity between the ith user and clustermembers using getSimilarity()
        
        denominator = len(clusterMembers) - np.transpose(np.array(clusterMembers.isnull().sum())) 
        #The denominator is the number of users that read the books the user has read, which is why we are removing the users that do not           have a common book.  
        for i in range(len(denominator)):
            if denominator[i] == 0:
                weightedSum[0][i] = -100
            else:
                weightedSum[i] = weightedSum[0][i] / denominator[i]
        return weightedSum

    def getRecommendationPerUser(self, i, model):
        '''
        Gets the Recommendations for the user
        
        '''
        cluster = self.predictLabel(i, model)
        clusterMembers = self.getMembers(model, cluster) 
        iThDataVector = self.testData.iloc[i, :] 
        #The data of the ith user
        print(iThDataVector.shape)
        similarityList = self.getSimilarity(clusterMembers, iThDataVector)
        bookScore = self.getWeightedSum(similarityList, np.array(clusterMembers))
        columnSortedIndices = np.argsort(-bookScore)
        # Sorting the books by descending order of scores
        
        recommendations = clusterMembers.columns[columnSortedIndices[:10]]
        #Getting the 10 best books to recommend
        
        return recommendations

    def getRecommendations(self, model):  #IMPLEMENT FOR ALL USER
        #print(self.testData.head())
        user = 2
        print(self.getRecommendationPerUser(user, model))
        return user
    
    def getTestData(self, user):
        return self.testData.iloc[[user], :]

'''
userProfileMatrix = UserProfile('finaldset.csv')
userProfileMatrix.pivotData()
userProfileMatrix.normalizeData()
userProfileMatrix.splitTrainAndTest()
userProfileMatrix.fixNa()
labels, model = userProfileMatrix.clusterAnalysis(iterFlag=False, k=25, model='kmeans')
#%%
print(userProfileMatrix.getTestData())

#%%
user = userProfileMatrix.getRecommendations(model)

#%%
data = userProfileMatrix.getTestData(user)
'''
