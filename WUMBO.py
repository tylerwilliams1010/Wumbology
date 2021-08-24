# [WUMBO]
# Weighted UID-filtered Multi-Metric Based Outlier detection
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import BallTree
import pandas as pd



class Wumbo:

    def __init__ (self,dataframe, uidColumnName, filterValue=True, alpha=0.01, features=["avgDistance", "maxDistance", "localDensity"]):
        self.dataframe = dataframe
        self.uidColumnName = uidColumnName
        self.filter = filterValue
        self.alpha = alpha 
        self.features = features

    
    def calculateFeatures(self, distancesArray, nearestNeighborsArray, iterVector) -> dict:
        
        resultsDict = {}


        if "avgDistance" in self.features:
            # Calculate Average Distance for k-Nearest Neighbos
            resultsDict["avgDistance"] = np.mean(distancesArray)

        if "maxDistance" in self.features:
            # Calculate Max Distance for k-th Neighbor
            resultsDict["maxDistance"] = np.max(distancesArray)

        if "localDensity" in self.features:
            kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(nearestNeighborsArray)
            
            resultsDict["localDensity"] = -1 * kde.score(iterVector)

        
        return resultsDict

    def filterUID(self):
        deduped_df = self.dataframe[self.dataframe[self.uidColumnName]!=self.currentUID].drop([self.uidColumnName], axis=1)
    
        return deduped_df

    def convertAlpha(self):
        # Convert alpha to "Risk Threshold", which is just alpha inverted
        riskThreshold = 1/self.alpha 
        return riskThreshold

    def apply(self):
        # [WUMBO] - Weighted UID-filtered Multi-Metric Based Outlier detection 
        ##############################################################################
        # Wumbo is an anomaly detector designed for large datasets with pockets of
        # common groups.
        #
        # Part of the output of Wumbo is the Outlier Score [0,1], the other part
        # is the weight, or Risk Score, a number that represents "Outlierness"
        # of the node.
        #
        # The algorithm requires no hyperparameters to be chosen except
        # alpha. 
        #
        # There are some optional arguments to select starting with filtering out similar
        # identities/uid's so that one identity/uid can't cluster with itself to poison 
        # the density values. 
        #
        # Additionally, the default measurements are kernel density, average distance, 
        # and max distance from some k-Nearest Neighbors where k is already
        # calculated by 5 <= sqrt(N) <= 50. This helps scale the dataset to a large
        # number of data points with common behavioral characteristics relative to the
        # size of the total, larger dataset.
        # 
        # These metrics are then combined and evaluated against the entire dataset
        # to find outliers and Score the weighted "Risk."
        ##############################################################################


        
        # Initialize a temporary and return dataframe 
        temp_df = self.dataframe.copy(deep=True)
        results_df = pd.DataFrame()
        
        # Calculate number of k-Neighbors 
        ##############################################################################
        # This is a number that is equal to the square root of distinct count of UID's
        # with a minimum of 5 and a maximum of 50. (Concept comes from t-SNE paper)
        # This way the number of k-Neighbors scales with the size of the data.
        ##############################################################################


        kNeighbors = min(max(int(len(self.dataframe[self.uidColumnName].unique()) ** 0.5),5),50)
        numberOfRows = len(self.dataframe.index)
    
    
        # Initialize feature columns
        if "avgDistance" in self.features:
            results_df["avgDistance"] = 0
        if "maxDistance" in self.features:
            results_df["maxDistance"] = 0
        if "localDensity" in self.features:
            results_df["localDensity"] = 0

        
        # Iterate through dataframe
        for x in range(numberOfRows):
            
            # Identify current UID Value
            iterVector: np.array
            
            # Filter (or not)
            if self.filter==True: 
                temp_df = self.dataframe.copy(deep=True)
                currentUID = temp_df.loc[x,self.uidColumnName]

                # Filter out UID's and convert to numpy
                iterVector = temp_df.loc[x].drop(self.uidColumnName).reset_index(drop=True).to_numpy().reshape(1,-1)
                neighbors = NearestNeighbors(n_neighbors=kNeighbors)
                neighbors.fit(temp_df.loc[x | temp_df[self.uidColumnName]!=currentUID].drop([self.uidColumnName], axis=1).to_numpy())
                distancesArray, ind = neighbors.kneighbors(iterVector, return_distance=True)
                
                nearestNeighborsArray = temp_df[temp_df.index.isin(ind[0])].drop(labels=self.uidColumnName,axis=1).to_numpy()

            else:
                currentUID = temp_df.loc[x,self.uidColumnName]
                #print(len(temp_df))
                # Find nearest neighbors
                iterVector = temp_df.loc[x].drop(self.uidColumnName).reset_index(drop=True).to_numpy().reshape(1,-1)
                neighbors = NearestNeighbors(n_neighbors=kNeighbors)
                neighbors.fit(temp_df.drop([self.uidColumnName], axis=1).to_numpy())
                distancesArray, ind = neighbors.kneighbors(iterVector, return_distance=True)
                
                nearestNeighborsArray = temp_df[temp_df.index.isin(ind[0])].drop(labels=self.uidColumnName,axis=1).to_numpy()
            
            
            
            # Calculate Features (for layer 1) based on Feature Values for Model
            resultsDict = self.calculateFeatures(distancesArray=distancesArray, nearestNeighborsArray=nearestNeighborsArray, iterVector=iterVector)

            resultsDict[self.uidColumnName] = currentUID
            results_df = results_df.append(resultsDict, ignore_index=True)
            


        output_df = self.dataframe.copy(deep=True)
        kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(results_df[self.features].to_numpy())

        for x in range(numberOfRows):

            iterRow = results_df.loc[x].drop("uid").reset_index(drop=True).to_numpy().reshape(1,-1)    
                
            output_df.loc[x, "Risk Score"] = 1/(numberOfRows * 10 ** (kde.score(iterRow)))
            if 1/((numberOfRows * 10 ** (kde.score(iterRow)))) > (1/self.alpha):
                output_df.loc[x, "Outlier"] = 1
            else:
                output_df.loc[x, "Outlier"] = 0
        
        return output_df
