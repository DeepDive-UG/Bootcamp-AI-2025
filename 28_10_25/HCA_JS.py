"""
HCA - Jakub Susoł
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

'''
Nie, ten kod nie został obfuskowany
Po prostu go tak napisałem :D .
'''

Data_IN = pd.read_excel('Dane_JS.xlsx', 'Dane')


# Distances and autoscaling calculation
def autoscaleData(data: pd.DataFrame):
    """
    Autoscales given data using Z-Score method
    :param data: Data to autoscale
    :return: Autoscaled data array
    """
    variables = list(data.columns)
    autoscaled = []
    for var in variables:  # gets variable from dataset
        std = data[var].std()
        mean = data[var].mean()
        tmp = (data[var]-mean)/std  # performs autoscaling by z-score
        autoscaled.append(tmp)  # add scaled column as row to the matrix
    # Return transposed autoscaled array to reflect the original dataset [Var == Column]
    return np.array(autoscaled).T


def euclideanDistance(data: np.ndarray):
    """
    Calculates Euclidean distances between objects in data array.
    :param data: Input data array to calculate distances on
    :return: Symmetric array of calculated distances
    """
    euclideanDistanceMatrix = []
    for var1 in range(len(data)):  # Select variable
        tmp = []
        for var2 in range(len(data)):  # calculate distances between selected variable and other variables
            tmp.append(
                np.sqrt(
                    np.sum(
                        np.square(
                            np.array(data[var1]) - np.array(data[var2])
                        )
                    )))
        euclideanDistanceMatrix.append(tmp)
    return np.array(euclideanDistanceMatrix)


def manhattanDistance(data: np.ndarray):
    """
    Calculates Manhattan distances between objects in data array.
    :param data: Input data array to calculate distances on
    :return: Symmetric array of calculated distances
    """
    manhattanDistanceMatrix = []
    for data1 in range(len(data)):  # Select variable
        tmp = []
        for data2 in range(len(data)):  # calculate distances between selected variable and other variables
            tmp.append(
                np.sum(
                    np.abs(
                        np.array(data[data1]) - np.array(data[data2])
                    )
                ))
        manhattanDistanceMatrix.append(tmp)
    return np.array(manhattanDistanceMatrix)

def makeLinkageMatrix(data: np.ndarray, method='complete'):
    """
    Creates linkage matrix for use in plotting dendrograms using specified method
    :param data: Input data array
    :param method: Method with which to construct the dendrogram. Default: complete
    :return: Array of arrays of clusters for use in dendrogram
    plotting. In the inner arrays indices 0 and 1 specify the clustered objects, index 2 represents the height of the
    cluster and index 3 represents the number of elements in the cluster.
    """

    def zeroRowCol(array: np.ndarray, row, column):
        """
        Zeroes out data at specified rows and columns
        :param array: Input data array
        :param row: Row at which data is zeroed out
        :param column: Column at which data is zeroed out
        :return: Array with data at row and column zeroed out
        """

        array[row] = np.zeros(len(array[row]))
        array[column] = np.zeros(len(array[column]))
        array[:, row] = np.zeros(len(array[row]))
        array[:, column] = np.zeros(len(array[column]))
        return array

    def addCluster(array, obj1, obj2, _method=method, retClusterArray=False):
        """
        Calculates and adds a cluster to the data matrix
        :param array: Input data array
        :param obj1: 1st element subjected to comparison with selected method
        :param obj2: 2nd element subjected to comparison with selected method
        :param _method: Method with which the cluster is defined
        :return: Array containing appended cluster
        """

        index = len(array)
        if _method == 'complete':
            distances = [max((array[obj1, i], array[obj2, i])) for i in range(len(array[obj1]))]
            distances.append(0)
        elif _method == 'single':
            distances = [min((array[obj1, i], array[obj2, i])) for i in range(len(array[obj1]))]
            distances.append(0)
        array = np.insert(array, index, np.zeros(len(array)), 0)
        array = np.insert(array, index, np.zeros(len(array)), 1)
        array[index] = distances
        array[:, index] = distances
        if retClusterArray:
            return array
        else:
            return zeroRowCol(array, obj1, obj2)

    def minDist(data):
        """
        Returns the minimal value and its indices in an array
        :param data: Input data array
        :return: List containing position of the data in input array at indices 0 and 1, value of the data at index 2
        """
        return [*np.where(data == np.min(data[np.nonzero(data)]))[0], np.min(data[np.nonzero(data)])]

    def countObjects(matrix, cluster, ogElem, mapDict):

        # A zero to true switch, because 0 is taken as False in boolean comparisons
        c0 = False
        if cluster[0] == 0:
            c0 = True

        if c0 or cluster[0] and cluster[1] in ogElem:
            n = 2
        elif cluster[0] not in ogElem and cluster[1] in ogElem:
            n = 1
            n += countObjects(matrix, matrix[mapDict[str(cluster[0])]], ogElem, mapDict)
        elif cluster[1] not in ogElem and cluster[0] in ogElem:
            n = 1
            n += countObjects(matrix, matrix[mapDict[str(cluster[1])]], ogElem, mapDict)
        else:
            n = countObjects(matrix, matrix[mapDict[str(cluster[0])]], ogElem, mapDict)
            n += countObjects(matrix, matrix[mapDict[str(cluster[1])]], ogElem, mapDict)
        return n

    workdata = data.copy()
    clusterData = data.copy()
    Z = []
    originalElements = list(range(len(data)))
    clusterZ_Map = {f'{str(len(data) + _)}': _ for _ in range(len(data) - 1)}

    # Cluster all objects and append them into Z
    for i in range(len(workdata), (2 * len(workdata)) - 1):
        Z.append(minDist(workdata))
        workdata = addCluster(workdata, minDist(workdata)[0], minDist(workdata)[1])
        clusterData = addCluster(clusterData, minDist(clusterData)[0], minDist(clusterData)[1], retClusterArray=True)

    # Parse through Z and determine the number of objects in each cluster
    for cluster in Z:
        cluster.append(countObjects(Z, cluster, originalElements, clusterZ_Map))

    return np.array(Z)

if __name__ == '__main__':

    # Moje wykresy
    euclDistArr = euclideanDistance(autoscaleData(Data_IN))
    mhttDistArr = manhattanDistance(autoscaleData(Data_IN))
    # Zapis obliczonych macierzy dystansów
    pd.DataFrame(euclDistArr).to_excel("Euclidean_Distances.xlsx")
    pd.DataFrame(mhttDistArr).to_excel("Manhattan_Distances.xlsx")


    euclZ_Matrix_Complete = makeLinkageMatrix(euclDistArr, 'complete')
    euclZ_Matrix_Single = makeLinkageMatrix(euclDistArr, 'single')
    mhttZ_Matrix_Complete = makeLinkageMatrix(mhttDistArr, 'complete')
    mhttZ_Matrix_Single = makeLinkageMatrix(mhttDistArr, 'single')

    dend_CompleteEuclidean = dendrogram(euclZ_Matrix_Complete)
    plt.title('Complete-linkage Euclidean distances')
    plt.savefig('CLED.png')
    plt.close()

    dend_SingleEuclidean = dendrogram(euclZ_Matrix_Single)
    plt.title('Single-linkage Euclidean distances')
    plt.savefig('SLED.png')
    plt.close()

    dend_CompleteManhattan = dendrogram(mhttZ_Matrix_Complete)
    plt.title('Complete-linkage Manhattan distances')
    plt.savefig('CLMD.png')
    plt.close()

    dend_SingleManhattan = dendrogram(mhttZ_Matrix_Single)
    plt.title('Single-linkage Manhattan distances')
    plt.savefig('SLMD.png')
    plt.close()


    # Wykresy SciPy
    dend_spCompleteEuclidean = dendrogram(linkage(autoscaleData(Data_IN), method='complete', metric='euclidean'))
    plt.title('Scipy Complete Euclidean')
    plt.savefig('spCLED.png')
    plt.close()

    dend_spSingleEuclidean = dendrogram(linkage(autoscaleData(Data_IN), method='single', metric='euclidean'))
    plt.title('Scipy Single Euclidean')
    plt.savefig('spSLED.png')
    plt.close()

    dend_spWardEuclidean = dendrogram(linkage(autoscaleData(Data_IN), method='average', metric='euclidean'))
    plt.title('Scipy Average Euclidean')
    plt.savefig('spWED.png')
    plt.close()

    dend_spCompleteChebyshev = dendrogram(linkage(autoscaleData(Data_IN), method='complete', metric='chebyshev'))
    plt.title('Scipy Complete Chebyshev')
    plt.savefig('spCLCD.png')
    plt.close()

    dend_spSingleChebyshev = dendrogram(linkage(autoscaleData(Data_IN), method='single', metric='chebyshev'))
    plt.title('Scipy Single Chebyshev')
    plt.savefig('spSLCD.png')
    plt.close()

    dend_spWardChebyshev = dendrogram(linkage(autoscaleData(Data_IN), method='average', metric='chebyshev'))
    plt.title('Scipy Average Chebyshev')
    plt.savefig('spACD.png')
    plt.close()

    dend_spCompleteCityblock = dendrogram(linkage(autoscaleData(Data_IN), method='complete', metric='cityblock'))
    plt.title('Scipy Complete Cityblock')
    plt.savefig('spCLCbD.png')
    plt.close()

    dend_spSingleCityblock = dendrogram(linkage(autoscaleData(Data_IN), method='single', metric='cityblock'))
    plt.title('Scipy Single Cityblock')
    plt.savefig('spSLCbD.png')
    plt.close()

    dend_spWardCityblock = dendrogram(linkage(autoscaleData(Data_IN), method='average', metric='cityblock'))
    plt.title('Scipy Average Cityblock')
    plt.savefig('spWCbD.png')
    plt.close()

