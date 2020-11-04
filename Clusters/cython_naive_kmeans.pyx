# cython: profile=True

from imports import *

DTYPE = np.intc

# The K-means algorithm function
def NaiveKMeans(pydata, k, eps= 1e-4, max_iter=1e3):
    cdef unsigned short [:,:] data = pydata
    return CythonNaiveKMeans(data, k, eps, max_iter)

cdef CythonNaiveKMeans(unsigned short [:,:] data, int k, float eps, int max_iter):
    
    # Decalre all variables
    cdef Py_ssize_t i, j
    cdef int index = 0
    cdef int centroid_max_diff = 1000000000 # 9 zeros
    
    cdef Py_ssize_t num_points = data.shape[0]
    cdef Py_ssize_t dims = data.shape[1]

    pybins = np.zeros((num_points,), dtype=DTYPE)
    cdef int [:]  bins = pybins

    pybounds = np.zeros((2,dims), dtype=DTYPE)
    cdef int [:,:] bounds = pybounds

    pycentroids = np.zeros((k,dims), dtype=DTYPE)
    cdef int [:,:] centroids = pycentroids

    pycentroids_prime = np.zeros((k,dims), dtype=DTYPE)
    cdef int [:,:] centroids_prime = pycentroids_prime

    pyrandos = np.full((dims,), np.random.rand(), dtype=DTYPE)
    cdef int [:] randos = pyrandos

    # Find the min and max of each dimension
    for i in range(dims):
        bounds[0, i] = np.min(data[:, i])
        bounds[1, i] = np.max(data[:, i])
    
    # Initialize k centroids randomly
    for i in range(k):
        for j in range(dims):
            centroids[i, j] = bounds[0, j] + (bounds[1, j] - bounds[0, j])*np.random.rand()
    
    # Starting K-Means
    while index < max_iter and centroid_max_diff > eps:
        
        # Step 1: Want to assign data to centroids
        for i in range(num_points):
            bins[i] = AssignToCluster(data[i,:], k, centroids)

        # Step 2: Recalculate centroid position for every cluster
        centroids_prime = CalculateCentroids(data, bins, k, bounds)
 
        # Step 3: See if they have converged
        centroid_max_diff = np.max(np.abs(np.subtract(centroids_prime, centroids)))
        
        # New centroids
        centroids = centroids_prime
 
        # Increment index
        index += 1
 
        if index%10 == 0:
            print("index: ",index)

    #pycentroids = centroids

    #pybins = bins

    return pycentroids, pybins

cdef int AssignToCluster(unsigned short [:] data_row, int k, int [:,:] centroids):
    
    # Initialize variables
    cdef Py_ssize_t i
    
    pyretarr = np.zeros((k,), dtype=DTYPE)
    cdef int [:] retarr = pyretarr

    # Loop through all the points for each centroid and calculate their euclid dist
    for i in range(k):
        retarr[i] = EuclidianDistance(data_row, i, centroids[i, :])

    return np.argmin(retarr)

cdef int EuclidianDistance(unsigned short [:] p1, int kth, int [:] p2):
    
    # Initialize variables
    cdef int value, retval
    
    retval = 0

    cdef Py_ssize_t row = p1.shape[0]
    cdef Py_ssize_t dims = p1.shape[1]

    for value in range(row):
 
        retval += (p1[value] - p2[value])*(p1[value] - p2[value])
    
    retval = np.sqrt(retval)
 
    return retval

cdef CalculateCentroids(unsigned short [:,:] data, int [:] bins, int K, int [:,:] bounds):
 
    # Initialize variables
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t num_points = data.shape[0]
    cdef Py_ssize_t dims = data.shape[1]

    pycentroids = np.zeros((K,dims), dtype=DTYPE)
    cdef int [:,:] centroids = pycentroids

    pypoints = np.zeros((K,dims), dtype=DTYPE)
    cdef int [:,:] points = pypoints

    # For each bin, for every point in that bin, those points belong to that cluster
    for i in range(K): # max_bins + 1 = k
        for j in range(num_points):
            if bins[j] == i:
                for k in range(dims):
                    points[i,k] = data[j,k]
                
        # If no points belong to this centroid, rerandomize the centroid, else calc with mean
        if points.size == 0:
            for j in range(dims):
                centroids[i, j] = bounds[0, j] + (bounds[1, j] - bounds[0, j])*np.random.rand()
        else:
            for j in range(dims):
                for k in range(K):
                    centroids[i,j] = np.mean(points[:, k]) 
    
    return centroids
