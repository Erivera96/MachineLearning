# cython: profile=True

from imports import *
 
# The K-means algorithm function
def CythonNaiveKMeans(int num_points, int dims, int* data, int k, float eps, int max_iter):
    
    # Decalre all variables
    cdef int i = 0 
    cdef int j = 0
    cdef int index = 0
    cdef int centroid_max_diff = np.inf
    cdef int * bins = <int *>malloc(sizeof(int)*num_points)
    cdef int * bounds = <int *>malloc(sizeof(int)*2*dims)
    cdef int * centroids = <int *>malloc(sizeof(int)*k*dims)

    # Find the min and max of each dimension
    for i in range(dims):
        bounds[0*dims + i] = np.min(data[:, i])
        bounds[1*dims + i] = np.max(data[:, i])
    
    # Initialize k centroids randomly
    for i in range(k):
        for j in range(dims):
            centroids[i*dims + j] = bounds[0*dims + j] + (bounds[1*dims + j] - bounds[0*dims + j])*np.random.rand()
    
    # Starting K-Means
    while index < max_iter and centroid_max_diff > eps:
        
        # Step 1: Want to assign data to centroids
        for i in range(num_points):
            bins[i] = AssignToCluster(i, dims, data[i,:], k, centroids)

        # Step 2: Recalculate centroid position for every cluster
        centroids_prime = CalculateCentroids(data, bins, k, bounds)
 
        # Step 3: See if they have converged
        centroid_max_diff = np.max(np.abs(centroids_prime - centroids))
        
        # New centroids
        centroids = centroids_prime
 
        # Increment index
        index += 1
 
        if index%10 == 0:
            f"index: {index}"
 
    return centroids, bins

def AssignToCluster(int row, int dims, int* data_row, int k, int** centroids):
    
    # Initialize variables
    cdef int i
    cdef int retarr

    # Allocate memory
    i = 0
    retarr = <int *>malloc(k*sizeof(int))

    # Loop through all the points for each centroid and calculate their euclid dist
    for i in range(k):
        retarr[i] = EuclidianDistance(row, dims, data_row, i, centroids[i, :])

    return np.argmin(retarr)

def EuclidianDistance(int row, int dims, int* p1, int kth, int* p2):
    
    # Initialize variables
    cdef int value, retval
 
    # Set values
    value = 0
    retval = 0

    for value in range(row):
 
        retval += (p1[value] - p2[value])*(p1[value] - p2[value])
    
    retval = np.sqrt(retval)
 
    return retval

def CalculateCentroids(int num_points, int dims, int** data, int* bins, int K, int** bounds):
 
    # Initialize variables
    cdef int i, j, k
    cdef int centroids, points
    
    # Set values & allocate memory
    i = 0
    j = 0
    k = 0
    centroids = <int *>malloc(K*sizeof(int))
    points = <int *>malloc(K*sizeof(int))

    # Allocate more memory
    for i in range(K):
        centroids[i] = <int *>malloc(dims*sizeof(int))
        points[i] = <int *>malloc(dims*sizeof(int))

    # For each bin, for every point in that bin, those points belong to that cluster
    for i in range(K): # max_bins + 1 = k
        for j in range(num_points):
            if bins[j] == i:
                for k in range(dims):
                    points[i][k] = data[j][k]
                
        # If no points belong to this centroid, rerandomize the centroid, else calc with mean
        if points.size == 0:
            for j in range(dims):
                centroids[i][j] = bounds[0][j] + (bounds[1][j]-bounds[0][j])*np.random.rand()
        else:
            for j in range(dims):
                
                for k in range(K):
                    centroids[i][j] = np.mean(points[:][k]) 
    
    return centroids
