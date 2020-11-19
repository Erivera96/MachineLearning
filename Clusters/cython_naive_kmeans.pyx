#cython: profile=True
#cython: language_level=3 

from imports import *

DTYPE = np.double

# The K-means algorithm function
def NaiveKMeans(pydata, k, eps= 1e-6, max_iter=1e5):
    cdef double [:,:] data = pydata.astype(DTYPE)
    return CythonNaiveKMeans(data, k, eps, max_iter)

cdef CythonNaiveKMeans(double [:,:] data, int k, double eps, long max_iter):
    
    # Decalre all variables
    cdef Py_ssize_t i, j
    cdef long index = 0
    cdef long centroid_max_diff = 1000000000 # 9 zeros

    cdef Py_ssize_t num_points = data.shape[0]
    cdef Py_ssize_t dims = data.shape[1]

    # Set up the arrays and views
    pybins = np.zeros((num_points,), dtype=DTYPE)
    cdef double [:]  bins = pybins

    pybounds = np.zeros((2,dims), dtype=DTYPE)
    cdef double [:,:] bounds = pybounds

    pycentroids = np.zeros((k,dims), dtype=DTYPE)
    cdef double [:,:] centroids = pycentroids

    pycentroids_prime = np.zeros((k,dims), dtype=DTYPE)
    cdef double [:,:] centroids_prime = pycentroids_prime

    pyrandos = np.full((dims,), np.random.rand(), dtype=DTYPE)
    cdef double [:] randos = pyrandos

    # Find the min and max of each dimension
    for i in range(dims):
        bounds[0, i] = np.min(data[:, i])
        bounds[1, i] = np.max(data[:, i])
    
    # Initialize k centroids randomly
    for i in range(k):
        for j in range(dims):
            centroids[i, j] = bounds[0, j] + (bounds[1, j] - bounds[0, j])*np.random.rand()
    
    #print("Initial centroids: ",pycentroids)

    # Starting K-Means
    while index < max_iter and centroid_max_diff > eps:
        
        # Step 1: Want to assign data to centroids
        for i in range(num_points):
            bins[i] = AssignToCluster(data[i,:], k, centroids)

        #print("Bins: \n",pybins)

        # Step 2: Recalculate centroid position for every cluster
        CalculateCentroids(data, bins, centroids_prime, k, bounds)

        #print("Centroids Prime: \n",pycentroids_prime)

        # Step 3: See if they have converged
        centroid_max_diff = np.max(np.abs(np.subtract(centroids_prime, centroids)))
        
        #print("The difference of points with centroids: \n",centroid_max_diff)

        # New centroids
        #for i in range(k):
            #for j in range(dims):
        centroids[i,j] = centroids_prime[i,j]
        
        #print("Centroids: \n",pycentroids)
        
        if (index % 1000) == 0:
            print("Index: ",index, "    Error: ",centroid_max_diff)
        
        # Increment index
        index += 1
 
    return pycentroids, pybins

cdef double AssignToCluster(double [:] data_row, int k, double [:,:] centroids):
    
    # Initialize variables
    cdef Py_ssize_t i
    
    pyretarr = np.zeros((k,), dtype=DTYPE)
    cdef double [:] retarr = pyretarr

    # Loop through all the points for each centroid and calculate their euclid dist
    for i in range(k):
        retarr[i] = EuclidianDistance(data_row, centroids[i, :])

    return np.argmin(retarr)

cdef double EuclidianDistance(double [:] p1,  double [:] p2):
    
    # Initialize variables
    cdef int value
    cdef double retval

    retval = 0.0
    cdef Py_ssize_t dims = p1.shape[0]

    # For each dimension, find the distance of that point and that cluster
    for value in range(dims):
 
        retval += (p1[value] - p2[value])*(p1[value] - p2[value])
    
    retval = np.sqrt(retval)
 
    return retval

cdef CalculateCentroids(double [:,:] data, double [:] bins, double [:,:] centroids_prime, int K, double [:,:] bounds):
 
    # Initialize variables
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t num_points = data.shape[0]
    cdef Py_ssize_t dims = data.shape[1]
    
    cdef double sum, mean, num_points_in_cluster

    sum = 0.0
    mean = 0.0

    pypoints_in_cluster = np.zeros((num_points,dims), dtype=DTYPE)
    cdef double [:,:] points_in_cluster = pypoints_in_cluster

    # For each bin, for every point in that bin, those points belong to that cluster
    for i in range(K): # max_bins + 1 = k
        
        num_points_in_cluster = 0.0 # reset the number of points inside cluster to 0 

        # for all points, every dimension, if the current cluster is the same as the bin for each point,
        # add those points and all their dimentions to the points in cluster otherwise set it to 0,
        # so that the points in cluster != data
        for j in range(num_points):
            for r in range(dims):
                if bins[j] == i:
                    points_in_cluster[j,r] = data[j,r]
                    num_points_in_cluster += 1
                else:
                    points_in_cluster[j,r] = 0.0

        # If no points belong to this centroid, rerandomize the centroid
        if np.sum(points_in_cluster) == 0:
            for j in range(dims):
             centroids_prime[i, j] = bounds[0, j] + (bounds[1, j] - bounds[0, j])*np.random.rand()
        # Otherwise, sum all points for each dimension and get the average, and set that as the value of the
        # of the cluster
        else:
            for j in range(dims):
                for r in range(num_points):
                    sum += points_in_cluster[r,j]
                mean = sum/num_points_in_cluster
                centroids_prime[i, j] = mean
     
    '''
    print('centroids: ')
    for i in range(K):
        for j in range(dims):
            print(centroids[i,j],end=' ')
        print()
    '''
    return
