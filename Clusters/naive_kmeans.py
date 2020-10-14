from imports import *
 
# The K-means algorithm function
def NaiveKMeans(P, k, eps=1e-2, max_iter=3e2):
 
    # find how many dimensions there are
    dims = P.shape[1] 
 
    # find the min and max of each dimension
    bounds = bounds = np.array([(np.min(P[:,i]), np.max(P[:,i])) for i in range(dims)])
    
    # Initialize k centroids randomly
    centroids =  []
    
    for i in range(0,k):
        centroids.append(np.array([bounds[i,0] + (bounds[i,1] - bounds[i,0])*np.random.rand() for i in range(dims)]))
    
    centroids = np.array(centroids)
    
    # Starting K-Means
    num_points = P.shape[0]
    bins = None
    iter = 0
    centroid_max_diff = np.inf
 
    while iter < int(max_iter) and centroid_max_diff > eps:
 
        # Step 1: Want to assign data to centroids
        bins = np.array([AssignToCluster(P[row,:], centroids) for row in range(num_points)])
 
        # Step 2: Recalculate centroid position for every cluster
        centroids_prime = CalculateCentroids(P, bins, k, bounds)
 
        # Step 3: See if they have converged
        centroid_max_diff = np.max(np.abs(centroids_prime - centroids))
        
        # new centroids
        centroids = centroids_prime
 
        # increment iter
        iter += 1
 
        if iter%50 == 0:
            print("iter:{}".format(iter))
 
    return centroids, bins

def AssignToCluster(P_row, centroids):
    return np.argmin([EuclidianDistance(P_row, centroids[i,:]) for i in range(centroids.shape[0])])

def EuclidianDistance(p1, p2):
    
    retval = 0
 
    for i in range(p1.shape[0]):
 
        retval += (p1[i] - p2[i])*(p1[i] - p2[i])
    
    retval = np.sqrt(retval)
 
    return retval

def CalculateCentroids(P, bins, k, bounds):
 
    centroids = []
    
    for i in range(0, k): # max_bins + 1 = k
 
       points = np.array([P[j,:] for j in range(P.shape[0]) if bins[j] == i]) # gets all points for a single cluster
 
       # if no points belong to this centroid, rerandomize the centroid, else calc with mean
       if points.size == 0:
           centroids.append(np.array([bounds[i,0] + (bounds[i,1]-bounds[i,0])*np.random.rand() for i in range(P.shape[1])]))
       else:
           centroids.append(np.array([np.mean(points[:,i]) for i in range(P.shape[1])])) 
    
    return np.array(centroids)
