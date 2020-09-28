from imports import *

# Standard Principal Component Analysis

def SPCA(features, feature_removal="set_threshold", threshold=0.4, threshold_round="down",xdimension=0, reduce_instances=False, precision=2):

    # STEP 1:
    #         Structuring the data

    # -> Here we want the mean to actually be 0 so then we can standardize the data

    X = features - np.mean(features)

    # STEP 2:
    #         Standardize the data
    
    # -> Here you question what is 'important' or a good representation
    #    of the features that predict your labels.
    
    # -> Features with higher variance mean that the values of those
    #    features deviate from the mean more. (lower variance means opposite)
    
    # -> If the variance has nothing to do with how the features predict the labels,
    #    then divide each element in a feature column by that feature column's
    #    standard deviation. Call this new feature matrix Z.

    Z = np.transpose(np.array([X[:,j]/np.std(X[:,j]) for j in range(X.shape[1])]))

    # STEP 3:
    #         Get covariance matrix

    # -> The simple answer, where C is covariance matrix, is: C = transpose(Z) * Z
    #    Why? Well here's a small example:
    
    #    |a b c|              |a d g|    |a*a  b*d  c*g|
    #    |d e f| * (transpose:|b e h|) = |d*b  e*e  f*h|
    #    |g h i|              |c f i|    |g*c  h*f  i*i|
    
    #    When we multiply this out, we see that the major diagonal is the covariance
    #    of its respective column. The upper and lower triangular halves are the
    #    covariances of the combinations of pairs of feature columns.
    
    rows, cols = Z.shape
    
    if rows >= cols:
        
        if reduce_instances:
            C = np.dot(Z.T, Z)
        else:
            C = np.dot(Z, Z.T)

    if cols > rows:
        
        if reduce_instances:
            C = np.dot(Z, Z.T)
        else:
            C = np.dot(Z.T, Z)

    # STEP 4:
    #         Calculate the eigen vectors and their corresponding eigen values
    
    # -> Why? From general linear algebra, eigen vectors are vectors in space that
    #    can only be stretched and their eigen values are how much they stretch.
    #    For this, it means in which direction is the variance and how much variance.
    #    By default they are given highest to lowest, but if not, order them so.
    #    Keeping in mind that the eigen value with the highest value has the highest
    #    variance, in other words carries the most information -> is the Principal Component!
    #    Arrange sorted eigen vectors into a matrix P

    eigen_values, eigen_vectors = la.eig(C)

    eigvals = np.sort(eigen_values)[::-1]
    
    sortdex = np.argsort(eigen_values)[::-1]
    eigvecs = np.array([eigen_vectors[dex] for dex in sortdex])

    # STEP 5:
    #         Calculate new features
    
    # -> Short answer: C_star = C * P
    #    What's really happening: you shift and rotate the covariance matrix (C) to represent
    #    the axis that will represent it the best (the eigenvectors) in a meaningful way.

    C_star = np.real(np.dot(C, eigvecs))

    # STEP 6:
    #         Remove unnecessary features

    # -> First we will note that since the eigenvalues are a measure of importance (variance),
    #    The new proportion of variance would be the sum(eigenvalues of features kept) divided
    #    by sum(all eigenvalues)
    
    # -> Three choices to remove:

    # 1. Set a desired dimension
    # 2. Set a threshold for how many primary components you'll keep 
    # 3. Same as 2 but instead plot the cumulative proportion of variance and as you add more
    #    features, once you hit a feature that starts dropping the variance, stop.
    
    if feature_removal.lower() == "set_threshold":
        
        # Find the percentage of principal components you want, floor it, return it
        return C_star[:, :floor(len(eigvals)*threshold)]

    elif feature_removal.lower() == "set_dimension":
        
        # Only return up to desired row and col dimensions
        return C_star[:, :dimension[1]]
    
    elif feature_removal.lower() == "show_variance":
        
        # Calculate the cumulative variance
        cumulative_variance = np.array([sum(eigvals[:i+1])/sum(eigvals) for i in range(len(eigvals))])
        #print("cumulative variance\n{}\n".format(cumulative_variance))

        cumulative_variance = [trunc(cumulative_variance[i]*10**precision) for i in range(len(cumulative_variance))]
        var_sortdex = np.argsort(cumulative_variance)[::-1]
        cumulative_variance = cumulative_variance[::-1]
        print("truncated cumu_var\n{}\n".format(cumulative_variance))
        
        index = [i+1 for i in range(len(cumulative_variance)-1) if cumulative_variance[i] == cumulative_variance[i+1]]
        print("Adding {} principal components\n".format(index))
        
        #plt.plot(eigvals, '*',cumulative_variance)
        #plt.title("Cumulative Aariance as Eigen Values are Added")
        #plt.xlabel("Eigenvalues")
        #plt.ylabel("Cumulative Variance")
        #plt.show()

        return C_star[:, index]

    else:
        print("ERROR: Expected one of the following:\n\t\"set_threshold\"\n\t\"set_dimension\"\n\t\"show_variance\"\nBut got something else.")
        return
