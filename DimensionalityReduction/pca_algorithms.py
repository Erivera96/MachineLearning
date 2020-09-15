from imports import *

# Standard Principal Component Analysis

def SPCA(features, feature_removal="set_threshold", threshold=1, dimensions=()):

    # STEP 0:
    #         Checking for square data
    
    square = True
    rows = features.shape[0]
    cols = features.shape[1]
    
    if rows != cols:
        square = False

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
    #         Pad matrix if not square
    
    right_pad = False

    if not square:

        if features.shape[0] > features.shape[1]:
            Z_pad = np.pad(Z, ((0,0),(0, rows - cols)), 'constant', constant_values=1)
            right_pad = True
        else:
            Z_pad = np.pad(Z, ((0, cols - rows),(0,0)), 'constant', constant_values=1)
            bottom_pad = True

    # STEP 3.5:
    #         Get covariance matrix

    # -> The simple answer, where C is covariance matrix, is: C = transpose(Z) * Z
    #    Why? Well here's a small example:
    
    #    |a b c|              |a d g|    |a*a  b*d  c*g|
    #    |d e f| * (transpose:|b e h|) = |d*b  e*e  f*h|
    #    |g h i|              |c f i|    |g*c  h*f  i*i|
    
    #    When we multiply this out, we see that the major diagonal is the covariance
    #    of its respective column. The upper and lower triangular halves are the
    #    covariances of the combinations of pairs of feature columns.

    C = np.transpose(Z_pad) * Z_pad

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

    P_vals = np.sort(eigen_values)[::-1]
    print("eigen values: {}".format(P_vals))
    
    sortdex = np.argsort(eigen_values)[::-1]
    P = np.array([eigen_vectors[dex] for dex in sortdex])

    # STEP 5:
    #         Calculate new features
    
    # -> Short answer: C_star = C * P
    #    What's really happening: you shift and rotate the covariance matrix (C) to represent
    #    the axis that will represent it the best (the eigenvectors) in a meaningful way.

    C_star = C * P

    if right_pad:
        C_star = C_star[:, :(C_star.shape[1] - (rows - cols))]
    else:
        C_star = C_star[:(C_star.shape[0] - (cols - rows)), :]

    # STEP 6:
    #         Remove unnecessary features

    # -> First we will note that since the eigenvalues are a measure of importance (variance),
    #    The new proportion of variance would be the sum(eigenvalues of features kept) divided
    #    by sum(all eigenvalues)
    
    # -> Three choices to remove:

    # 1. Set a desired dimension
    # 2. Calculate proportion of variance for each feature and set a threshold
    # 3. Same as 2 but instead plot the cumulative proportion of variance and as you add more
    #    features, once you hit a feature that starts dropping the variance, stop.

    retmat = C_star
    
    if feature_removal == "set_threshold":
        
        index = len(P_vals)
        variance = sum(P_vals[:index])/sum(P_vals)
        
        while variance >= threshold:
            index -= 1
            variance = sum(P_vals[:index])/sum(P_vals)
            print("variance: {} at index: {}".format(variance, index))

        retmat = retmat[:, :index]
    
    elif feature_removal == "set_dimension":
        
        retmat = retmat[:dimensions[0], :dimensions[1]]
    
    #elif feature_removal == "show_variance":
        

    # STEP 7:
    #         Return the new matrix
    return retmat
