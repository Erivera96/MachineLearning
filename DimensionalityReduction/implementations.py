from imports import *

X = r.rand(10,4)

#X_PCA = SPCA(X, feature_removal="set_dimension", dimensions=(5,2))
X_PCA = SPCA(X, feature_removal="set_threshold")

print("Shape of X: ",str(X.shape),"\n",X,"\n")
print("Shape of X_PCA: ",str(X_PCA.shape),"\n",X_PCA)
