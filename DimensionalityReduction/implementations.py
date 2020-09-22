from imports import *

x = np.arange(1,101,0.1)
y = np.arange(1,2001, 2) 
z = np.cos(x)
w = y**2
s = np.sin(y)
r = np.sin(2*y)*3
t = 1/x**2


X = np.array([x, y, z, w, s, r, t]).T

#X_PCA = SPCA(X, feature_removal="set_dimension", dimensions=(5,2))
X_PCA = SPCA(X, feature_removal="show_variance")

print("Shape of X: ",str(X.shape),"\n",X,"\n")
print("Shape of X_PCA: ",str(X_PCA.shape),"\n",X_PCA)
