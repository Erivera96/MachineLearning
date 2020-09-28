from imports import *

# Set array with data files
files = np.array(['mnist_train.csv', 'mnist_test.csv'])

# Read the data in
df = [pd.read_csv(files[i]) for i in range(len(files))]

# Concatenate the data into one dataframe
digits = pd.concat(df, axis=0, sort=False)

# Display some basic information about the data
#print("Shape: ", digits.shape, "\nColumn Names: ", digits.columns, "\nStatistics: ", digits.describe())

# Turn the dataframe into a numpy array (to do dimred on)
digits_array = np.array(digits)

'''
# Visualize the data

fig = px.histogram(digits, x = 'label')
fig.show()

for i in range(5):
    fig2 = px.imshow(digits_array[i,1:].reshape(28,28))
    fig2.show()
'''

rows, cols = digits_array.shape

# Standard Principal Component Analysis
digits_PCA = SPCA(digits_array, feature_removal="set_dimension", xdimension=floor(cols*.6))

print("After pca: ",digits_PCA.shape, "\n")

print(digits_PCA)

# Visualize transformed data

plt.hist(digits_PCA[:,0])
plt.show()
