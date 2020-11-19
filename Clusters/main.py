from imports import *
CWD = os.getcwd()

# Load the uncleaned data
digits_dataframe = LoadData('{}/mnist/mnist_test.csv'.format(CWD))

# How many clusters there should be
number_of_labels = 3

# Number of labels * the number of images per label = total number of images
number_of_images_per_label = 100

# Weed out some data (too much data!!!???)
labels, digits_array = PrepData(digits_dataframe, number_of_labels, number_of_images_per_label)

# Show the original data dimensions for later comparison
print("Original data shape: ", digits_array.shape)

# Convert the color values into images (this is for thresholding)
CreateImages(digits_array)

# Convert grayscale images to black and white
binary_digits_array = BinaryThresholding(digits_array)

# how many nXn should the figure be?
grid_of_images = 3

# Save the images to see the difference
print("Saving image figure...")
MyPlotImages(grid_of_images, digits_array, "Origial")
MyPlotImages(grid_of_images, binary_digits_array, "Binary")

# Convert the black and white back to array and make sure the dimensions match with original
print("Cleaned data shape: ", binary_digits_array.shape)

# Doing Kmeans using pure python
#py_cent, py_bins = NaiveKMeans(binary_digits_array, 2)

# KMeans using Cython
print("Starting kmeans...")
start = time.time()
cy_cent, cy_bins = NaiveKMeans(binary_digits_array, number_of_labels)
end = time.time()

print('Total time: ',(end-start),' sec')

# Combine data to labels and turn to datagrame (for plotting purposes)
data_reunited = np.column_stack((labels, binary_digits_array))
df = pd.DataFrame(data=data_reunited, columns=digits_dataframe.columns)

print(labels)
print(cy_bins)

# Save a picture of how well it did (work in progress)
print("Saving scatter...")
PlotClusters(cy_bins, data_reunited, "eval")
