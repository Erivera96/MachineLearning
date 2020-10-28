#QT_STYLE_OVERRIDE=""

from imports import *

CWD = os.getcwd()

digits_dataframe = LoadData('{}/mnist/mnist_test.csv'.format(CWD))

number_of_labels = 2
number_of_images_per_label = 10
labels, digits_array = PrepData(digits_dataframe, number_of_labels, number_of_images_per_label)

grid_of_images = 3

CreateImages(digits_array)

binary_digits_array = BinaryThresholding(digits_array)

print("Saving image figure...")
MyPlotImages(grid_of_images, digits_array, "Origial")
MyPlotImages(grid_of_images, binary_digits_array, "Binary")

print(binary_digits_array.shape)

#py_cent, py_bins = NaiveKMeans(binary_digits_array, 2)


print("Starting kmeans...")
start = time.time()
cy_cent, cy_bins = CythonNaiveKMeans(binary_digits_array, number_of_labels)
end = time.time()

print('Total time: ',(end-start),' sec')

#cy_bins = np.random.randint(2, size=binary_digits_array.shape[0])

data_reunited = np.column_stack((labels,binary_digits_array))

df = pd.DataFrame(data=data_reunited, columns=digits_dataframe.columns)

print("Saving scatter...")
PlotClusters(cy_bins, data_reunited, "eval")
