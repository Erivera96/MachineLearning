QT_STYLE_OVERRIDE=""

from imports import *

digits_dataframe = LoadData('mnist_train.csv')

number_of_labels = 2
labels, digits_array = PrepData(digits_dataframe, number_of_labels)

number_of_images = 2

CreateImages(digits_array)

binary_digits_array = BinaryThresholding(digits_array)

MyPlotImages(number_of_images, digits_array)
MyPlotImages(number_of_images, binary_digits_array)
