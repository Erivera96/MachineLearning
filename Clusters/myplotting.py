from imports import *

def MyPlotImages(n, img_array, name):

    fig, ax = plt.subplots(n,n, figsize=(10,10))
    
    for i in range(n):
        for j in range(n):
            ax[i, j].imshow(img_array[i*n + j,:].reshape(28, 28))

    fig.savefig("/mnt/c/Users/river/figures/{}.png".format(name))
