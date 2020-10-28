from imports import *
CWD = os.getcwd()

def MyPlotImages(n, img_array, name):

    fig, ax = plt.subplots(n,n, figsize=(10,10))
    
    for i in range(n):
        for j in range(n):
            ax[i, j].imshow(img_array[i*n + j,:].reshape(28, 28))

    fig.savefig("{}/figures/{}.png".format(CWD,name))

def PlotClusters(bins, binary_digits_array, name):

    fig = px.bar(binary_digits_array, x=bins, color = binary_digits_array[:,0])        
    #fig.savefig("{}/figures/{}.png".format(CWD,name))
    fig.show()
