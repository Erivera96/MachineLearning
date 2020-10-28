from imports import *

def LoadData(fileName):
    
    print("Loading the data...")

    # assuming data is given in csv type file
    return pd.read_csv(fileName)
