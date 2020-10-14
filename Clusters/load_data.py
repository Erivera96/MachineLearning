from imports import *

def LoadData(fileName):
    
    # assuming data is given in csv type file
    return pd.read_csv(fileName)
