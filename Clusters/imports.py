import numpy as np
import random as r
import matplotlib.pyplot as plt
import pandas as pd
import png
import cv2
import cProfile
import re
import os
from distutils.core import setup
from Cython.Build import cythonize
from load_data import *
from prep_data import *
from naive_kmeans import *
from myplotting import *
