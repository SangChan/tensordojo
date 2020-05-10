# 20200428 -> 20200503

from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

#if canImport(PythonKit)
    import PythonKit
#else
    import Python
#endif
print(Python.version)

import TensorFlow
var x = Tensor<Float>([[1, 2], [3, 4]])
print(x + x)