import numpy as np
import pysindy as ps
import pandas as pd
from pysindy.feature_library import FourierLibrary

t = np.linspace(0, 1, 100)
x = 3 * np.exp(-2 * t) + 3
y = 0.5 * np.exp(t)
X = np.stack((x, y), axis=-1)  # First column is x, second is y

model = ps.SINDy(feature_names=["x", "y"], feature_library=FourierLibrary())
model.fit(X, t=t)
