import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import integrate
plt.figure(1)
#####################X(t)###########################
func = 4
time = np.arange(-2,0,0.01)
time1 = np.arange(0,2,0.01)
x1 = time/(np.sqrt(func))
x2 = (-0.5)*time1

plt.plot(time, x1)
plt.plot(time1, x2)
plt.show()

