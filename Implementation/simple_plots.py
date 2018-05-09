import numpy as np 
import matplotlib.pyplot as plt 

with open('./loss.txt') as f:
    txt = f.read()
txt = txt.split("\n")
txt = np.array([float(x) for x in txt if x != ''])

plt.plot(txt)
plt.show()