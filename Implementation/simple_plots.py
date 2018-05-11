import numpy as np 
import matplotlib.pyplot as plt 

with open('./epoch_loss.txt') as f:
    txt = f.read()
txt = txt.split("\n")
txt = np.array([float(x) for x in txt if x != ''])

plt.plot(txt)
plt.xlabel("Num Epochs")
plt.ylabel("Binary-Cross Entroy Loss")
plt.ylim([0.35,0.8])
plt.savefig("./epoch_loss.png")
plt.show()