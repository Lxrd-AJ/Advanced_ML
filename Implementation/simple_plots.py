import numpy as np 
import matplotlib.pyplot as plt 
import json 

# with open('./epoch_loss.txt') as f:
#     txt = f.read()
# txt = txt.split("\n")
# txt = np.array([float(x) for x in txt if x != ''])

# plt.plot(txt)
# 
# plt.ylabel("Binary-Cross Entroy Loss")
# # plt.ylim([0.35,0.8])
# 
# plt.show()

with open("epoch_data_2.json") as f:
    epoch_data = json.load(f)

bce_loss = []
mean_iou = []

for k,data in epoch_data.items():
    bce_loss.append( data["loss"] )
    mean_iou.append( data["MeanIOU"] )

plt.plot(bce_loss)
plt.plot(mean_iou)
plt.legend(["Binary Cross Entropy Loss","Mean IOU"])
plt.xlabel("Num Epochs")
plt.savefig("./epoch_data_2.png")
plt.show()