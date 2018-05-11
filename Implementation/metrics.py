import numpy as np 
import torch 

def computeJacquard( ground_truth, predictions ):
    # predictions => torch.Size([4, 10, 600, 600])
    predictions = predictions[-1]
    ground_truth = ground_truth[-1]
    # predictions => torch.Size([10, 600, 600])
    num_pred = predictions.size()[0]

    pred = predictions.data.cpu().numpy() if torch.cuda.is_available() else predictions.data.numpy()
    target = ground_truth.data.cpu().numpy() if torch.cuda.is_available() else ground_truth.data.numpy()

    jacquard = np.zeros(num_pred)
    for i in range(num_pred):
        area_predicted = np.count_nonzero( np.round(pred[i]) )
        area_truth = np.count_nonzero( target[i] )
        area_intersection = np.count_nonzero( np.round(pred[i]) * target[i] )
        area_total = area_truth + area_predicted - area_intersection
        jacquard[i] = 0 if area_total == 0 else (area_intersection / area_total) 
    
    return jacquard