import numpy as np 
from sklearn.metrics import confusion_matrix

class ConfusionMatrix():
    def __init__(self, labels):
        self.labels = labels
        self._confusion_matrix = None 

    def update_matrix(self, predictions, ground_truth):
        """
        predictions => [10,300,300]
        """
        if predictions.size() != ground_truth.size():
            print("***** Error")

        num_pred = predictions.size()[0]
        for i in range(0, num_pred):
        