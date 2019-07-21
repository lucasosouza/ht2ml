"""
    A Spatial Pooler implementation

    TODO:
    - learning - adjusting the permanence values over multiple passes
    - boosting - maybe simply a multiplier to the number of synapses calculated for each cell, based on its previous number of activations - will need to keep track of duty cycles for each cell

"""

import numpy as np 
import torch
import torch.nn as nn
from collections import defaultdict

class SpatialPooler():

    def __init__(self):

        self.input_size = 784
        self.num_columns = 2000
        self.on_percent = 0.04
        self.on_threshold = self.on_percent/2

        self.kcolumns = int(self.num_columns*self.on_percent)
        self.kposition = self.num_columns - self.kcolumns
        self.perm_std = 0.1
        self.perm_threshold = 0.50
        self.perc_of_active_columns = 0.60
        
        # create a mask of active columns
        shape = (self.input_size, self.num_columns)
        active_columns = torch.rand(shape) > self.perc_of_active_columns

        # initialize random permanence, centered around the threshold
        self.permanences = torch.randn(shape) * self.perm_std + self.perm_threshold
        # apply mask of only active columns
        self.permanences = self.permanences * active_columns.float()


    def forward(self, input_sdr):

        input_sdr = torch.tensor(input_sdr)
        # keep only the permanences which are above threshold
        perm_mask = self.permanences > self.perm_threshold
        perm_surviving = self.permanences * perm_mask.float()

        # dot-product between input and permanences
        activation = input_sdr @ perm_surviving
        # select top k columns
        col_threshold, _ = torch.kthvalue(activation.view(-1), self.kposition)
        output_sdr = activation > col_threshold

        return output_sdr


class Classifier():

    def __init__(self):
        self.pooler = SpatialPooler()

    def learn(self, train):

        # pass all images through the pooler and get the SDRs
        rep_dict = defaultdict(list)
        for img, label in zip(*train):
            rep_dict[label].append(self.pooler.forward(img))

        # calculate the unions
        self.unions = []
        for label in range(10):
            reps = rep_dict[label]
            union = None
            for rep in reps:
                if union is None:
                    union = rep
                union = union | rep
            self.unions.append(union)
    
    def predict_one(self, img):

        output_sdr = self.pooler.forward(img)
        overlaps = []
        for union in self.unions:
            overlap = union & output_sdr
            overlap_value = torch.sum(overlap.float()).item()
            overlaps.append(overlap_value)

        return np.argmax(overlaps)

    def predict_many(self, imgs):

        return np.array([self.predict_one(img) for img in imgs])

    def get_accuracy(self, test):

        imgs, labels = test
        preds = self.predict_many(imgs)
        acc = np.sum(preds == labels) / len(labels)
        return acc



# test
# pooler = SpatialPooler()
# input_sdr = torch.rand(pooler.input_size)
# output_sdr = pooler.forward(input_sdr)
# print(torch.sum(output_sdr).item())











