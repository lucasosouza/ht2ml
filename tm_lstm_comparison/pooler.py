"""
    A Spatial Pooler implementation

    TODO:
    - learning - adjusting the permanence values over multiple passes
    - boosting - maybe simply a multiplier to the number of synapses calculated for each cell, based on its previous number of activations - will need to keep track of duty cycles for each cell

    Some more references: https://discourse.numenta.org/t/yet-another-yet-another-htm-implementation/4831/2

"""

import numpy as np 
import torch
import torch.nn as nn
from collections import defaultdict

# main difference seems to be temporal pooler has cells
# while spatial pooler does not have

class TemporalPooler():
    """ Missing boosting and learning in this SP """

    def __init__(self, input_size=784, num_columns=2000, k_percent=0.1, num_cells=10, 
                 perm_increase=0.05):

        # definitions of size 
        self.input_size = input_size
        self.num_columns = num_columns
        shape_proximal = (self.input_size, self.num_columns)

        # how many cells will be selected in the kWinner
        self.k_percent = k_percent
        self.k = int(self.num_columns*self.k_percent)

        self.perm_std = 0.1
        self.perm_threshold = 0.50
        self.perm_increase = perm_increase

        ######## proximal connections

        # initialize random permanence, centered around the threshold
        self.proximal = torch.randn(shape_proximal) * self.perm_std + self.perm_threshold

        # each column has a receptive field
        self.receptive_field_size = 0.60
        active_columns = torch.rand(shape_proximal) > self.receptive_field_size
        self.proximal = self.proximal * active_columns.float()

        ######## distal connections

        # initiate distal connections
        self.num_cells = num_cells
        total_cells = self.num_cells * self.num_columns
        shape_distal = (total_cells, total_cells)
        self.distal = torch.randn(shape_distal) * self.perm_std + self.perm_threshold

        # in a similar fashion, define a receptive field for the distal cell
        active_columns = torch.rand(shape_distal) > self.receptive_field_size
        self.distal *= active_columns.float()
        # don't allow connections to itself
        self.distal *= (torch.eye(total_cells) == 0).float()

    def forward(self, input_sdr, learning=False):

        #### select which columns will be active based on proximal connections

        # proximal input size 3, num columns 4
        # num columns 4, num cells 2, 8x8 matrix
        spatial_overlap = self.proximal * input_sdr.view(-1,1).float()
        active_proximal = spatial_overlap > self.perm_threshold
        # count sum synapses per column
        total_proximal = torch.sum(active_proximal, dim=0)
        # select topk columns
        kthvalue = self.num_columns - self.k
        vals, _ = torch.kthvalue(total_proximal, kthvalue)
        # selected columns as a binary masl
        selected_cols = (total_proximal > vals.view(-1,1)).squeeze()

        #### select which cells will be active based on distal connections 

        # select permanences above threshold
        active_distal = self.distal > self.perm_threshold
        # sum them per cell - have a list of cells which are active per column
        total_distal = torch.sum(active_distal, dim=0).float()
        # reshape them into a format of column x cell
        distal_cells = total_distal.view(self.num_columns, self.num_cells)
        # pick the top1, per column - add random component to break ties
        epsilon = torch.abs(torch.rand(distal_cells.shape)) * 0.001
        distal_cells += epsilon
        vals, indices = torch.kthvalue(distal_cells, 1)
        # predictive cells include even the ones bursting
        predictive_cells = (distal_cells > vals.view(-1,1)).squeeze()
        # predictive cols as a mask
        predictive_cols = vals > 0.001
        # fired cells doesn't include bursting cells
        firing_cells = predictive_cells & predictive_cols.view(-1, 1)

        #### temporal learning 
        # calculate overlap
        temporal_overlap = predictive_cols * selected_cols
        if learning:
            # if predictive and overlap, increase
            increase_cells = predictive_cells & temporal_overlap.view(-1,1)
            increase = (increase_cells.view(-1).float() * self.perm_increase).view(1, -1)
            # if predictive but not overlap, decrease
            decrease_cells = firing_cells & (temporal_overlap.view(-1,1) == 0)
            decrease = (decrease_cells.view(-1).float() * self.perm_increase).view(1, -1)
            # apply changes
            zero_mask = (self.distal > 0).float()
            self.distal = (self.distal + increase - decrease) * zero_mask
        else:
            print(torch.sum(temporal_overlap).item())

        return selected_cols, predictive_cols, temporal_overlap


class SpatialPooler():
    """ Missing boosting and learning in this SP """

    def __init__(self, input_size=784, num_columns=2000, k_percent=0.1):

        # definitions of size 
        self.input_size = input_size
        self.num_columns = num_columns
        shape = (self.input_size, self.num_columns)

        # how many cells will be selected in the kWinner
        self.k_percent = k_percent
        self.k = int(self.num_columns*self.k_percent)

        # initialize random permanence, centered around the threshold
        # only allow active columns to have permanences
        self.perm_std = 0.1
        self.perm_threshold = 0.50
        self.permanences = torch.randn(shape) * self.perm_std + self.perm_threshold

        # each column has a receptive field 
        # will only have a permanence value if in the receptive field
        self.receptive_field_size = 0.60
        active_columns = torch.rand(shape) > self.receptive_field_size
        self.permanences = self.permanences * active_columns.float()

    def forward(self, input_sdr):

        # select permanences above threshold
        overlap = self.permanences * input_sdr
        active_cols = overlap > self.perm_threshold
        # sum them per column
        totals, _ = torch.max(active_cols, dim=0)
        # select topk columns
        _, topk = torch.topk(totals, self.k)

        return topk


class WeirdSpatialPooler():

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











