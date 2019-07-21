# networks.py

"""

RNN- network that maintains some kind of state
LSTM, for each element, there is a corresponding hidden state ht which can contain information from arbitrary points earlier in the sequence

Tensor
1st - the sequence
2nd - indexes instances in minibatch
3rd -  indexes elements of the input

The cow jumped (1,1, 3)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)