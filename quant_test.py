import argparse
import time
import torch
import torch.nn.functional as F
from quant import *


torch.manual_seed(23)
data = (torch.randn(16)*1)
sign = torch.sign(data) 
data = (data**2)*sign

# add some examples to test saturation limits
data = torch.cat([data, torch.tensor([7.666, 6.98, 7.01, 0.00879, 0.0142, 0.0158])])

torch.manual_seed(time.time())


num = BlockMinifloat(exp=3, man=2, tile=-1)
quant_func = quantizer(forward_number=num, forward_rounding="stochastic")

qdata = quant_func(data)


print("Input:", data, "\n-----------------------------")
print("Quant:", qdata, "\n-----------------------------")
print("--------------------------------------")
print("Error:", data-qdata, torch.sum((data-qdata)**2)/(len(data)))


