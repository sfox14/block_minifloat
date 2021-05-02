import torch
import torch.nn.functional as F
#from qtorch.quant import float_quantize
import numpy as np
from .number import BlockMinifloat, Number
from .block_design import block_design


__all__ = ['block_minifloat_quantize', "quantizer"]


def logr2(data):
    const = torch.zeros_like(data) + 2**(-0.5)
    return torch.log(data)/torch.log(const)

def r2(data):
    return (2**(-0.5))**(data)

def add_r_(data):
    r = torch.rand_like(data)
    data.add_(r)


def block_minifloat_quantize(x, number, rounding="stochastic", tensor_type="x"):

    assert isinstance(x, torch.Tensor), "x is not a single precision Floating Point Tensor"

    # shared exponent
    mean_func = lambda x, dim: torch.mean(x, dim)
    max_func = lambda x, dim: torch.max(x, dim)[0]

    # compute max exponent
    max_exponent = block_design(x, number.tile, tensor_type, max_func) 

    # log
    if number.man == 0:
        i = x * 2**(-max_exponent + number.bias)
        sgn = torch.sign(i)
        #i = torch.log2(torch.abs(i)+1e-60)
        i = logr2(torch.abs(i)+1e-60)
        add_r_(i)
        i.floor_()
        #i = 2**(i)
        i = r2(i)
        i = torch.where(i<=2**(-2**(number.exp+1-1) + 1), torch.zeros_like(i), i)
        i.clamp_(0, 1)
        i = i * sgn
        out = i * 2**(max_exponent-number.bias)
        return out
    
    # fixed
    elif number.exp == 0:
        bits = number.man + 1
        i = x * 2**(-max_exponent + number.bias + bits - 1)
        #i = fixed_point_quantize(i, number.man+1, number.man, rounding=rounding)
        if rounding == "stochastic":
            r = torch.rand_like(i)
            i.add_(r).floor_().clamp_(-2**(bits-1), 2**(bits-1)-1)
        else:
            i.round_().clamp_(-2**(bits-1), 2**(bits-1)-1)
        out = i * 2**(max_exponent-number.bias -bits + 1)
        return out
    
    # minifloat
    else:
        offset = max_exponent - number.emax

        # shared exponent shifting
        shift = 2**(-offset)
        i = x * shift

        # clamping at zero (uses QPyTorch float_quantizer - qtorch doesn't have a zero bit?)
        if (number.flush_to_zero):
            raise NotImplementedError
            #k = float_quantize(i, number.exp, number.man, rounding=rounding)
            #k = torch.where(torch.abs(i)<(2**(number.emin+1)), torch.zeros_like(i), k) # flush to zero
            #out = k * 2**(offset) 
            #return out
        

        # handle subnormal and normal quantization
        emin = number.emin 
        emax = number.emax # number.of_emax
        esbn = 2**(emin+1)
        lsbn = 2**(number.emax)
        mval = 2**(number.man)
        rlim = number.max_number

        sgn = torch.sign(i)
        i = torch.abs(i)
        e = torch.floor(torch.log2(i+1e-60))
        # clamp the exponent
        e.clamp_(emin+1, emax) # emin+1 for subnormal region
        # unpack frac for subnormal and normal region
        ie = i*2**(-e)
        me = 2**(e)
        f = torch.where(i<esbn, ie, ie-1)


        # rounding on frac
        if rounding == "stochastic":
            r = torch.rand_like(f)
            f.mul_(mval).add_(r).floor_()
            clipped = f.clamp_(0, mval)
            clipped.div_(mval).mul_(me)
        else:
            f.mul_(mval).round_()
            clipped.div_(mval).mul_(me)
        # sign magnitude multiplication for subnormal and normal
        k = torch.where(i<esbn, clipped, me+clipped)
        k.clamp_(-rlim, rlim)
        out = sgn * k * 2**(offset)
        return out


def quantizer(forward_number=None, backward_number=None,
              forward_rounding="stochastic", backward_rounding="stochastic",
              clamping_grad_zero=False, backward_hooks=[]):
    """
    Creates a quantization function to support quantizing forward and backward process differently.

    Args:
        - :param: forward_number (qtorch.Number, optional) : the number format used for forward quantization.
                  if is None, the quantization would be a identity mapping.
        - :param: backward_number (qtorch.Number, optional) : the number format used for backward quantization.
                  if is None, the quantization would be a identity mapping.
        - :param: forward_rounding (string) : rounding mode, \"stochastic\" or \"nearest\" (default: \"stochastic\")
        - :param: backward_rounding (string) : rounding mode, \"stochastic\" or \"nearest\" (default: \"stochastic\")
        - :param: clamping_grad_zero (bool) : zero out the gradient of numbers that are being clamped during forward propagation.
                  currently requires forward_number to be a fixed point number.
        - :param: backward_hooks (iterable) : iterable of functions that will be applied to gradients before backward quantization.
                  For example, this can be used to support custom scaling.

    Returns:
        A quantization function as specified (torch.Tensor -> torch.Tensor)
    """
    if forward_number is not None:
        if forward_number.exp == -1 or forward_number.man == -1:
            forward_number = None
    if backward_number is not None:
        if backward_number.exp == -1 or backward_number.man == -1:
            backward_number = None


    for rounding in [forward_rounding, backward_rounding]:
        assert rounding in ["stochastic", "nearest"], "invalid rounding type {:s}".format(rounding)
    for num in [forward_number, backward_number]:
        if num != None: assert isinstance(num, Number)

   
    # forward and backward quantisation functions
    tensor_type = "w" if backward_number is None else "x"
    forward_quant = lambda x, num, rd, tt: block_minifloat_quantize(x, num, rd, tt)
    backward_quant = lambda x, num, rd, tt: block_minifloat_quantize(x, num, rd, tt)  


    class Rounding(torch.autograd.Function):
        @staticmethod
        def forward(self, x):
            if forward_number==None: return x

            out = forward_quant(x.contiguous(), forward_number, forward_rounding, tensor_type)

            return out.clone()

        @staticmethod
        def backward(self, grad_output):
            if self.needs_input_grad[0]:
                if backward_number == None:
                    grad_input = grad_output
                else:
                    grad_input = backward_quant(grad_output.contiguous(), backward_number, 
                        backward_rounding, tensor_type)
            else:
                grad_input = None

            return grad_input.clone()

    return Rounding.apply


