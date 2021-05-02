"""
Code modified from Qpytorch repository. https://github.com/Tiiiger/QPyTorch/blob/master
"""

__all__ = ['BlockMinifloat']

class Number:
    def __init__(self):
        pass

    def __str__(self):
        raise NotImplemented

class BlockMinifloat(Number):
    """
    Low-Precision Block Minifloat (BM) Format.

    We set the exponent bias to be :math:`2^{exp-1}`. In our simulation, we do
    not handle denormal/subnormal numbers and infinities/NaNs. For rounding
    mode, we apply *round to nearest even*.

    Args:
        - :attr: `exp`: number of bits allocated for exponent
        - :attr: `man`: number of bits allocated for mantissa, referring to number of bits that are
                        supposed to be stored on hardware (not counting the virtual bits).
        - :attr: `tile`: tile dimensions for the shared exponent 
    """
    def __init__(self, exp, man, tile=-1, flush_to_zero=False):
        assert 8 >= exp >= -1, "invalid bits for exponent:{}".format(exp)
        assert 23 >= man >= -1, "invalid bits for mantissa:{}".format(man)
        self.exp = exp
        self.man = man
        self.tile = tile
        self.emax = 2**(exp)-1 - 2**(exp-1)
        self.emin = -2**(exp-1)
        self.max_number = 2**(self.emax)*(2-2**(-self.man))
        self.flush_to_zero = flush_to_zero
        #self.of_emax = self.emax
        #self.of_man = self.man 


    def __str__(self):
        if (self.exp == -1 or self.man == -1):
            return "Default (pytorch fp32)"
        else:
            return "BlockMinifloat (exponent={:d}, mantissa={:d}, tile={:d})".format(
                self.exp, self.man, self.tile)

    def __repr__(self):
        if (self.exp == -1 or self.man == -1):
            return "Default (pytorch fp32)"
        else:
            return "BlockMinifloat (exponent={:d}, mantissa={:d}, tile={:d})".format(
                self.exp, self.man, self.tile)


