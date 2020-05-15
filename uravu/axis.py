import numpy as np
from uravu import UREG
from uravu.distribution import Distribution


class Axis:
    def __init__(self, values):
        self.values = values
        self.ranges = np.zeros((len(values)))
        if isinstance(self.values[0], Distribution):
            for i, v in enumerate(values):
                self.ranges[i] = np.log(v.con_int[1] - v.con_int[0])


    @property
    def n(self):
        v = np.zeros(self.shape)
        if isinstance(self.values[0], Distribution):
            for i, o in enumerate(self.values):
                v[i] = o.n
            return v 
        else:
            return self.values
    
    @property
    def s(self):
        if isinstance(self.values[0], Distribution):
            dv = np.zeros((2, self.size))
            for i, o in enumerate(self.values):
                dv[0, i] = o.n - o.con_int[0]
                dv[1, i] = o.con_int[1] - o.n
            return dv 
        else:
            return np.zeros(self.shape)

    @property
    def size(self):
        if isinstance(self.values[0], Distribution):
            return len(self.values)
        else:
            return self.values.size

    @property
    def shape(self):
        if isinstance(self.values[0], Distribution):
            return len(self.values)
        else:
            return self.values.shape
    
