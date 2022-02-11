import numpy as np
import copy
import ggpeps.utils as utils

class Measurement:
    use_rebinning=True

    def __init__(self,name,binsize):
        self.counter=0
        self.name=name
        self.binsize=binsize
        self.acc=None
        self.datavec=[]

    def append(self,data):
        #We assume that the data stored in one measurement is homogeneous
        if self.counter==0:
            #We set the first element
            self.acc=copy.deepcopy(data)
        else:
            #Subsequent elements are added
            self.acc+=data
        if self.counter==self.binsize-1:
            #We just filled up the array
            self.datavec.append(self.acc/self.binsize)
            self.counter=0
        else:
            self.counter+=1

    def extend(self,data):
        self.datavec.extend(data)

    def get_timeseries(self):
        return self.datavec

    def mean(self):
        return np.mean(self.datavec,axis=0)

    def mean_err(self, use_binning=True):
        if use_binning:
            return utils.rebin_eom(self.datavec)
        else:
            return np.std(self.datavec,ddof=1,axis=0)/np.sqrt(len(self.datavec))

    def std(self):
        return np.std(self.datavec,ddof=1,axis=0)

    def var(self):
        return np.var(self.datavec,ddof=1,axis=0)

    def __len__(self):
        return len(self.datavec)

    def __mul__(self,other):
        if type(other) is Measurement:
            if other.counter==self.counter:
                dest=Measurement(self.name+"_x_"+other.name,self.binsize)
                dest.datavec=[x*y for (x,y) in zip(self.datavec,other.datavec)]
                return dest
        else:
            return NotImplemented

    def __add__(self,other):
        if type(other) is Measurement:
            if other.counter==self.counter:
                dest=Measurement(self.name+"_+_"+other.name,self.binsize)
                dest.datavec=[x+y for (x,y) in zip(self.datavec,other.datavec)]
                return dest
        else:
            return NotImplemented

    def __sub__(self,other):
        if type(other) is Measurement:
            if other.counter==self.counter:
                dest=Measurement(self.name+"_-_"+other.name,self.binsize)
                dest.datavec=[x-y for (x,y) in zip(self.datavec,other.datavec)]
                return dest
        else:
            return NotImplemented