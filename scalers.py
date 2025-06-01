
import numpy as np


class NormalScaler:
    
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, x):
        self.mean = x.mean(0)
        self.std = x.std(0)
        
    def transform(self, x):
        return (x-self.mean) / self.std
    
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
