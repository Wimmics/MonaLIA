# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 18:02:18 2019

@author: abobashe
"""

class OneClassToOneHot (object):
    def __init__(self, class_num):
        self.class_num = class_num
        

    def __call__(self, target):
        if isinstance(target, int):
            new_target = [0] * self.class_num
            new_target[target] = 1
            return new_target
                 
        return target