from __future__ import annotations
from hop.instance import Instance
from typing import List, Optional
import numpy as np


class StrongLabel:
    predecessor = None
    current_v = 0
    visited = list()
    profit = 0.0
    travel_time = 0.0
    prob = 1.0
    obj = 0.0
    unexplored = True
    
    def __init__(self, instance: Instance):
        self.instance = instance
        self.visited = [0] * instance.n_vertices
        
    def extend(self, j: int) -> Optional[StrongLabel]:
        """ Extends the current label adding vertex j at the beginning of the partial path. """

        # Do not extend using a loop.
        if j == self.current_v:
            return None

        # We already reached the origin.
        if self.visited[0] == 1:
            return None
        
        # We have already visited j.
        if self.visited[j] == 1:
            return None

        i = self.current_v
        
        # We would violate the travel time bound.
        if self.travel_time + self.instance.t[j][i] + self.instance.t[0][j] > self.instance.time_bound:
            return None
        
        l = StrongLabel(self.instance)
        l.predecessor = self
        l.current_v = j
        l.visited = self.visited.copy()
        l.visited[j] = 1
        l.profit = self.profit + self.instance.p[j]
        l.travel_time = self.travel_time + self.instance.t[j][i]
        l.prob = self.prob * np.exp(- self.instance.l[j] * l.travel_time)
        l.obj = l.profit * l.prob
        
        if j == 0:
            l.unexplored = False
        
        return l
    
    def __str__(self) -> str:
        return f"[v = {self.current_v}, W = {self.visited}, p = {self.profit:.2f}, t = {self.travel_time:.2f}, η = {self.prob:.2f}, obj = {self.obj:.2f}, unexp = {self.unexplored}]"
    
    def get_path(self) -> List[int]:
        current_label = self
        path = list()
        
        while current_label is not None:
            path.append(current_label.current_v)
            current_label = current_label.predecessor
            
        return path


def strong_dominance(l1: StrongLabel, l2: StrongLabel) -> bool:
    """ Does l1 dominate l2? """
    
    weak_dominance = \
            np.all(np.less_equal(l1.visited, l2.visited)) and \
            l1.travel_time <= l2.travel_time and \
            l1.profit >= l2.profit and \
            l1.prob >= l2.prob
    
    if not weak_dominance:
        return False
    
    at_least_one_strict = \
            np.all(np.less(l1.visited, l2.visited)) or \
            l1.travel_time < l2.travel_time or \
            l1.profit > l2.profit or \
            l1.prob > l2.prob
    
    return at_least_one_strict
