from __future__ import annotations
from hop.instance import Instance
from typing import List, Optional
import numpy as np


class RelaxedTCLabel:
    predecessor = None
    current_v = 0
    previous_v = None
    n_visited = 0
    profit = 0.0
    travel_time = 0.0
    prob = 1.0
    obj = 0.0
    unexplored = True
    
    def __init__(self, instance: Instance):
        self.instance = instance
        
    def extend(self, j:int) -> Optional[RelaxedTCLabel]:
        """ Extends the current label adding vertex j at the beginning of the partial path. """
        
        # Do not extend using a loop.
        if j == self.current_v:
            return None

        # Do not create a 2-cycle. (Only exception: to go back to the depot).
        if self.previous_v != 0 and j == self.previous_v:
            return None
        
        # We already visited all nodes.
        if self.n_visited == self.instance.n_vertices:
            return None
        
        i = self.current_v
        
        # We would violate the travel time bound.
        if self.travel_time + self.instance.t[j][i] + self.instance.t[0][j] > self.instance.time_bound:
            return None
        
        l = RelaxedTCLabel(self.instance)
        l.predecessor = self
        l.current_v = j
        l.previous_v = self.current_v
        l.n_visited = self.n_visited + 1
        l.profit = self.profit + self.instance.p[j]
        l.travel_time = self.travel_time + self.instance.t[j][i]
        l.prob = self.prob * np.exp(- self.instance.l[j] * l.travel_time)
        l.obj = l.profit * l.prob
        
        if j == 0:
            l.unexplored = False
        
        return l
    
    def __str__(self) -> str:
        return f"[v = {self.current_v}, Σ = {self.n_visited}, p = {self.profit:.2f}, t = {self.travel_time:.2f}, η = {self.prob:.2f}, obj = {self.obj:.2f}, unexp = {self.unexplored}]"
    
    def get_path(self) -> List[int]:
        current_label = self
        path = list()
        
        while current_label is not None:
            path.append(current_label.current_v)
            current_label = current_label.predecessor
            
        return path


def relaxedtc_dominance(l1: RelaxedTCLabel, l2: RelaxedTCLabel) -> bool:
    """ Does l1 dominate l2? """

    if l1.previous_v != l2.previous_v:
        return False
    
    weak_dominance = \
            l1.n_visited <= l2.n_visited and \
            l1.travel_time <= l2.travel_time and \
            l1.profit >= l2.profit and \
            l1.prob >= l2.prob
    
    if not weak_dominance:
        return False
    
    at_least_one_strict = \
            l1.n_visited < l2.n_visited or \
            l1.travel_time < l2.travel_time or \
            l1.profit > l2.profit or \
            l1.prob > l2.prob
    
    return at_least_one_strict
