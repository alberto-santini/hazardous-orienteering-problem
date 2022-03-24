from __future__ import annotations
from hop.instance import Instance
from typing import List, Optional
import numpy as np


class RelaxedLabel:
    predecessor = None
    current_v = 0
    n_visited = 0
    profit = 0.0
    travel_time = 0.0
    prob = 1.0
    obj = 0.0
    unexplored = True
    
    def __init__(self, instance: Instance):
        self.instance = instance
        
    def extend(self, j: int) -> Optional[RelaxedLabel]:
        """ Extends the current label adding vertex j at the beginning of the partial path. """
        
        # Do not extend using a loop.
        if j == self.current_v:
            return None
        
        # We already visited all nodes.
        if self.n_visited == self.instance.n_vertices:
            return None
        
        i = self.current_v
        
        # We would violate the travel time bound.
        if self.travel_time + self.instance.t[j][i] + self.instance.t[0][j] > self.instance.time_bound:
            return None
        
        l = RelaxedLabel(self.instance)
        l.predecessor = self
        l.current_v = j
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


def relaxed_dominance(l1: RelaxedLabel, l2: RelaxedLabel) -> bool:
    """ Does l1 dominate l2? """
    
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
    RelaxedLabel
    return at_least_one_strict
