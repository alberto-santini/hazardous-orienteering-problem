from __future__ import annotations
from hop.instance import Instance
from hop.results import Results
from hop.tour import Tour
from gurobipy import Env, Model, GRB
from graph_tool.generation import complete_graph
from graph_tool import GraphView
from graph_tool.flow import boykov_kolmogorov_max_flow, min_st_cut
from typing import List
from bidict import bidict
from copy import deepcopy
import numpy as np
import pickle


class InstanceReductor:
    def __init__(self, instance: Instance, mapping: bidict = None, **kwargs):
        self.old_instance = instance

        self.time_limit = kwargs.get('time_limit', 300)
        self.n_threads = kwargs.get('n_threads', 1)

        if mapping is None:
            self.env = None
            self.model = None
            self.keep = self.__solve_op()
            self.mapping = self.__get_mapping()
        else:
            self.mapping = mapping
            self.keep = list(self.mapping.keys())

        self.new_instance = self.__get_reduced_instance()

    def vertex_old_to_new(self, i: int) -> int:
        return self.mapping[i]

    def vertex_new_to_old(self, i: int) -> int:
        return self.mapping.inverse[i]

    def tour_new_to_old(self, tour: Tour) -> Tour:
        old_tour = [self.vertex_new_to_old(i) for i in tour.tour]
        return Tour(instance=self.old_instance, tour=old_tour)

    def results_new_to_old(self, results: Results) -> Results:
        new_res = deepcopy(results)
        new_res.tour_object = self.tour_new_to_old(results.tour_object)
        new_res.obj = new_res.tour_object.objval()
        return new_res

    def save(self, filename: str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump(self.mapping, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename: str, old_instance: Instance) -> InstanceReductor:
        with open(filename, 'rb') as f:
            mapping = pickle.load(f)

        return InstanceReductor(instance=old_instance, mapping=mapping)

    def __get_reduced_instance(self) -> Instance:
        n_vertices = len(self.mapping)
        xs = [self.old_instance.xs[self.vertex_new_to_old(i)] for i in range(n_vertices)]
        ys = [self.old_instance.ys[self.vertex_new_to_old(i)] for i in range(n_vertices)]
        p = [self.old_instance.p[self.vertex_new_to_old(i)] for i in range(n_vertices)]
        l = [self.old_instance.l[self.vertex_new_to_old(i)] for i in range(n_vertices)]

        return Instance(xs=xs, ys=ys, p=p, l=l, time_bound=self.old_instance.time_bound, instance_file=('[REDUCED] ' + self.old_instance.instance_file))

    def __get_mapping(self) -> bidict:
        assert len(self.keep) > 0
        assert 0 in self.keep

        mapping = bidict()

        for idx, i in enumerate(sorted(self.keep)):
            mapping[i] = idx

        return mapping

    def __solve_op(self) -> List[int]:
        self.env = Env()
        self.env.setParam('OutputFlag', 0)
        self.env.setParam('LogToConsole', 0)
        self.model = Model(env=self.env)
        self.model.setParam(GRB.Param.TimeLimit, self.time_limit)
        self.model.setParam(GRB.Param.LazyConstraints, self.n_threads)

        self.G = complete_graph(N=self.old_instance.n_vertices, self_loops=False, directed=True)
        self.include = self.G.new_vertex_property('bool')
        self.G.vertex_properties['include'] = self.include
        self.cap = self.G.new_edge_property('float')
        self.G.edge_properties['cap'] = self.cap

        self.V = range(self.old_instance.n_vertices)
        self.x = self.model.addVars([(i, j) for i in self.V for j in self.V if i != j], vtype=GRB.BINARY, name='x')
        self.y = self.model.addVars(self.V, vtype=GRB.BINARY, name='y')

        # Assume that, in an optimal OP solution, the travel time is very close to the time bound T
        # because the tour uses almost all available time. In this case, the average travel time
        # of a parcel is T/2. Thus, if a TB item exploding only lost its own profit, the expected profit
        # associated with a TB item i would be p[i] * exp(- l[i] * T/2). But a TB customer is likely
        # to be visited around the end of the tour. Therefore, its expected travel time is much smaller
        # than T/2. For example, here we assume that it will be T/4. Moreover, if it explodes, the entire
        # vehicle content is lost. So its profit should be made less attractive because of this. In our
        # code, we divide its profit by 8.
        modified_profits = [(p / 8.0) * np.exp(-l * self.old_instance.time_bound / 4) for (p, l) in zip(self.old_instance.p, self.old_instance.l)]

        # We put a very large travel time to arcs (0, i) where i is a TB item.
        # We cannot tell the OP to place TB customers towards the end of the tour, but at least we can tell
        # it not to put a TB customer as the very first customer!
        modified_travel_times = deepcopy(self.old_instance.t)
        for i in self.old_instance.tb_customers():
            modified_travel_times[0][i] = 9999

        self.model.setObjective(sum(modified_profits[i] * self.y[i] for i in self.V))
        self.model.addConstr((self.y[0] == 1), name='visit_depot')
        self.model.addConstrs((sum(self.x[i, j] for j in self.V if i != j) == self.y[i] for i in self.V), name='out_arcs')
        self.model.addConstrs((sum(self.x[j, i] for j in self.V if j != i) == self.y[i] for i in self.V), name='in_arcs')
        self.model.addConstr((sum(modified_travel_times[i][j] * self.x[i, j] for i in self.V for j in self.V if i != j) <= self.old_instance.time_bound), name='time_bound')

        def callback(_, where):
            if where != GRB.Callback.MIPNODE or self.model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                return

            yval = [self.model.cbGetNodeRel(self.y[i]) for i in self.V]
            self.include = [yv > 0.0001 for yv in yval]
            g = GraphView(self.G, vfilt=self.include)

            for e in g.edges():
                i, j = e.source(), e.target()
                self.cap[e] = self.model.cbGetNodeRel(self.x[i,j])

            v_values = [i for i in np.argsort(yval) if yval[i] > 0.0001 and i != 0]
            source = g.vertex(0)

            for i in v_values:
                target = g.vertex(i)
                res = boykov_kolmogorov_max_flow(g, source=source, target=target, capacity=self.cap)
                flow = res.copy()
                flow.a = self.cap.a - res.a
                tot_flow = sum(flow[e] for e in target.in_edges())

                if tot_flow < yval[i] - 0.0001:
                    cut = min_st_cut(g, source=source, capacity=self.cap, residual=res)
                    subtour = set(j for j in self.V if j != 0 and yval[j] > 0.0001 and cut[g.vertex(j)] == False)
                    not_subtour = set(self.V) - subtour
                    e = g.edge(source, target)
                    self.cap[e] += 1 - sum(self.model.cbGetNodeRel(self.x[j,k]) for j in not_subtour for k in subtour)

                    for k in subtour:
                        self.model.cbLazy(sum(self.x[i, j] for i in subtour for j in not_subtour) >= self.y[k])

        self.model.optimize(callback)

        solution = [i for i in self.V if self.y[i].getAttr(GRB.Attr.X) > 0.0001]

        self.model.dispose()
        self.env.dispose()

        return solution
