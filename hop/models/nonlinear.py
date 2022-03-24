from tkinter import E
from hop.instance import Instance
from hop.utils import get_cplex_so_path
from hop.results import Results
from hop.tour import Tour
from dataclasses import dataclass
from math import prod
from time import time
from typing import List, Optional
from enum import Enum
import numpy as np
import pyomo.opt as popt
import pyomo.environ as penv
import logging


@dataclass
class NonLinearModelResults(Results):
    obj_type: str
    integer_model: bool
    has_feasible_solution: bool
    original_obj: Optional[float]
    obj_bound: Optional[float]

    def csv_header(self) -> str:
        return f"{super().csv_header()},obj_type,integer_model,has_feasible_solution,original_obj,obj_bound"

    def to_csv(self) -> str:
        return f"{super().to_csv()},{self.obj_type},{self.integer_model},{self.has_feasible_solution},{self.original_obj},{self.obj_bound}"


class NonLinearModelObjectiveFunction(Enum):
    ORIGINAL = 1
    CONCAVE = 2


class NonLinearModel:
    def __init__(self, instance: Instance, **kwargs):
        self.instance = instance

        logging.getLogger('pyomo.core').setLevel(logging.WARNING)

        self.solve_continuous = kwargs.get('solve_continuous', False)
        self.current_tour_insert = kwargs.get('current_tour_insert', None)
        self.current_tour_remove = kwargs.get('current_tour_remove', None)
        self.time_limit = kwargs.get('time_limit', 3600)
        self.obj_type = kwargs.get('obj_type', NonLinearModelObjectiveFunction.ORIGINAL)
        self.add_vi = kwargs.get('add_vi', True)

        if self.current_tour_insert is not None and self.current_tour_remove is not None:
            raise ValueError('Only one between current_tour_insert and current_tour_remove can be not None')

        if self.current_tour_remove is not None:
            assert self.current_tour_remove.n_customers() > 0
            self.__build_model_for_remove()
        else:
            self.__build_model()

    def __build_model_for_remove(self):
        assert self.current_tour_remove is not None

        T = self.current_tour_remove
        V_cst = T.tour[1:-1]
        V_cst = [i for i in V_cst if self.instance.t[0][i] + self.instance.t[i][0] <= self.instance.time_bound]

        assert 0 not in V_cst

        V_all = [0] + V_cst
        V_tb = [i for i in V_cst if i in self.instance.tb_customers()]
        
        tour_cst_indices = range(1, len(V_all))
        tour_cst_indices_without_last = range(1, len(V_all) - 1)

        arcs = list()
        for idx, i in enumerate(V_cst):
            arcs.append((0, i))
            arcs.append((i, 0))

            for j in V_cst[idx+1:]:
                if i == j or j == 0:
                    continue

                arcs.append((i, j))

        self.m = penv.ConcreteModel()
        self.m.v = penv.Set(initialize=V_all)
        self.m.h = penv.Set(initialize=V_tb)
        self.m.a = penv.Set(initialize=arcs)
        self.m.ids1 = penv.Set(initialize=tour_cst_indices)
        self.m.ids2 = penv.Set(initialize=tour_cst_indices_without_last)

        def w_bounds(model, i: int):
            return (0, T.w[T.tour.index(i)] + 0.0001)

        
        if self.solve_continuous:
            self.m.x = penv.Var(self.m.a, domain=penv.NonNegativeReals, bounds=lambda _: (0, 1))
            self.m.y = penv.Var(self.m.v, domain=penv.NonNegativeReals, bounds=lambda _: (0, 1))
        else:
            self.m.x = penv.Var(self.m.a, domain=penv.Binary)
            self.m.y = penv.Var(self.m.v, domain=penv.Binary)
        
        self.m.w = penv.Var(self.m.v, domain=penv.NonNegativeReals, bounds=w_bounds)

        def obj_f_original(model):
            psum = sum(self.instance.p[i] * self.m.y[i] for i in self.m.v)
            wprod = prod(penv.exp(- self.instance.l[i] * self.m.w[i]) for i in self.m.h)
            return psum * wprod

        def obj_f_concave(model):
            psum = sum(self.instance.p[i] * self.m.y[i] for i in self.m.v)
            wsum = sum(self.instance.l[i] * self.m.w[i] for i in self.m.v)
            return penv.log(psum) - wsum

        if self.obj_type == NonLinearModelObjectiveFunction.ORIGINAL:
            self.m.OBJ = penv.Objective(rule=obj_f_original, sense=penv.maximize)
        elif self.obj_type == NonLinearModelObjectiveFunction.CONCAVE:
            self.m.OBJ = penv.Objective(rule=obj_f_concave, sense=penv.maximize)
        else:
            raise ValueError('Objective function type not recognised')

        def out_arcs(model, i: int):
            return sum(self.m.x[arc] for arc in self.m.a if arc[0] == i) == self.m.y[i]
        self.m.out_arcs = penv.Constraint(self.m.v, rule=out_arcs)

        def in_arcs(model, i: int):
            return sum(self.m.x[arc] for arc in self.m.a if arc[1] == i) == self.m.y[i]
        self.m.in_arcs = penv.Constraint(self.m.v, rule=in_arcs)

        def tbound(model):
            return sum(self.instance.t[arc[0]][arc[1]] * self.m.x[arc] for arc in self.m.a) <= self.instance.time_bound
        self.m.tbound = penv.Constraint(rule=tbound)

        def link_wy(model, i: int):
            if i == 0:
                return penv.Constraint.Skip
            else:
                return self.m.w[i] <= self.instance.T[i] * self.m.y[i]
        self.m.link_wy = penv.Constraint(self.m.v, rule=link_wy)

        def link_wx1(model, i: int, j: int):
            if i == 0 or j == 0:
                return penv.Constraint.Skip
            else:
                assert (i,j) in arcs
                assert i != j
                bigM = self.instance.time_bound - self.instance.t[j][0] + self.instance.t[i][j]
                return self.m.w[i] >= self.m.w[j] + self.instance.t[i][j] - bigM * (1 - self.m.x[i, j])
        self.m.link_wx1 = penv.Constraint(self.m.a, rule=link_wx1)

        def link_wx2(model, i: int):
            if i == 0 or (i, 0) not in arcs:
                return penv.Constraint.Skip
            else:
                return self.m.w[i] >= self.instance.t[i][0] * self.m.x[i, 0]
        self.m.link_wx2 = penv.Constraint(self.m.v, rule=link_wx2)

        def preserve_order(model, k_idx: int, h_idx: int):
            if h_idx <= k_idx:
                return penv.Constraint.Skip
            k = self.m.v.at(k_idx)
            h = self.m.v.at(h_idx)
            bigM = self.instance.time_bound - self.instance.t[h][0]
            return self.m.w[k] >= self.m.w[h] - bigM * (2 - self.m.y[k] - self.m.y[h])
        self.m.preserve_order = penv.Constraint(self.m.ids2, self.m.ids1, rule=preserve_order)

        self.m.y[0] = 1
        self.m.y[0].fixed = True

    def __build_model(self):
        vertices = [i for i in range(self.instance.n_vertices) if self.instance.t[i][0] + self.instance.t[0][i] <= self.instance.time_bound]
        arcs = list()

        for i in vertices:
            for j in vertices:
                if i == j:
                    continue
                if self.instance.t[i][j] >= self.instance.time_bound:
                    continue

                arcs.append((i,j))

        self.m = penv.ConcreteModel()
        self.m.v = penv.Set(initialize=vertices)
        self.m.h = penv.Set(initialize=[i for i in vertices if self.instance.l[i] > 0])
        self.m.a = penv.Set(initialize=arcs)

        def w_bounds(model, i: int):
            return (0, self.instance.T[i])

        if self.solve_continuous:
            self.m.x = penv.Var(self.m.a, domain=penv.NonNegativeReals, bounds=lambda _: (0, 1))
            self.m.y = penv.Var(self.m.v, domain=penv.NonNegativeReals, bounds=lambda _: (0, 1))
        else:
            self.m.x = penv.Var(self.m.a, domain=penv.Binary)
            self.m.y = penv.Var(self.m.v, domain=penv.Binary)
        
        self.m.w = penv.Var(self.m.v, domain=penv.NonNegativeReals, bounds=w_bounds)

        def obj_f_original(model):
            psum = sum(self.instance.p[i] * self.m.y[i] for i in self.m.v)
            wprod = prod(penv.exp(- self.instance.l[i] * self.m.w[i]) for i in self.m.h)
            return psum * wprod

        def obj_f_concave(model):
            psum = sum(self.instance.p[i] * self.m.y[i] for i in self.m.v)
            wsum = sum(self.instance.l[i] * self.m.w[i] for i in self.m.v)
            return penv.log(psum) - wsum

        if self.obj_type == NonLinearModelObjectiveFunction.ORIGINAL:
            self.m.OBJ = penv.Objective(rule=obj_f_original, sense=penv.maximize)
        elif self.obj_type == NonLinearModelObjectiveFunction.CONCAVE:
            self.m.OBJ = penv.Objective(rule=obj_f_concave, sense=penv.maximize)
        else:
            raise ValueError('Objective function type not recognised')

        def out_arcs(model, i: int):
            return sum(self.m.x[arc] for arc in self.m.a if arc[0] == i and arc[1] != i) == self.m.y[i]
        self.m.out_arcs = penv.Constraint(self.m.v, rule=out_arcs)

        def in_arcs(model, i: int):
            return sum(self.m.x[arc] for arc in self.m.a if arc[1] == i and arc[0] != i) == self.m.y[i]
        self.m.in_arcs = penv.Constraint(self.m.v, rule=in_arcs)

        def tbound(model):
            return sum(self.instance.t[arc[0]][arc[1]] * self.m.x[arc] for arc in self.m.a) <= self.instance.time_bound
        self.m.tbound = penv.Constraint(rule=tbound)

        def link_wy(model, i: int):
            if i == 0:
                return penv.Constraint.Skip
            else:
                return self.m.w[i] <= self.instance.T[i] * self.m.y[i]
        self.m.link_wy = penv.Constraint(self.m.v, rule=link_wy)

        def link_wx1(model, i: int, j: int):
            if i == 0 or j == 0:
                return penv.Constraint.Skip
            else:
                assert i != j
                assert (i,j) in arcs
                bigM = self.instance.time_bound - self.instance.t[j][0] + self.instance.t[i][j]
                return self.m.w[i] >= self.m.w[j] + self.instance.t[i][j] - bigM * (1 - self.m.x[i, j])
        self.m.link_wx1 = penv.Constraint(self.m.a, rule=link_wx1)

        def link_wx2(model, i: int):
            if i == 0 or (i, 0) not in arcs:
                return penv.Constraint.Skip
            else:
                return self.m.w[i] >= self.instance.t[i][0] * self.m.x[i, 0]
        self.m.link_wx2 = penv.Constraint(self.m.v, rule=link_wx2)

        self.m.y[0] = 1
        self.m.y[0].fixed = True

        if self.add_vi:
            def vi1(model, i: int):
                return self.m.w[i] >= sum(self.instance.t[arc[0]][arc[1]] * self.m.x[arc] for arc in self.m.a if arc[0] == i)
            self.m.vi1 = penv.Constraint(self.m.v, rule=vi1)

            def vi2(model, i: int):
                if self.instance.l[i] > 0:
                    return penv.Constraint.Skip
                else:
                    return self.m.w[i] <= sum(self.instance.t[arc[0]][arc[1]] * self.m.x[arc] for arc in self.m.a) - self.instance.t[i][0]
            self.m.vi2 = penv.Constraint(self.m.v, rule=vi2)

            def vi3(model, i: int):
                if self.instance.l[i] > 0 or (0, i) not in self.m.a:
                    return penv.Constraint.Skip
                else:
                    bigM = self.instance.tprime[i] - self.instance.t[0][i]
                    return self.m.w[i] <= sum(self.instance.t[arc[0]][arc[1]] * self.m.x[arc] for arc in self.m.a) -\
                        self.instance.tprime[i] - bigM * self.m.x[0, i]
            self.m.vi3 = penv.Constraint(self.m.v, rule=vi3)

        if self.current_tour_insert is not None:
            cust_idx = self.current_tour_insert.customer_indices()
            cust_idx_not_last = cust_idx[:-1]
            self.m.tour_custs_not_last = penv.Set(initialize=cust_idx_not_last)

            def preserve_order(model, k: int):
                if (v_k := self.current_tour_insert.tour[k]) in vertices and (v_k1 := self.current_tour_insert.tour[k+1]) in vertices:
                    return self.m.w[v_k] >= self.m.w[v_k1]
                else:
                    return penv.Constraint.Skip
            self.m.current_tour_insert_preserve_order = penv.Constraint(self.m.tour_custs_not_last, rule=preserve_order)

            for i in cust_idx:
                if (v_i := self.current_tour_insert.tour[i]) in vertices:
                    self.m.y[v_i] = 1
                    self.m.y[v_i].fixed = True
                    self.m.w[v_i].setlb(self.current_tour_insert.w[i] - 0.0001)

            for h in range(len(cust_idx)):
                v_h = self.current_tour_insert.tour[h]

                if v_h not in vertices:
                    continue

                for k in range(len(cust_idx)):
                    if k == h or k == h + 1:
                        continue

                    v_k = self.current_tour_insert.tour[k]

                    if v_k not in vertices:
                        continue
                    if (v_k, v_h) not in arcs:
                        continue

                    self.m.x[v_k,v_h] = 0
                    self.m.x[v_k,v_h].fixed = True

    def __get_next_vertex(self, current: int) -> int:
        for i in self.m.v:
            if (current, i) not in self.m.a:
                continue

            try:
                if self.m.x[current, i].value > 0.5:
                    return i
            except:
                raise RuntimeError('Trying to access a solution but Baron did not find any feasible solutions')
        else:
            raise RuntimeError(f"Cannot find successor of {current}")

    def __retrieve_solution_tour(self) -> List[int]:
        current = self.__get_next_vertex(0)
        tour = [0, current]

        while (current := self.__get_next_vertex(current)) != 0:
            tour.append(current)

        return tour + [0]

    def __get_continuous_relaxation_original_obj(self) -> float:
        sump = 0
        prodp = 0

        for i in self.m.v:
            sump += self.instance.p[i] * self.m.y[i].value
        for i in self.m.h:
            prodp *= np.exp(- self.instance.l[i] * self.m.w[i].value)

        return sump * prodp

    def solve(self) -> NonLinearModelResults:
        optimiser = popt.SolverFactory('baron')
        optimiser.options['CplexLibName'] = get_cplex_so_path()
        optimiser.options['MaxTime'] = self.time_limit

        start_time = time()
        res = optimiser.solve(self.m, tee=False)
        end_time = time()

        model_name = 'nonlinear_model' if self.obj_type == NonLinearModelObjectiveFunction.ORIGINAL else 'nonlinear_concave_model'
        model_name += '_without_vi' if not self.add_vi else ''

        if res.solver.status == penv.SolverStatus.ok and not (res.solver.termination_condition == penv.TerminationCondition.infeasible):
            if self.solve_continuous:
                return NonLinearModelResults(
                    instance=self.instance,
                    algorithm=f"continuous_{model_name}",
                    integer_model=False,
                    obj=res['problem'][0]['Upper bound'],
                    obj_type=('original' if self.obj_type == NonLinearModelObjectiveFunction.ORIGINAL else 'concave'),
                    obj_bound=None,
                    time_s=(end_time - start_time),
                    tour_object=None,
                    original_obj=self.__get_continuous_relaxation_original_obj(),
                    has_feasible_solution=True
                )
            else:
                tour_l = None

                # Pyomo does not cease to amaze in how lacking it is in support
                # for the most common operations a user might want to run. There
                # is apparently no way to know whether a(ny) feasible solution
                # was produced before the time limit hits. Therefore, I am implementing
                # this hack...
                try:
                    tour_l = self.__retrieve_solution_tour()
                except RuntimeError:
                    pass

                if tour_l is None:
                    tour_object = None
                    original_obj = 0.0,
                else:
                    tour_object = Tour(instance=self.instance, tour=tour_l)
                    original_obj = tour_object.objval()
            
            return NonLinearModelResults(
                instance=self.instance,
                algorithm=model_name,
                integer_model=True,
                obj=res['problem'][0]['Lower bound'],
                obj_type=('original' if self.obj_type == NonLinearModelObjectiveFunction.ORIGINAL else 'concave'),
                obj_bound=res['problem'][0]['Upper bound'],
                time_s=(end_time - start_time),
                tour_object=tour_object,
                original_obj=original_obj,
                has_feasible_solution=(
                    tour_l is not None
                )
            )
        else:
            self.m.write('infeasible-model.bar', io_options=dict(symbolic_solver_labels=True))
            raise RuntimeError(f"Baron error. Status: {res.solver.status}, Termination Condition: {res.solver.termination_condition}")
