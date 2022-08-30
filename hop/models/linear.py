from hop.instance import Instance
from hop.results import Results
from hop.tour import Tour
from enum import Enum
from dataclasses import dataclass
import gurobipy as gp
from gurobipy import GRB, quicksum
from graph_tool import Graph, GraphView
from graph_tool.flow import boykov_kolmogorov_max_flow, min_st_cut
from typing import Iterable, Optional, List
from copy import deepcopy
import numpy as np
import logging


class ObjectiveNotsupportedException(Exception):
    pass


class LinearModelObjectiveFunction(Enum):
    ORIGINAL = 1
    FRANK_WOLFE = 2
    LINEAR_APPROX_LOOSE = 3
    LINEAR_APPROX_TIGHT = 4


@dataclass
class LinearModelResults(Results):
    xvar: dict
    yvar: dict
    wvar: dict
    lift_mtz: bool
    add_vi1: bool
    add_vi2: bool
    add_vi3: bool
    add_vi4:bool

    def csv_header(self) -> str:
        return f"{super().csv_header()},lift_mtz,add_vi1,add_vi2,add_vi3,add_vi4"

    def to_csv(self) -> str:
        return f"{super().to_csv()},{self.lift_mtz},{self.add_vi1},{self.add_vi2},{self.add_vi3},{self.add_vi4}"


@dataclass
class IntegerLinearModelResults(LinearModelResults):
    obj_bound: float
    original_obj_value: float

    def csv_header(self) -> str:
        return f"{super().csv_header()},obj_bound,lb_original_obj_value"

    def to_csv(self) -> str:
        return f"{super().to_csv()},{self.obj_bound},{self.original_obj_value}"


@dataclass
class ContinuousLinearModelResults(LinearModelResults):
    cutting_planes_iters: int
    yvar_np: np.ndarray
    wvar_np: np.ndarray

    def csv_header(self) -> str:
        return f"{super().csv_header()},cutting_plane_iters"

    def to_csv(self) -> str:
        return f"{super().to_csv()},{self.cutting_planes_iters}"


class LinearModel:
    EPS = 0.0001 # Gurobi likes to set to 1e-15 variables which should be at 0!

    def __init__(self, instance: Instance, **kwargs):
        self.instance = instance
        self.env = None
        self.model = None

        self.v = range(self.instance.n_vertices)
        self.v = [i for i in self.v if self.instance.t[0][i] + self.instance.t[i][0] <= self.instance.time_bound]
        self.v0 = [i for i in self.v if i != 0]

        self.integer = kwargs.get('integer', True)
        self.obj_type = kwargs.get('obj_type', LinearModelObjectiveFunction.ORIGINAL)
        self.time_limit = kwargs.get('time_limit', 60)
        self.n_threads = kwargs.get('n_threads', 1)
        
        self.__read_constraints_args(**kwargs)

        logging.getLogger('gurobipy').setLevel(logging.WARNING)

        if self.obj_type == LinearModelObjectiveFunction.FRANK_WOLFE:
            self.set_frank_wolfe_coeff(y_coeff=kwargs.get('y_coeff'), w_coeff=kwargs.get('w_coeff'), rebuild_obj=False)

        self.__build_graph_for_sec_separation()
        self.__build_model()

    def __read_constraints_args(self, **kwargs) -> None:
        self.add_vi = kwargs.get('add_vi', False)
        self.add_vi1 = False
        self.add_vi2 = False
        self.add_vi3 = False
        self.add_vi4 = False

        if self.add_vi:
            self.add_vi1 = True
            self.add_vi2 = True
            self.add_vi3 = True
            self.add_vi4 = True

        if 'add_vi1' in kwargs:
            self.add_vi1 = kwargs.get('add_vi1')
        if 'add_vi2' in kwargs:
            self.add_vi2 = kwargs.get('add_vi2')
        if 'add_vi3' in kwargs:
            self.add_vi3 = kwargs.get('add_vi3')
        if 'add_vi4' in kwargs:
            self.add_vi4 = kwargs.get('add_vi4')

        self.lift_mtz = kwargs.get('lift_mtz', False)

    def set_frank_wolfe_coeff(self, y_coeff: Iterable[float], w_coeff: Iterable[float], rebuild_obj: bool = True) -> None:
        self.y_coeff = y_coeff
        self.w_coeff = w_coeff

        if rebuild_obj:
            self.__build_obj_function()

    def __build_graph_for_sec_separation(self) -> None:
        self.a = []

        for i in self.v:
            for j in self.v:
                if i == j:
                    continue
                if self.instance.t[i][j] >= self.instance.time_bound:
                    continue

                self.a.append((i,j))

        self.graph = Graph(directed=True)
        self.graph.add_vertex(n=self.instance.n_vertices)
        self.graph.add_edge_list(self.a)

        self.include = self.graph.new_vertex_property('bool')
        self.graph.vertex_properties['include'] = self.include
        self.cap = self.graph.new_edge_property('double')
        self.graph.edge_properties['cap'] = self.cap

    def __build_model(self) -> None:
        if self.integer:
            vtype= GRB.BINARY
        else:
            vtype = GRB.CONTINUOUS

        self.env = gp.Env()
        self.env.setParam('OutputFlag', 0)
        self.env.setParam('LogToConsole', 0)

        self.model = gp.Model(env=self.env)

        self.x = self.model.addVars(self.a, lb=0, ub=1, vtype=vtype, name='x')
        self.y = self.model.addVars(self.v, lb=0, ub=1, vtype=vtype, name='y')
        self.w = self.model.addVars(self.v, lb=0, ub=[self.instance.T[i] for i in self.v], vtype=GRB.CONTINUOUS, name='w')

        self.__build_obj_function()
        self.__build_constraints()
        self.__build_valid_inequalities()

    def __build_obj_function(self) -> None:
        if self.obj_type == LinearModelObjectiveFunction.ORIGINAL:
            raise ObjectiveNotsupportedException('Gurobi does not support the original objective function! Use Baron!')

        elif self.obj_type == LinearModelObjectiveFunction.FRANK_WOLFE:
            self.model.setObjective(quicksum([self.y_coeff[i] * self.y[i] for i in self.v0]) +
                                    quicksum([self.w_coeff[i] * self.w[i] for i in self.v0 if self.instance.l[i] > 0]),
                                    sense=GRB.MINIMIZE)

        elif self.obj_type == LinearModelObjectiveFunction.LINEAR_APPROX_LOOSE:
                wc = {i: self.instance.p[i] * (self.instance.pi[i] - 1) / self.instance.T[i] for i in self.v}
                self.model.setObjective(quicksum([self.instance.p[i] * self.y[i] for i in self.v0]) +
                                        quicksum([wc[i] * self.w[i] for i in self.v0 if self.instance.l[i] > 0]),
                                        sense=GRB.MAXIMIZE)

        elif self.obj_type == LinearModelObjectiveFunction.LINEAR_APPROX_TIGHT:
            mu = {i: np.exp(- self.instance.l[i] * self.instance.t[i][0]) for i in self.v}
            alpha = {i: (mu[i] - self.instance.pi[i]) / (self.instance.t[i][0] - self.instance.T[i]) for i in self.v}
            beta = {i: self.instance.pi[i] - self.instance.T[i] * alpha[i] for i in self.v}
            yc = {i: self.instance.p[i] * beta[i] for i in self.v}
            wc = {i: self.instance.p[i] * alpha[i] for i in self.v}

            self.model.setObjective(quicksum([yc[i] * self.y[i] for i in self.v0]) +
                                    quicksum([wc[i] * self.w[i] for i in self.v0 if self.instance.l[i] > 0]),
                                    sense=GRB.MAXIMIZE)

    def __build_constraints(self) -> None:
        self.y[0].setAttr(GRB.Attr.LB, 1)
        self.model.addConstrs((
            (quicksum((self.x[i, int(j)] for j in self.graph.vertex(i).out_neighbors())) == self.y[i]) for i in self.v), name='outgoing')
        self.model.addConstrs((
            (quicksum((self.x[int(j), i] for j in self.graph.vertex(i).in_neighbors())) == self.y[i]) for i in self.v), name='incoming')
        self.model.addConstr(
            quicksum((self.instance.t[i][j] * self.x[i, j]) for i, j in self.a) <= self.instance.time_bound, name='time_bound')
        
        if self.lift_mtz:
            self.model.addConstrs((
                (self.w[i] <= self.instance.T[i] * self.y[i] +
                sum(
                    (self.instance.T[arc[1]] - self.instance.t[arc[1]][arc[0]] + self.instance.t[0][arc[0]]) * self.x[arc]
                    for arc in self.a if arc[0] == i
                ))
                for i in self.v0
            ), name='link_w_y_lifted')

            for i, j in self.a:
                if i == 0 or j == 0:
                    continue

                tij = self.instance.t[i][j]
                tji = self.instance.t[j][i]

                bigM = max((
                    self.instance.t[i][0] - tij,
                    self.instance.T[j] + tij,
                    self.instance.T[i] - self.instance.t[j][0] + tij
                ))

                self.model.addConstr((
                    self.w[i] >= self.w[j] + tij - bigM * (1 - self.x[i,j]) + (bigM - tij - tji) * self.x[j,i]
                ), name=f"link_w_x_lifted_{i}_{j}")
        else:
            self.model.addConstrs((
                (self.w[i] <= self.instance.T[i] * self.y[i]) for i in self.v0), name='link_w_y')
            self.model.addConstrs((
                self.w[i] >= self.w[j] + self.instance.t[i][j] -
                (self.instance.time_bound - self.instance.t[j][0] + self.instance.t[i][j]) * (1 - self.x[i, j])
                for i, j in self.a if i != 0 and j != 0), name='link_w_x')
        self.model.addConstrs((self.w[i] >= self.instance.t[i][0] * self.x[i, 0] for i in self.v0 if (i,0) in self.a), name='link_w_x_depot')

    def __build_valid_inequalities(self) -> None:
        if self.add_vi1:
            self.model.addConstrs((self.w[i] >=
                                quicksum((self.instance.t[i][int(j)] * self.x[i, int(j)]) for j in self.graph.vertex(i).out_neighbors()) for i in self.v),
                                name='vi1')

        if self.add_vi2:
            self.model.addConstrs((self.w[i] <=
                                quicksum((self.instance.t[j][k] * self.x[j, k]) for j, k in self.a) - self.instance.t[0][i] for i in self.v if self.instance.l[i] > 0),
                                name='vi2')

        if self.add_vi3:
            bigM = {i: self.instance.tprime[i] - self.instance.t[0][i] for i in self.v}
            self.model.addConstrs((self.w[i] <=
                                quicksum((self.instance.t[j][k] * self.x[j,k]) for j, k in self.a) - self.instance.tprime[i] + bigM[i] * self.x[0,i] for i in self.v if self.instance.l[i] > 0 and (0,i) in self.a),
                                name='vi3')

    def __sol_int_get_next_vertex(self, i: int) -> Optional[int]:
        for j in self.graph.vertex(i).out_neighbors():
            xval = self.x[i, int(j)].getAttr(GRB.Attr.X)

            if xval > self.EPS:
                return int(j)

        return None

    def __sol_int_get_subtour_starting_at(self, start_v: int) -> Optional[List[int]]:
        if self.y[start_v].getAttr(GRB.Attr.X) < self.EPS:
            return None

        subtour = [start_v]
        current_v = self.__sol_int_get_next_vertex(start_v)

        if current_v is None:
            return subtour

        while current_v != start_v:
            subtour.append(current_v)
            current_v = self.__sol_int_get_next_vertex(current_v)

        return subtour

    def __sol_print_sol(self) -> None:
        for i in self.v:
            print(i)
            yval = self.y[i].getAttr(GRB.Attr.X)

            if yval > self.EPS:
                print(f'y[{i}] = {yval}')

        for i, j in self.a:
            xval = self.x[i,j].getAttr(GRB.Attr.X)

            if xval > self.EPS:
                print(f'x[{i},{j}] = {xval}')

        for i in self.v0:
            wval = self.w[i].getAttr(GRB.Attr.X)

            if wval > self.EPS:
                print(f'w[{i}] = {wval}')

    def __sol_cont_separate(self) -> List[set]:
        return self.__cont_separate(cb=False)

    def __cb_cont_separate(self) -> None:
        subtours = self.__cont_separate(cb=True)

        for subtour in subtours:
            not_subtour = set(self.v) - subtour
            for k in subtour:
                self.model.cbLazy(quicksum(self.x[i, j] for i in subtour for j in not_subtour if (i,j) in self.a) >= self.y[k])

    def __cont_separate(self, cb: bool) -> List[set]:
        if cb:
            yval = {i: self.model.cbGetNodeRel(self.y[i]) for i in self.v}
        else:
            yval = {i: self.y[i].getAttr(GRB.Attr.X) for i in self.v}

        for i in range(self.instance.n_vertices):
            self.include[i] = (i in self.v and yval[i] > self.EPS)

        g = GraphView(self.graph, vfilt=self.include)

        for e in g.edges():
            i, j = e.source(), e.target()

            if cb:
                self.cap[e] = self.model.cbGetNodeRel(self.x[i, j])
            else:
                self.cap[e] = self.x[i, j].getAttr(GRB.Attr.X)

        # Sort yval by value (i.e. by the value of the y variable)
        yval = dict(sorted(yval.items(), key=lambda kv: kv[1]))
        v_values = {i: yv for i, yv in yval.items() if i != 0 and yv > self.EPS}
        source = g.vertex(0)
        sets = list()

        for i, yv in v_values.items():
            target = g.vertex(i)
            res = boykov_kolmogorov_max_flow(g, source=source, target=target, capacity=self.cap)
            flow = res.copy()
            flow.a = self.cap.a - res.a
            tot_flow = sum(flow[e] for e in target.in_edges())

            if tot_flow < yv - self.EPS:
                cut = min_st_cut(g, source=source, capacity=self.cap, residual=res)

                subtour = set(j for j in v_values.keys() if cut[g.vertex(j)] == False)
                not_subtour = set(self.v) - subtour

                sets.append(subtour)

                e = g.edge(source, target)

                if cb:
                    self.cap[e] += 1 - sum(self.model.cbGetNodeRel(self.x[j, k]) for j in not_subtour for k in subtour if (j,k) in self.a)
                else:
                    self.cap[e] += 1 - sum(self.x[j, k].getAttr(GRB.Attr.X) for j in not_subtour for k in subtour if (j,k) in self.a)

        return sets

    def __compute_original_obj(self) -> float:
        obj = sum(self.instance.p[i] * self.y[i].getAttr(GRB.Attr.X) for i in self.v0)
        for i in self.v0:
            if self.instance.l[i] > 0:
                obj *= np.exp(-self.instance.l[i] * self.w[i].getAttr(GRB.Attr.X))
        return obj

    def __solve_integer(self) -> IntegerLinearModelResults:
        def callback(model, where):
            if where == GRB.Callback.MIPNODE and self.model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL and self.add_vi4:
                self.__cb_cont_separate()

        self.model.setParam(GRB.Param.TimeLimit, self.time_limit)
        self.model.setParam(GRB.Param.Threads, self.n_threads)
        self.model.setParam(GRB.Param.LazyConstraints, 1)
        self.model.optimize(callback)

        alg_name = f"integer_linear_model_{self.obj_type}"
        if self.add_vi1:
            alg_name += '_with_vi1'
        if self.add_vi2:
            alg_name += '_with_vi2'
        if self.add_vi3:
            alg_name += '_with_vi3'
        if self.add_vi4:
            alg_name += '_with_vi4'
        if self.lift_mtz:
            alg_name += "_liftMTZ"

        return IntegerLinearModelResults(
            instance=self.instance,
            algorithm=alg_name,
            lift_mtz=self.lift_mtz,
            add_vi1=self.add_vi1,
            add_vi2=self.add_vi2,
            add_vi3=self.add_vi3,
            add_vi4=self.add_vi4,
            tour_object=Tour(instance=self.instance, tour=self.__sol_int_get_subtour_starting_at(0) + [0]),
            obj=self.model.getAttr(GRB.Attr.ObjVal),
            obj_bound=self.model.getAttr(GRB.Attr.ObjBound),
            original_obj_value=self.__compute_original_obj(),
            xvar={(i, j): self.x[i, j].getAttr(GRB.Attr.X) for i, j in self.a},
            yvar={i: self.y[i].getAttr(GRB.Attr.X) for i in self.v},
            wvar={i: self.w[i].getAttr(GRB.Attr.X) for i in self.v},
            time_s=self.model.getAttr(GRB.Attr.Runtime)
        )

    def __solve_continuous(self) -> ContinuousLinearModelResults:
        elapsed_time = 0.0
        previous_iteration_subtours = list()
        cutting_planes_iterations = 0

        self.model.setParam(GRB.Param.Threads, self.n_threads)

        while True:
            self.model.setParam(GRB.Param.TimeLimit, max(self.time_limit - elapsed_time, 0))
            self.model.optimize()
            cutting_planes_iterations += 1

            if self.model.Status == GRB.INFEASIBLE:
                self.model.computeIIS()
                self.model.write(f"Infeasible_LP_{self.obj_type}.mps")
                self.model.write(f"Infeasible_LP_{self.obj_type}.lp")
                raise RuntimeError('Gurobi linear model infeasible!')

            if not self.add_vi4: # If no subtour elimination, quit
                break

            subtours = self.__sol_cont_separate()

            if len(subtours) == 0:
                break

            if subtours == previous_iteration_subtours:
                print('Numerical error: same subtours in two consecutive iterations!')
                break

            for subtour in subtours:
                not_subtour = set(self.v) - subtour

                for k in subtour:
                    self.model.addConstr(quicksum(self.x[i, j] for i in subtour for j in not_subtour if (i,j) in self.a) >= self.y[k])

            previous_iteration_subtours = deepcopy(subtours)

            elapsed_time += self.model.getAttr(GRB.Attr.Runtime)

        yvar_vector = [self.y[i].getAttr(GRB.Attr.X) if i in self.v else 0 for i in range(self.instance.n_vertices)]
        wvar_vector = [self.w[i].getAttr(GRB.Attr.X) if i in self.v else 0 for i in range(self.instance.n_vertices)]
        
        alg_name = f"continuous_linear_model_{self.obj_type}"
        if self.add_vi1:
            alg_name += '_with_vi1'
        if self.add_vi2:
            alg_name += '_with_vi2'
        if self.add_vi3:
            alg_name += '_with_vi3'
        if self.add_vi4:
            alg_name += '_with_vi4'
        if self.lift_mtz:
            alg_name += "_liftMTZ"

        return ContinuousLinearModelResults(
            instance=self.instance,
            algorithm=alg_name,
            lift_mtz=self.lift_mtz,
            add_vi1=self.add_vi1,
            add_vi2=self.add_vi2,
            add_vi3=self.add_vi3,
            add_vi4=self.add_vi4,
            tour_object=None,
            obj=self.model.getAttr(GRB.Attr.ObjVal),
            time_s=self.model.getAttr(GRB.Attr.Runtime),
            cutting_planes_iters=cutting_planes_iterations,
            xvar={(i, j): self.x[i,j].getAttr(GRB.Attr.X) for i, j in self.a},
            yvar={i: self.y[i].getAttr(GRB.Attr.X) for i in self.v},
            wvar={i: self.w[i].getAttr(GRB.Attr.X) for i in self.v},
            yvar_np=np.array(yvar_vector),
            wvar_np=np.array(wvar_vector)
        )

    def __del__(self):
        if self.model is not None:
            self.model.dispose()
        if self.env is not None:
            self.env.dispose()

    def solve(self) -> LinearModelResults:
        if self.integer:
            return self.__solve_integer()
        else:
            return self.__solve_continuous()
