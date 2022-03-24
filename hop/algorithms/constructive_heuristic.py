import os
import numpy as np
from tempfile import NamedTemporaryFile
from subprocess import run
from concorde.tsp import TSPSolver
from collections import deque
from typing import List
from time import time
from hop.tour import Tour
from hop.results import Results
from hop.instance import Instance
from hop.utils import suppress_output, get_tbk_path
from hop.models.nonlinear import NonLinearModel


class ConstructiveHeuristic:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self.customers = list()
        self.tour = Tour(instance=self.instance)

    def __write_tbkp_file(self, tmp_file: NamedTemporaryFile, time_bound_frac: float = 0.25) -> None:
        expected_travel_time = self.instance.time_bound * time_bound_frac / 2

        # Number of items ; capacity
        tmp_file.write(f"{self.instance.n_vertices - 1} {self.instance.n_vertices - 1}\n")

        for i in range(1, self.instance.n_vertices):
            # Weight ; profit ; probability
            tmp_file.write(f"1 {int(100 * self.instance.p[i])} {np.exp(- self.instance.l[i] * expected_travel_time):.6f}\n")

        tmp_file.flush()

    def __solve_tbkp(self) -> None:
        tbkp_timeout = 60
        inst_file = NamedTemporaryFile(prefix='tbkp-inst', mode='w', delete=False)
        sol_file = NamedTemporaryFile(prefix='tbkp-sol', mode='r', delete=False)

        self.__write_tbkp_file(tmp_file=inst_file)

        exe = get_tbk_path()
        cmd = f"{exe} -i {inst_file.name} -S {sol_file.name} -s bb -t {tbkp_timeout} -T 1 -R 10 -c 1 -p 1 -r 1 -d 1 -b 1 -f 1000 -n 0 -g 0 -q 1"
        run(cmd.split())

        self.customers = [int(x) + 1 for x in sol_file.readline().split()]

        inst_file.close()
        sol_file.close()
        os.unlink(inst_file.name)
        os.unlink(sol_file.name)

    def __solve_tsp(self, customers: list) -> List[int]:
        nodes = [0] + customers
        xs = [self.instance.xs[i] for i in nodes]
        ys = [self.instance.ys[i] for i in nodes]
        
        with suppress_output():
            solver = TSPSolver.from_data(xs=xs, ys=ys, norm='EUC_2D')
            tour = solver.solve()

        tour = deque([nodes[i] for i in tour.tour])
        tour.rotate(-tour.index(0))
        return list(tour) + [0]

    def __create_tour(self) -> None:
        deterministic_customers = [i for i in self.customers if self.instance.l[i] <= 0]
        tb_customers = [i for i in self.customers if self.instance.l[i] > 0]

        det_tsp = self.__solve_tsp(customers=deterministic_customers)
        tb_tsp = self.__solve_tsp(customers=tb_customers)

        det_cst, tb_cst = det_tsp[1:-1], tb_tsp[1:-1]
        possible_tours = [
            [0] + det_cst + tb_cst + [0],
            [0] + det_cst + list(reversed(tb_cst)) + [0],
            [0] + list(reversed(det_cst)) + tb_cst + [0],
            [0] + list(reversed(det_cst)) + list(reversed(tb_cst)) + [0]
        ]

        self.tour = max(
            [Tour(instance=self.instance, tour=tour) for tour in possible_tours],
            key=lambda t: t.objval()
        )

    def solve(self) -> Results:
        start = time()

        # Phase 1: choose customers
        self.__solve_tbkp()
        # Phase 2: create tour
        self.__create_tour()

        # If the tour is too long, make it feasible
        if self.tour.duration() > self.instance.time_bound:
            slv = NonLinearModel(instance=self.instance, current_tour_remove=self.tour, time_limit=10)
            res = slv.solve()
            self.tour = res.tour_object

        end = time()

        return Results(
            instance=self.instance,
            algorithm='constructive',
            tour_object=self.tour,
            obj=self.tour.objval(),
            time_s=(end - start)
        )
