from __future__ import annotations
from hop.tour import Tour
from hop.instance import Instance
from hop.results import Results
from hop.models.nonlinear import NonLinearModel
from copy import deepcopy
from typing import Tuple, List
from time import time
from random import random
from colorama import Fore
from dataclasses import dataclass
from numpy.random import choice
from collections import defaultdict
from os.path import basename, splitext
import numpy as np


@dataclass
class ANSResults(Results):
    feasible: bool
    params: ANSParams
    status: ANSStatus

    def csv_header(self) -> str:
        return f"{super().csv_header()},feasible,{ANSParams.csv_header()},{ANSStatus.csv_header()}"

    def to_csv(self) -> str:
        return f"{super().to_csv()},{self.feasible},{self.params.to_csv()},{self.status.to_csv()}"


@dataclass
class ANSParams:
    mlt_improve_best: float = 1.01
    mlt_improve_current: float = 1.005
    mlt_accepted: float = 1.0

    rrt_start_threshold: float = 0.8
    rrt_end_threshold: float = 0.4

    reset_current: int = 2000
    local_search: int = 500

    n_iterations: int = 10000
    output_iterations: int = 1000

    def as_ndarray(self) -> np.ndarray:
        return np.asarray([
            self.mlt_improve_best, self.mlt_improve_current,
            self.mlt_accepted, self.rrt_start_threshold,
            self.rrt_end_threshold, self.reset_current,
            self.local_search
        ])

    @staticmethod
    def from_ndarray(ary: np.ndarray) -> ANSParams:
        return ANSParams(
            mlt_improve_best=ary[0],
            mlt_improve_current=ary[1],
            mlt_accepted=ary[2],
            rrt_start_threshold=ary[3],
            rrt_end_threshold=ary[4],
            reset_current=ary[5],
            local_search=ary[6]
        )

    @staticmethod
    def csv_header() -> str:
        fields = [k for k in ANSParams.__annotations__ if k != 'output_iterations']
        return ','.join(fields)

    def to_csv(self) -> str:
        return self.__join_params(delim=',')

    def param_info_for_filenames(self) -> str:
        return self.__join_params(delim='_')

    def __join_params(self, delim: str) -> str:
        row = list()

        for attr, t in self.__annotations__.items():
            if attr == 'output_iterations':
                continue

            val = getattr(self, attr)

            if t == 'int':
                row.append(f"{val}")
            elif t == 'float':
                row.append(f"{val:.2f}")

        return delim.join(row)


class ANSStatus:
    start_time: float
    end_time: float
    iter: int
    best_sol_history: List[dict]
    current_iteration: int
    reset_current_base_iteration: int

    def __init__(self, instance: Instance):
        self.start_time = time()
        self.scores = [1.0] * len(AdaptiveNeighbourhoodSearch.move_funcs)
        self.best_sol_history = list()
        self.current_iteration = 1
        self.reset_current_base_iteration = 1

    def elapsed_time(self):
        return time() - self.start_time

    def tot_elapsed_time(self):
        return self.end_time - self.start_time

    def add_best(self, best: Tour):
        self.best_sol_history.append(
            dict(fitness=best.fitness(), time=self.elapsed_time(), iter=self.current_iteration)
        )
        self.reset_current_base_iteration = self.current_iteration

    @staticmethod
    def csv_header() -> str:
        s = 'elapsed_time,'
        for m in AdaptiveNeighbourhoodSearch.move_names:
            m = m.lower().replace(' ', '_')
            s += f"{m}_score,"
        s += 'iter_last_update_best'
        return s

    def to_csv(self) -> str:
        s = f"{self.tot_elapsed_time():.2f},"
        s += ','.join([f"{x:.3f}" for x in self.scores])
        s += f",{self.best_sol_history[-1]['iter']}"
        return s

class AdaptiveNeighbourhoodSearch:
    move_names = [
        'Remove best',
        'Remove best or rnd',
        'Remove rnd',
        'Remove rnd 3',
        'Swap TB with DET',
        'Swap TB with TB',
        'Swap DET with DET',
        'Swap rnd with rnd',
        'Insert best at best',
        'Insert rnd at best',
        'Insert rnd at rnd'
    ]

    move_funcs = [
        lambda t: t.remove_best(),
        lambda t: t.remove_best_if_improving_else_random(),
        lambda t: t.remove_random(n=1),
        lambda t: t.remove_random(n=3),
        lambda t: t.swap_earlier_tb_with_later_det(),
        lambda t: t.swap_earlier_more_tb_with_later_less_tb(),
        lambda t: t.swap_det_with_det(),
        lambda t: t.swap_random_with_random(),
        lambda t: t.insert_best_customer_best_position(),
        lambda t: t.insert_random_customer_best_position(),
        lambda t: t.insert_random_customer_random_position()
    ]

    def __init__(self, initial: Tour, **kwargs) -> None:
        self.params = kwargs.get('params', ANSParams())
        self.instance = initial.instance
        self.status = None
        self.current = deepcopy(initial)
        self.best = deepcopy(initial)
        self.n_accepted_since_output = 0
        self.rrt_treshold_span = self.params.rrt_end_threshold - self.params.rrt_start_threshold
        self.local_search_cache = dict()

        self.record_solution_count = kwargs.get('record_solution_count', False)
        if self.record_solution_count:
            self.solcount = defaultdict(int)

    def __roulette_wheel_move_id(self) -> int:
        return choice(len(self.move_funcs), p=np.asarray(self.status.scores)/sum(self.status.scores))

    def __next_solution(self) -> Tuple[Tour, int]:
        move_id = self.__roulette_wheel_move_id()
        move = self.move_funcs[move_id]
        return move(self.current), move_id

    def __update_score(self, move_id: int, ibest: bool, icurr: bool, acc: bool) -> None:
        if ibest:
            self.status.scores[move_id] *= self.params.mlt_improve_best
        elif icurr:
            self.status.scores[move_id] *= self.params.mlt_improve_current
        elif acc:
            self.status.scores[move_id] *= self.params.mlt_accepted

        # Clamping:
        self.status.scores[move_id] = max(0.1, min(10.0, self.status.scores[move_id]))

    def __accept(self, solution: Tour) -> bool:
        if self.record_solution_count:
            self.solcount[solution] += 1

        progress = self.status.current_iteration / self.params.n_iterations
        current_threshold = self.params.rrt_end_threshold + self.rrt_treshold_span * (1 - progress)

        if self.best.fitness() != 0:
            gap = (self.best.fitness() - solution.fitness()) / self.best.fitness()
        else:
            gap = 1.0

        return gap < current_threshold

    def __reset_current(self) -> None:
        if random() > 0.5:
            print(Fore.RED, end='')
            self.current = deepcopy(self.best)
        else:
            print(Fore.GREEN, end='')
            self.current = Tour.get_random(instance=self.instance)

    def __local_search_current(self) -> None:
        if self.current in self.local_search_cache:
            return self.local_search_cache[self.current]

        sols = list()

        # If there is anything to insert:
        if self.current.n_customers() < self.instance.n_vertices - 1:
            ins_slv = NonLinearModel(instance=self.instance, current_tour_insert=self.current, time_limit=10)
            solution = ins_slv.solve()

            if solution.has_feasible_solution:
                sols.append(solution)

        # If there is anything to remove:
        if self.current.n_customers() > 0:
            rem_slv = NonLinearModel(instance=self.instance, current_tour_remove=self.current, time_limit=10)

            solution = rem_slv.solve()

            if solution.has_feasible_solution:
                sols.append(solution)

        if len(sols) > 0:
            best = max(sols, key=lambda r: r.obj).tour_object

            if best.hierarchical_fitness() > self.current.hierarchical_fitness():
                self.local_search_cache[deepcopy(self.current)] = deepcopy(best)
                self.current = deepcopy(best)

    def solve(self) -> ANSResults:
        self.status = ANSStatus(instance=self.instance)
        self.status.add_best(self.best)

        while self.status.current_iteration < self.params.n_iterations:
            new_solution, move_id = self.__next_solution()
            accepted, icurr, ibest = False, False, False

            if self.__accept(new_solution):
                accepted = True
                self.n_accepted_since_output += 1

                if new_solution.fitness() > self.current.fitness():
                    icurr = True

                    assert self.best.fitness() >= self.current.fitness()

                    if new_solution.fitness() > self.best.fitness():
                        ibest = True
                        self.best = deepcopy(new_solution)
                        self.status.add_best(best=self.best)

                self.current = deepcopy(new_solution)

            self.__update_score(move_id=move_id, ibest=ibest, icurr=icurr, acc=accepted)
            self.status.current_iteration += 1

            if self.status.current_iteration % self.params.local_search == 0:
                self.__local_search_current()

                if self.current.fitness() > self.best.fitness():
                    self.best = deepcopy(self.current)
                    self.status.add_best(best=self.best)

            if self.status.current_iteration >= self.status.reset_current_base_iteration + self.params.reset_current:
                self.__reset_current()
                self.status.reset_current_base_iteration = self.status.current_iteration

            if self.status.current_iteration % self.params.output_iterations == 0:
                print(f"Iter: {self.status.current_iteration:6}{Fore.RESET} ", end='')
                print(f"- Acc%: {100 * self.n_accepted_since_output / self.params.output_iterations:6.2f} ", end='')
                print(f"- Best: {self.best.fitness():8.2f} ", end='')
                print(f"- Current: {self.current.fitness():8.2f} {self.current.tour} ", end='')
                print(f"- Scores: ", end='')
                for val in self.status.scores:
                    print(f"{val:6.2f} ", end='')
                print()
                self.n_accepted_since_output = 0

        self.status.end_time = time()
        print(f"Best solution value = {self.best.objval():.3f}")

        if self.record_solution_count:
            self.__solcount_write()

        return ANSResults(
            instance=self.instance,
            algorithm=f"ans_{self.params.param_info_for_filenames()}",
            tour_object=self.best,
            obj=self.best.objval(),
            feasible=(self.best.duration() <= self.best.instance.time_bound),
            time_s=self.status.tot_elapsed_time(),
            params=self.params,
            status=self.status
        )

    def __solcount_write(self):
        assert self.record_solution_count

        bn = splitext(basename(self.instance.instance_file))[0]

        with open(f"solcount_{bn}.csv", 'w') as f:
            f.write('solution,count,obj,fitness\n')

            for solution, count in self.solcount.items():
                f.write(f"\"{solution.tour}\",{count},{solution.objval():.3f},{solution.fitness():.3f}\n")
