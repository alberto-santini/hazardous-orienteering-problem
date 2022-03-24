from hop.algorithms.ans import ANSParams, ANSResults, AdaptiveNeighbourhoodSearch
from hop.algorithms.constructive_heuristic import ConstructiveHeuristic
from hop.models.linear import LinearModel, LinearModelObjectiveFunction
from hop.utils import ans_param_tuning_results_dir
from hop.tour import Tour
from hop.instance import Instance
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.optimize import fmin
from colorama import Fore
from typing import List
from time import strftime
from glob import glob
import numpy as np
import pandas as pd
import traceback
import os


def __get_bounds(instances: List[Instance], n_threads: int) -> List[float]:
    def solve_cont(instance: Instance) -> float:
        return LinearModel(instance, obj_type=LinearModelObjectiveFunction.LINEAR_APPROX_TIGHT).solve().obj

    with Pool(processes=n_threads) as p:
        return p.map(solve_cont, instances)


def __get_initial_sols(instances: List[Instance], n_threads: int) -> List[Tour]:
    def get_init(instance: Instance) -> Tour:
        return ConstructiveHeuristic(instance=instance).solve().tour_object

    with Pool(processes=n_threads) as p:
        return p.map(get_init, instances)


def __solve_instance_reruns(initial: Tour, params: ANSParams, n_threads: int) -> float:
    def solve_instance(initial: Tour, params: ANSParams) -> float:
        instance_name = initial.instance.instance_info_for_filenames()

        # Check if results already exist:
        res_file_names = os.path.join(ans_param_tuning_results_dir(),
                                      f"partuning-{instance_name}-*-{params.param_info_for_filenames()}.txt")
        res_files = list(glob(res_file_names))

        if len(res_files) > 0:
            res_file = res_files[0]
            res_df = pd.read_csv(res_file)
            return res_df.obj[0]

        # If results do not already exist, solve ANS:
        ans = AdaptiveNeighbourhoodSearch(initial=initial, params=params)

        # If something goes wrong, log it and use 0 as the result of the algorithm
        try:
            res = ans.solve()
        except AssertionError as err:
            print(f"AssertionError triggered! {err=}")
            print('Priting the entire stack trace...')
            traceback.print_exc()
            return 0
        except Exception as err:
            print(f"Something else went wrong! {err=}")
            print('Printing the entire stack trace...')
            traceback.print_exc()
            return 0

        # Save results
        fname = strftime('%Y-%m-%d-%H-%M-%S')
        fname = os.path.join(ans_param_tuning_results_dir(),
                             f"partuning-{instance_name}-{fname}-{params.param_info_for_filenames()}.txt")
        res.save_csv(fname)

        return res.obj

    with Pool(processes=n_threads) as p:
        # Use the same initial solution and parameters for each rerun
        res = p.map(solve_instance, [initial] * n_threads, [params] * n_threads)

    return np.average(res)


def __solve_all_instances_reruns(initials: List[Tour], bounds: List[float], params: ANSParams, n_threads: int) -> float:
    gaps = list()

    for initial, bound in zip(initials, bounds):
        print(f"{Fore.CYAN}Instance: {initial.instance.instance_file}, bound: {bound:.3f}{Fore.RESET}")
        obj = __solve_instance_reruns(initial=initial, params=params, n_threads=n_threads)

        if obj == 0:
            gaps.append(1)
        else:
            gaps.append((obj - bound)  / obj)

    return np.average(gaps)


def __evaluate_params(params: np.ndarray, initials: List[Tour], bounds: List[float], n_threads: int) -> float:
    p = ANSParams.from_ndarray(params)

    print(f"{Fore.GREEN}Evaluating params: {p}{Fore.RESET}")

    return -1 * __solve_all_instances_reruns(initials=initials, bounds=bounds, params=p, n_threads=n_threads)


def tune_params(instances: List[Instance], n_threads: int = 4):
    bounds = __get_bounds(instances=instances, n_threads=n_threads)
    initials = __get_initial_sols(instances=instances, n_threads=n_threads)
    x0 = ANSParams().as_ndarray()
    max_function_evals = 50

    def evaluate_params(params: np.ndarray) -> float:
        return __evaluate_params(params=params, initials=initials, bounds=bounds, n_threads=n_threads)

    best_params, best_obj, n_iters, n_func_eval, _ = fmin(
        func=evaluate_params,
        x0=x0,
        maxfun=max_function_evals,
        full_output=True,
        disp=True
    )

    fname = os.path.join(ans_param_tuning_results_dir(), 'tuning-summary.txt')

    with open(fname, 'w') as f:
        f.write('Best parameters:\n')
        f.write(f"{best_params}\n")
        f.write('\nBest average objective:\n')
        f.write(f"{best_obj}\n")
        f.write(f"\nNelder-Mead iterations: {n_iters}\n")
        f.write(f"\nAlgorithm runs: {n_func_eval * n_threads}")

    return best_params
