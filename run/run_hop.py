import argparse
import os
import numpy as np
from argparse import Namespace
from time import strftime
from glob import glob
from random import sample
from typing import Optional, Dict
from hop.instance import Instance
from hop.instance.tsiligirides import TsiligiridesInstance
from hop.results import Results
from hop.utils import run_results_dir, tsiligirides_hop_dir
from hop.instance.reduction import InstanceReductor
from hop.algorithms.constructive_heuristic import ConstructiveHeuristic
from hop.algorithms.ans import AdaptiveNeighbourhoodSearch, ANSParams
from hop.algorithms.ans_param_tuning import tune_params
from hop.algorithms.frank_wolfe import FrankWolfe
from hop.algorithms.labelling import strong_labelling, relaxed_labelling, relaxedtc_labelling
from hop.models.linear import LinearModel, LinearModelObjectiveFunction
from hop.models.nonlinear import NonLinearModel, NonLinearModelObjectiveFunction


def run_ans_tune(n_threads: int, n_instances: int = 10) -> None:
    instances_folder = tsiligirides_hop_dir()
    all_instances = glob(os.path.join(instances_folder, 'hop_*.json'))
    tuning_instances = sample(all_instances, n_instances)
    instances = [TsiligiridesInstance.load(i) for i in tuning_instances]
    tune_params(instances=instances, n_threads=n_threads)


def run_ans(instance: Instance, args: Optional[Namespace] = None, params: Optional[ANSParams] = None) -> Results:
    if params is not None:
        ans_params = params
    else:
        ans_params = ANSParams()

    if args is not None:
        if (x := args.mlt_improve_best) is not None:
            ans_params.mlt_improve_best = x
        if (x := args.mlt_improve_current) is not None:
            ans_params.mlt_improve_current = x
        if (x := args.mlt_accepted) is not None:
            ans_params.mlt_accepted = x
        if (x := args.rrt_start) is not None:
            ans_params.rrt_start_threshold = x
        if (x := args.rrt_end) is not None:
            ans_params.rrt_end_threshold = x
        if (x := args.reset_current) is not None:
            ans_params.reset_current = x
        if (x := args.local_search) is not None:
            ans_params.local_search = x
        if (x := args.ans_iterations) is not None:
            ans_params.n_iterations = x

    cons_heur = ConstructiveHeuristic(instance=instance)
    initial_sol = cons_heur.solve().tour_object
    solver = AdaptiveNeighbourhoodSearch(initial=initial_sol, params=ans_params)

    return solver.solve()


def run_frank_wolfe(instance: Instance, **kwargs) -> Results:
    y_np = np.array([1.0] * 2 + [0] * (instance.n_vertices - 2))
    w_np = np.array([instance.t[0][1] + instance.t[1][0], instance.t[1][0]] + [0] * (instance.n_vertices - 2))

    solver = FrankWolfe(instance=instance, initial_y=y_np, initial_w=w_np, **kwargs)
    return solver.solve()


def run_labelling_exact(instance: Instance, time_limit: float) -> Results:
    return strong_labelling(instance, time_limit=time_limit)


def run_labelling_ssr(instance: Instance, time_limit: float) -> Results:
    return relaxed_labelling(instance, time_limit=time_limit)


def run_labelling_ssr2ce(instance: Instance, time_limit: float) -> Results:
    return relaxedtc_labelling(instance, time_limit=time_limit)


def run_linear_ub(instance: Instance, **kwargs) -> Results:
    solver = LinearModel(
        instance=instance,
        obj_type=LinearModelObjectiveFunction.LINEAR_APPROX_LOOSE,
        **kwargs)
    return solver.solve()


def run_pw_linear_ub(instance: Instance, **kwargs) -> Results:
    solver = LinearModel(
        instance=instance,
        obj_type=LinearModelObjectiveFunction.LINEAR_APPROX_TIGHT,
        **kwargs)
    return solver.solve()


def run_nonlinear_baron(instance: Instance, concave: bool, continuous: bool, **kwargs) -> Results:
    obj_type = NonLinearModelObjectiveFunction.CONCAVE if concave else NonLinearModelObjectiveFunction.ORIGINAL
    solver = NonLinearModel(
        instance=instance,
        obj_type=obj_type,
        solve_continuous=continuous,
        **kwargs)
    return solver.solve()


def results_filename(algorithm: str, instance: Instance) -> str:
    res_dir = run_results_dir()
    fname = strftime('%Y-%m-%d-%H-%M-%S')
    fname += f"-{algorithm}"
    fname += f"-{instance.instance_info_for_filenames()}"
    fname += '.txt'

    return os.path.join(res_dir, fname)


def write_results(results: Results, instance: Instance) -> None:
    results.save_csv(filename=results_filename(algorithm=results.algorithm, instance=instance))


def read_constraints_args(args: Namespace) -> Dict[str, bool]:
    constraints_args = {}

    if args.lift_mtz:
        constraints_args['lift_mtz'] = True

    if args.add_vi:
        constraints_args['add_vi'] = True
        return constraints_args

    if args.add_vi1:
        constraints_args['add_vi1'] = True
    if args.add_vi2:
        constraints_args['add_vi2'] = True
    if args.add_vi3:
        constraints_args['add_vi3'] = True
    if args.add_vi4:
        constraints_args['add_vi4'] = True

    return constraints_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exact an heuristic methods for the Hazardous Orienteering Problem')

    parser.add_argument('--action', type=str, help='action to perform', choices=['ans', 'ans-tune', 'frank-wolfe', 'labelling-exact', 'labelling-ssr', 'labelling-ssr2ce', 'linearisation', 'pw-linearisation', 'baron'], action='store', required=True)
    parser.add_argument('--threads', type=int, help='number of threads to use for parallel runs (used during ANS parameter tuning and by Gurobi)', default=4)
    parser.add_argument('--instance', type=str, help='instance to solve (compulsory for all actions except ans-tune)', action='store')
    parser.add_argument('--reduce', help='whether to reduce the instance', action='store_true')
    parser.add_argument('--time-limit', type=float, help='time limit in seconds (supported by actions: labelling-*, linearisation, pw-linearisation, baron)', action='store', default=3600)
    parser.add_argument('--add-vi', help='wether to add all valid inequalities (supported by actions: frank-wolfe, linearisation, pw-linearisation, baron', action='store_true')
    parser.add_argument('--add-vi1', help='wether to add valid inequality 1 (supported by actions: frank-wolfe, linearisation, pw-linearisation, baron', action='store_true')
    parser.add_argument('--add-vi2', help='wether to add valid inequality 2 (supported by actions: frank-wolfe, linearisation, pw-linearisation, baron', action='store_true')
    parser.add_argument('--add-vi3', help='wether to add valid inequality 3 (supported by actions: frank-wolfe, linearisation, pw-linearisation, baron', action='store_true')
    parser.add_argument('--add-vi4', help='wether to add valid inequality 4 (supported by actions: frank-wolfe, linearisation, pw-linearisation', action='store_true')
    parser.add_argument('--lift-mtz', help='wether to lift MTZ constraints (supported by actions: frank-wolfe, linearisation, pw-linearisation, baron', action='store_true')

    ans_args = parser.add_argument_group('ANS arguments')
    ans_args.add_argument('--mlt-improve-best', type=float, help='ans improve on best multiplier (only for action ans)')
    ans_args.add_argument('--mlt-improve-current', type=float, help='ans improve on current multiplier (only for action ans)')
    ans_args.add_argument('--mlt-accepted', type=float, help='ans accepted multiplier (only for action ans)')
    ans_args.add_argument('--rrt-start', type=float, help='record-to-record travel start threshold (only for action ans)')
    ans_args.add_argument('--rrt-end', type=float, help='record-to-record travel end threshold (only for action ans)')
    ans_args.add_argument('--reset-current', type=int, help='reset current solution every this many iterations (only for action ans)')
    ans_args.add_argument('--local-search', type=int, help='perform local search every this many iterations (only for action ans)')
    ans_args.add_argument('--ans-iterations', type=int, help='number of ANS iterations to perform (only for action ans)')

    bar_args = parser.add_argument_group('Baron arguments')
    bar_args.add_argument('--concave', help='use the concavisation of the objective function (only for action baron)', action='store_true')
    bar_args.add_argument('--continuous', help='solve the continuous relaxation of the model via baron', action='store_true')

    args = parser.parse_args()

    if args.action == 'ans-tune':
        run_ans_tune(n_threads=args.threads)
        exit(os.EX_OK)

    filename = args.instance
    original_instance = TsiligiridesInstance.load(filename=filename)
    reductor = None

    if args.reduce:
        reductor = InstanceReductor(instance=original_instance)
        instance = reductor.new_instance
    else:
        instance = original_instance

    constraints_args = read_constraints_args(args)

    res = None

    if args.action == 'ans':
        res = run_ans(instance, args=args)
    elif args.action == 'frank-wolfe':
        res = run_frank_wolfe(instance, **constraints_args)
    elif args.action == 'labelling-exact':
        res = run_labelling_exact(instance, time_limit=args.time_limit)
    elif args.action == 'labelling-ssr':
        res = run_labelling_ssr(instance, time_limit=args.time_limit)
    elif args.action == 'labelling-ssr2ce':
        res = run_labelling_ssr2ce(instance, time_limit=args.time_limit)
    elif args.action == 'linearisation':
        res = run_linear_ub(instance, time_limit=args.time_limit, n_threads=args.threads, **constraints_args)
    elif args.action == 'pw-linearisation':
        res = run_pw_linear_ub(instance, time_limit=args.time_limit, n_threads=args.threads, **constraints_args)
    elif args.action == 'baron':
        res = run_nonlinear_baron(
            instance, time_limit=args.time_limit, concave=args.concave,
            continuous=args.continuous, **constraints_args)

    if args.reduce:
        assert reductor is not None
        assert res is not None
        res = reductor.results_new_to_old(results=res)

    write_results(results=res, instance=instance)
    exit(os.EX_OK)
