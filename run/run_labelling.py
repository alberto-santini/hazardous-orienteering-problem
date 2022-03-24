from glob import glob
from hop.utils import tsiligirides_hop_dir
from hop.instance.tsiligirides import TsiligiridesInstance
from run_hop import run_labelling_exact, run_labelling_ssr, run_labelling_ssr2ce, write_results
from run_utils import already_solved
from time import time, strftime
from typing import Callable
import sys
import os


algorithms = ['exact', 'ssr', 'ssr2ce']


results_algorithm_names = {
    'exact': 'labelling_StrongLabel',
    'ssr': 'labelling_RelaxedLabel',
    'ssr2ce': 'labelling_RelaxedTCLabel'
}


def get_solver(algorithm: str) -> Callable:
    if algorithm not in algorithms:
        raise KeyError(f"{algorithm} is not a valid algorithm. Valid algorithms: {algorithms}.")

    if algorithm == 'exact':
        return run_labelling_exact
    elif algorithm == 'ssr':
        return run_labelling_ssr
    elif algorithm == 'ssr2ce':
        return run_labelling_ssr2ce
    else:
        raise KeyError(f"Unhandled case: {algorithm}")

if __name__ == '__main__':
    instances = list()

    if len(sys.argv) not in [2, 3]:
        print('Syntax: run_labelling.py [algorithm] [optional:instance]')
        exit(os.EX_USAGE)
    elif len(sys.argv) == 2:
        instances = list(glob(os.path.join(tsiligirides_hop_dir(), '*.json')))
    else:
        instances = [sys.argv[2]]

    solver = get_solver(sys.argv[1])
    start_time = time()

    for instance_num, instance_file in enumerate(instances):
        elapsed_time = time() - start_time
        print(f"Solving instance {instance_num + 1}/{len(instances)} ({os.path.basename(instance_file)}). Elapsed time: {elapsed_time:.2f}. ", end='')
        print('Current time: ' + strftime('%Y-%m-%d-%H-%M-%S'))

        instance = TsiligiridesInstance.load(filename=instance_file)

        if already_solved(instance=instance, algorithm=results_algorithm_names[sys.argv[1]]):
            print('\tResults chached: skipping.')
            exit(os.EX_OK)

        results = solver(instance=instance, time_limit=3600)
        write_results(results=results, instance=instance)
