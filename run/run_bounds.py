from glob import glob
from hop.utils import tsiligirides_hop_dir
from hop.instance.tsiligirides import TsiligiridesInstance
from run_hop import run_frank_wolfe, run_linear_ub, run_pw_linear_ub, write_results
from run_utils import already_solved
from time import time, strftime
import sys
import os


if __name__ == '__main__':
    if len(sys.argv) > 1:
        instances = [sys.argv[1]]
    else:
        instances = list(glob(os.path.join(tsiligirides_hop_dir(), '*.json')))

    start_time = time()

    for instance_num, instance_file in enumerate(instances):
        elapsed_time = time() - start_time
        print(f"Solving instance {instance_num+1}/{len(instances)} ({os.path.basename(instance_file)}). Elapsed time: {elapsed_time:.2f}. ", end='')
        print('Current time: ' + strftime('%Y-%m-%d-%H-%M-%S'))

        instance = TsiligiridesInstance.load(filename=instance_file)

        if already_solved(instance=instance, algorithm='frankwolfe'):
            print('\tFrank-Wolfe results cached: skipping.')
        else:
            results = run_frank_wolfe(instance=instance)
            write_results(results=results, instance=instance)

        if already_solved(instance=instance, algorithm='integer_linear_model_LinearModelObjectiveFunction.LINEAR_APPROX_LOOSE'):
            print('\tLinear (loose) results cached: skipping.')
        else:
            results = run_linear_ub(instance=instance, time_limit=60)
            write_results(results=results, instance=instance)

        if already_solved(instance=instance, algorithm='integer_linear_model_LinearModelObjectiveFunction.LINEAR_APPROX_TIGHT'):
            print('\tLinear (tight) results cached: skipping.')
        else:
            results = run_pw_linear_ub(instance=instance, time_limit=60)
            write_results(results=results, instance=instance)
