from glob import glob
from hop.utils import tsiligirides_hop_dir
from hop.instance.tsiligirides import TsiligiridesInstance
from run_hop import run_frank_wolfe, run_linear_ub, run_pw_linear_ub, write_results
from run_utils import existing_results_for
from time import time, strftime
import sys
import os


if __name__ == '__main__':
    if len(sys.argv) > 1:
        instances = [sys.argv[1]]
    else:
        instances = list(glob(os.path.join(tsiligirides_hop_dir(), '*.json')))

    params = dict(
        add_vi1=True, add_vi4=True, lift_mtz=True,
        time_limit=3600, n_threads=1
    )

    start_time = time()

    for instance_num, instance_file in enumerate(instances):
        elapsed_time = time() - start_time
        print(f"Solving instance {instance_num+1}/{len(instances)} ({os.path.basename(instance_file)}). Elapsed time: {elapsed_time:.2f}. ", end='')
        print('Current time: ' + strftime('%Y-%m-%d-%H-%M-%S'))

        instance = TsiligiridesInstance.load(filename=instance_file)

        results = existing_results_for(instance=instance, algorithm='frankwolfe_with_vi1_with_vi4_lifMTZ')
        if len(results) > 0:
            print('\tFrank-Wolfe results cached: skipping.')
        else:
            results = run_frank_wolfe(instance=instance, **params)
            write_results(results=results, instance=instance)

        results = existing_results_for(instance=instance, algorithm='integer_linear_model_LinearModelObjectiveFunction.LINEAR_APPROX_LOOSE_with_vi1_with_vi4_lifMTZ')
        if len(results) > 0:
            print('\tLinear (loose) results cached: skipping.')
        else:
            results = run_linear_ub(instance=instance, **params)
            write_results(results=results, instance=instance)

        results = existing_results_for(instance=instance, algorithm='integer_linear_model_LinearModelObjectiveFunction.LINEAR_APPROX_TIGHT_with_vi1_with_vi4_lifMTZ')
        if len(results) > 0:
            print('\tLinear (tight) results cached: skipping.')
        else:
            results = run_pw_linear_ub(instance=instance, **params)
            write_results(results=results, instance=instance)
