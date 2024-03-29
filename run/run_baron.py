from glob import glob
from hop.utils import tsiligirides_hop_dir
from hop.instance.tsiligirides import TsiligiridesInstance
from run_hop import run_nonlinear_baron, write_results
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

    params = dict(
        concave=False, continuous=False,
        add_vi1=True, lift_mtz=True,
        time_limit=3600, n_threads=1
    )

    for instance_num, instance_file in enumerate(instances):
        elapsed_time = time() - start_time
        print(f"Solving instance {instance_num+1}/{len(instances)} ({os.path.basename(instance_file)}). Elapsed time: {elapsed_time:.2f}. ", end='')
        print('Current time: ' + strftime('%Y-%m-%d-%H-%M-%S'))

        instance = TsiligiridesInstance.load(filename=instance_file)

        if already_solved(instance=instance, algorithm='nonlinear_model_with_vi1_liftMTZ'):
            print('\tBaron results cached: skipping.')
        else:
            results = run_nonlinear_baron(instance=instance, **params)
            write_results(results=results, instance=instance)
