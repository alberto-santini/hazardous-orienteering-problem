from glob import glob
from hop.algorithms.ans import ANSParams
from hop.utils import tsiligirides_hop_dir
from hop.instance.tsiligirides import TsiligiridesInstance
from run_hop import run_ans, write_results
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
        params = ANSParams()

        if already_solved(instance=instance, algorithm=f"ans_{params.param_info_for_filenames()}"):
            print('\tANS results cached: skipping.')
        else:
            results = run_ans(instance=instance, params=params)
            write_results(results=results, instance=instance)
