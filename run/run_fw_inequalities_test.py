from cProfile import run
from glob import glob
from hop.utils import tsiligirides_hop_dir
from hop.instance.tsiligirides import TsiligiridesInstance
from run_hop import run_frank_wolfe, write_results
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

        if already_solved(instance=instance, algorithm='frankwolfe_liftMTZ'):
            print('\tFrank-Wolfe with MTZ lifting results cached: skipping.')
        else:
            results = run_frank_wolfe(instance=instance, lift_mtz=True)
            write_results(results=results, instance=instance)

        if already_solved(instance=instance, algorithm='frankwolfe_with_vi1_liftMTZ'):
            print('\tFrank-Wolfe + Lifting + VI1 results cached: skipping.')
        else:
            results = run_frank_wolfe(instance=instance, lift_mtz=True, add_vi1=True)
            write_results(results=results, instance=instance)

        if already_solved(instance=instance, algorithm='frankwolfe_with_vi1_with_vi2_liftMTZ'):
            print('\tFrank-Wolfe + Lifting + VI1 + VI2 results cached: skipping.')
        else:
            results = run_frank_wolfe(instance=instance, lift_mtz=True, add_vi1=True, add_vi2=True)
            write_results(results=results, instance=instance)

        if already_solved(instance=instance, algorithm='frankwolfe_with_vi1_with_vi2_with_vi3_liftMTZ'):
            print('\tFrank-Wolfe + Lifting + VI1 + VI2 + VI3 results cached: skipping.')
        else:
            results = run_frank_wolfe(instance=instance, lift_mtz=True, add_vi1=True, add_vi2=True, add_vi3=True)
            write_results(results=results, instance=instance)

        if already_solved(instance=instance, algorithm='frankwolfe_with_vi1_with_vi2_with_vi3_with_vi4_liftMTZ'):
            print('\tFrank-Wolfe + Lifting + VI1 + VI2 + VI3 + VI4 results cached: skipping.')
        else:
            results = run_frank_wolfe(instance=instance, lift_mtz=True, add_vi1=True, add_vi2=True, add_vi3=True, add_vi4=True)
            write_results(results=results, instance=instance)