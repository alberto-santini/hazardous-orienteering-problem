# Solves all instances with the nonlinear model (baron) with a short-ish time
# limit of 5 minutes. In this way, we can get an idea of how the generated
# instances "behave": are TB customers ever selected? How many? How long are the
# tours? Etc...

from glob import glob
from hop.utils import tsiligirides_hop_dir
from hop.instance.tsiligirides import TsiligiridesInstance
from run_hop import run_nonlinear_baron, results_filename, write_results
from time import time
import os


def already_solved(instance: TsiligiridesInstance) -> bool:
    res_file = results_filename(algorithm='nonlinear_model', instance=instance)
    res_dirname = os.path.dirname(res_file)
    res_basename = os.path.basename(res_file)

    bn = '-'.join(res_basename.split('-')[6:])
    gl = os.path.join(os.path.realpath(res_dirname), f"*{bn}")

    return len(glob(gl)) > 0


if __name__ == '__main__':
    instances = list(glob(os.path.join(tsiligirides_hop_dir(), '*.json')))
    start_time = time()

    for instance_num, instance_file in enumerate(instances):
        elapsed_time = time() - start_time
        print(f"Solving instance {instance_num+1}/{len(instances)} ({os.path.basename(instance_file)}). Elapsed time: {elapsed_time:.2f}.")

        instance = TsiligiridesInstance.load(filename=instance_file)

        if already_solved(instance):
            print('\tResults cached: skipping.')
            continue

        results = run_nonlinear_baron(instance=instance, time_limit=5*60)
        write_results(results=results, instance=instance)
