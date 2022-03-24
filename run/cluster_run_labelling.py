from run_utils import get_script_base, cluster_mkdir_p
from run_labelling import algorithms
from hop.utils import tsiligirides_hop_dir
from glob import glob
import os

if __name__ == '__main__':
    launchers_dir = cluster_mkdir_p('launchers')
    output_dir = cluster_mkdir_p('output')

    for instance in glob(os.path.join(tsiligirides_hop_dir(), '*.json')):
        instance = os.path.realpath(instance)
        base = os.path.splitext(os.path.basename(instance))[0]

        for algorithm in algorithms:
            script_filename = os.path.join(launchers_dir, f"launcher_labelling_{algorithm}_{base}.sh")

            with open(script_filename, 'w') as f:
                f.write(get_script_base(output_dir=output_dir, script='run_labelling.py', cmdline_args=f"{algorithm} {instance}", instance_base=base, memcpu='8GB', timeout='01:30:00') + '\n')
