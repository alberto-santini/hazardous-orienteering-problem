from run_utils import get_script_base, cluster_mkdir_p
from hop.utils import tsiligirides_hop_dir
from glob import glob
import os

if __name__ == '__main__':
    launchers_dir = cluster_mkdir_p('launchers')
    output_dir = cluster_mkdir_p('output')

    for instance in glob(os.path.join(tsiligirides_hop_dir(), '*.json')):
        instance = os.path.realpath(instance)
        base = os.path.splitext(os.path.basename(instance))[0]
        script_filename = os.path.join(launchers_dir, f"launcher_ans_{base}.sh")

        with open(script_filename, 'w') as f:
            f.write(get_script_base(output_dir=output_dir, script='run_ans.py', cmdline_args=instance, instance_base=base, timeout='01:00:00') + '\n')
