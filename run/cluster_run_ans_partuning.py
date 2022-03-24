from run_utils import get_script_base, cluster_mkdir_p
import os

if __name__ == '__main__':
    launchers_dir = cluster_mkdir_p('launchers')
    output_dir = cluster_mkdir_p('output')
    script_filename = os.path.join(launchers_dir, f"launcher_ans_partuning.sh")

    with open(script_filename, 'w') as f:
        f.write(get_script_base(output_dir=output_dir, script='run_hop.py', cmdline_args='--action ans-tune --threads 4', cpus=4, instance_base='tuning', timeout='7-00:00:00') + '\n')
