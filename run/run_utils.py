from run_hop import results_filename
from hop.instance.tsiligirides import TsiligiridesInstance
from glob import glob
from textwrap import dedent
from os.path import basename, splitext
from hop.utils import get_solvers_libdir_paths
import os


def already_solved(instance: TsiligiridesInstance, algorithm: str) -> bool:
    res_file = results_filename(algorithm=algorithm, instance=instance)
    res_dirname = os.path.dirname(res_file)
    res_basename = os.path.basename(res_file)

    bn = '-'.join(res_basename.split('-')[6:])
    gl = os.path.join(os.path.realpath(res_dirname), f"*{bn}")

    return len(glob(gl)) > 0


def cluster_get_pythonpath():
    return os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))


def cluster_get_runfile_path(script: str):
    return os.path.realpath(os.path.join(cluster_get_pythonpath(), 'run', script))


def cluster_mkdir_p(dir: str) -> str:
    fulldir = os.path.realpath(os.path.join(cluster_get_pythonpath(), 'cluster', dir))
    os.makedirs(fulldir, exist_ok=True)
    return fulldir


def get_script_base(output_dir: str, script: str, cmdline_args: str, instance_base: str, **kwargs) -> str:
    script = f"""
        #!/bin/bash
        #SBATCH --partition=normal
        #SBATCH --time={kwargs.get('timeout', '01:00:00')}
        #SBATCH --cpus-per-task={kwargs.get('cpus', 1)}
        #SBATCH --mem-per-cpu={kwargs.get('memcpu', '1GB')}
        #SBATCH -o {os.path.join(output_dir, f"out_{instance_base}.txt")}
        #SBATCH -e {os.path.join(output_dir, f"err_{instance_base}.txt")}

        module load GCC/9.3.0
        module load Boost/1.72.0-gompi-2020a
        module load Gurobi/9.0.0-lic-GCC-9.3.0

        source $HOME/scratch/anaconda3/etc/profile.d/conda.sh
        source activate optimisation

        LD_LIBRARY_PATH="{get_solvers_libdir_paths()}" PYTHONPATH="{cluster_get_pythonpath()}" python3 {cluster_get_runfile_path(script=script)} {cmdline_args}
    """
    script = '\n'.join(script.split('\n', 1)[1:])

    return dedent(script)
