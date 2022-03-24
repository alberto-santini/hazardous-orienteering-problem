import os
from typing import List, Optional
from glob import glob

### PATHS TO EXTERNAL PROGRAMMES AND LIBRARIES ###


TBKP_PATHS = [
    '/home/alberto/local/bin/tbkp',
    '/Users/alberto/local/bin/tbkp',
    '/homes/users/asantini/local/bin/tbkp',
    '/homes/users/asantini/local/src/tbkp/build/tbkp',
    '/homes/users/asantini/scratch/src/tbkp/build/tbkp'
]

CPLEX_SO_PATHS = [
    '/opt/cplex/cplex/bin/x86-64_linux/libcplex2010.so',
    '/Applications/CPLEX_Studio201/cplex/bin/x86-64_osx/libcplex2010.dylib',
    '/homes/users/asantini/local/cplex/cplex/bin/x86-64_linux/libcplex2010.so'
]


CPLEX_LIBDIR_PATHS = [
    '/opt/cplex/cplex/lib/x86-64_linux/static_pic',
    '/Applications/CPLEX_Studio201/cplex/lib/x86-64_osx/static_pic',
    '/homes/users/asantini/local/cplex/cplex/lib/x86-64_linux/static_pic'
]


GUROBI_LIBDIR_PATHS = [
    '/usr/lib',
    '/homes/aplic/noarch/software/Gurobi/9.0.0-lic-GCC-9.3.0/lib'
]


BARON_PATHS = [
    '/home/alberto/local/bin/baron',
    '/Users/alberto/local/bin/baron',
    '/homes/users/asantini/local/bin/baron'
]


def get_pythonpath() -> str:
    return os.path.join(os.path.dirname(__file__), '..')


def get_tbk_path() -> str:
    return __get_path(env_name='TBKP_PATH', default_paths=TBKP_PATHS)


def get_cplex_so_path() -> str:
    return __get_path(env_name='CPLEX_SO_PATH', default_paths=CPLEX_SO_PATHS)


def get_solvers_libdir_paths() -> str:
    gpath = __get_path(env_name='CPLEX_LIBDIR_PATH', default_paths=CPLEX_LIBDIR_PATHS)
    cpath = __get_path(env_name='GUROBI_SO_PATH', default_paths=GUROBI_LIBDIR_PATHS, if_contains='libgurobi*.so')

    return f"{gpath}:{cpath}"


def get_baron_path() -> str:
    return __get_path(env_name='BARON_PATH', default_paths=BARON_PATHS)


def __get_path(env_name: str, default_paths: List[str], if_contains: Optional[str] = None) -> str:
    def right_path(p: str):
        if if_contains is None:
            return True
        else:
            return len(glob(os.path.join(p, if_contains))) > 0

    if env_name in os.environ and right_path(os.environ[env_name]):
        return os.environ[env_name]
    
    for path in default_paths:
        if os.path.exists(path) and right_path(path):
            return path

    raise RuntimeError('No valid path found')


### PATHS INTERNAL TO THIS LIBRARY ###


def _mkdir_p_and_return(dir: str) -> str:
    os.makedirs(dir, exist_ok=True)
    return dir


def _relative_dir(*args) -> str:
    return os.path.join(os.path.dirname(__file__), '..', *args)


def tsiligirides_op_dir(num: int) -> str:
    return _relative_dir('data', 'op-tsiligirides', f"set-{num}")


def tsiligirides_hop_dir() -> str:
    dir = _relative_dir('data', 'hop-tsiligirides')
    return _mkdir_p_and_return(dir)


def ans_param_tuning_results_dir() -> str:
    dir = _relative_dir('results', 'ans-param-tuning')
    return _mkdir_p_and_return(dir)


def run_results_dir() -> str:
    dir = _relative_dir('results', 'run')
    return _mkdir_p_and_return(dir)


### OUTPUT SUPPRESSION ###


class suppress_output:
    def __enter__(self):
        self.sout = os.dup(1)
        self.serr = os.dup(2)
        self.null = open(os.devnull, 'r+')
        
        os.dup2(self.null.fileno(), 1)
        os.dup2(self.null.fileno(), 2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.sout, 1)
        os.dup2(self.serr, 2)
        self.null.close()
        os.close(self.serr)
        os.close(self.sout)
