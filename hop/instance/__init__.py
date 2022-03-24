from __future__ import annotations
import numpy as np
import json
from typing import List, Tuple, Dict
from colorama import Style, Fore
from os.path import splitext, basename, realpath


__all__ = ['generation', 'reduction', 'tsiligirides', 'Instance']


def _eucl(x1: float, y1: float, x2: float, y2: float):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class Instance:
    xs: List[float]      # X coordinates
    ys: List[float]      # Y coordinates
    l: List[float]       # Lambdas of the exp distributions
    p: List[float]       # Profits
    t: List[List[float]]  # Travel time matrix
    time_bound: float     # Max travel time of the vehicle
    n_vertices: int       # Num of vertices including the depot
    instance_file: str    # Filename of the instance file

    def __init__(self, xs: List[float], ys: List[float], p: List[float], l: List[float], time_bound: float, instance_file: str = ''):
        assert len(xs) == len(ys)
        assert len(xs) == len(p)
        assert len(xs) == len(l)

        self.n_vertices = len(xs)
        self.xs, self.ys = xs, ys
        self.p, self.l = p, l
        self.time_bound = time_bound
        self.instance_file = instance_file

        self.__compute_travel_time_matrix()
        self.__compute_T()
        self.__compute_tprime()
        self.__compute_pi()

    def __compute_travel_time_matrix(self):
        self.t = [[_eucl(x1, y1, x2, y2) for x2, y2 in zip(self.xs, self.ys)] for x1, y1 in zip(self.xs, self.ys)]

    def __compute_T(self):
        self.T = [self.time_bound - self.t[0][i] for i in range(self.n_vertices)]
        
    def __compute_pi(self):
        self.pi = [np.exp(- self.l[i] * self.T[i]) for i in range(self.n_vertices)]

    def __compute_tprime(self):
        self.tprime = [min([self.t[0][j] + self.t[j][i] for j in range(self.n_vertices)]) for i in range(self.n_vertices)]

    def t_dict(self) -> Dict[Tuple[int, int], float]:
        return {
            (i, j): self.t[i][j] for i in range(self.n_vertices) for j in range(self.n_vertices)
        }

    def tb_customers(self) -> List[int]:
        return [i for i, lbd in enumerate(self.l) if i > 0 and lbd > 0]

    def det_customers(self) -> List[int]:
        return [i for i, lbd in enumerate(self.l) if i > 0 and lbd == 0]

    def instance_info_for_filenames(self) -> str:
        return splitext(basename(self.instance_file))[0]
    
    def __str__(self) -> str:
        s = f"{Style.BRIGHT}{Fore.BLUE}{'#':>3} {'p':>6} {'λ':>6}{Style.RESET_ALL}\n"
        for i, (p, l) in enumerate(zip(self.p, self.l)):
            if i == 0:
                continue # Skip the depot
            s += f"{i:>3} {p:>6.1f} {l:6.2f}\n"

        return s

    def csv_header(self) -> str:
        return 'instance_file,instance_basename,n_vertices,n_customers,n_det_customers,n_tb_customers,'\
               'sum_all_profits,avg_profit,avg_det_profit,avg_tb_profit,'\
               'avg_tb_lambda,avg_arc_cost,time_bound'

    def to_csv(self) -> str:
        bname = splitext(basename(self.instance_file))[0]
        nc = self.n_vertices - 1
        detc = self.det_customers()
        tbc = self.tb_customers()
        sum_p = sum(self.p)
        avg_p =  sum_p/ nc
        sum_p_det = sum(self.p[i] for i in detc)
        sum_p_tb = sum(self.p[i] for i in tbc)
        avg_p_det = sum_p_det / len(detc)
        avg_p_tb = sum_p_tb / len(tbc)
        sum_l_tb = sum(self.l[i] for i in tbc)
        avg_l_tb = sum_l_tb / len(tbc)
        sum_t = sum(sum(x for x in row) for row in self.t)
        avg_t = sum_t / (self.n_vertices * (self.n_vertices - 1))

        return f"{realpath(self.instance_file)},{bname},{self.n_vertices},{nc},{len(detc)},{len(tbc)},"\
               f"{sum_p},{avg_p},{avg_p_det},{avg_p_tb},{avg_l_tb},{avg_t},{self.time_bound}"

    def save(self, filename: str):
        j = dict()
        j['n_vertices'] = self.n_vertices
        j['n_customers'] = self.n_vertices - 1
        j['distance_f'] = 'exact_euclidean'
        j['depot_id'] = 0
        j['time_bound'] = self.time_bound
        j['vertices'] = [
            {
                'x_coord': self.xs[i],
                'y_coord': self.ys[i],
                'lambda': self.l[i],
                'profit': self.p[i]
            } for i in range(self.n_vertices)
        ]

        with open(filename, 'w') as f:
            json.dump(j, f, indent=4)

        self.instance_file = filename

    @staticmethod
    def load(filename: str) -> Instance:
        with open(filename) as f:
            j = json.load(f)

        n_vertices = int(j['n_vertices'])
        time_bound = float(j['time_bound'])
        xs, ys = list(), list()
        l, p = list(), list()

        for vertex in j['vertices']:
            xs.append(float(vertex['x_coord']))
            ys.append(float(vertex['y_coord']))
            l.append(float(vertex['lambda']))
            p.append(float(vertex['profit']))

        assert len(xs) == n_vertices
        
        return Instance(xs=xs, ys=ys, l=l, p=p, time_bound=time_bound, instance_file=filename)

    @staticmethod
    def get_random(n: int = 10):
        xmin, xmax = 0, 10
        ymin, ymax = 0, 10
        pmin, pmax = 1, 5
        lmin, lmax = 0.05, 0.1
        prob_non_hazardous = 0.6

        n_vertices = n
        time_bound = (xmax - xmin) + (ymax - ymin)

        xs = np.random.uniform(xmin, xmax, n_vertices)
        ys = np.random.uniform(ymin, ymax, n_vertices)

        non_hazardous = np.random.choice(np.arange(n_vertices), replace=False, size=int(n_vertices * prob_non_hazardous))
        
        l = np.random.uniform(lmin, lmax, n_vertices)
        l[0] = 0.0
        l[non_hazardous] = 0.0

        minl = min([l for l in l if l > 0])
        maxl = max(l)
        
        p = np.random.uniform(pmin, pmax, n_vertices)
        p[0] = 0.0
        for idx, (p, l) in enumerate(zip(p, l)):
            if l > 0:
                p[idx] *= 4 + (l - minl) / (maxl - minl)
        
        return Instance(xs=list(xs), ys=list(ys), l=list(l), p=list(p), time_bound=time_bound)
