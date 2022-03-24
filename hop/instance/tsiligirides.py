from __future__ import annotations
from hop.instance import Instance
from hop.utils import tsiligirides_op_dir
import numpy as np
import os


class TsiligiridesInstance(Instance):
    num: int
    tb: int
    alpha: float
    beta: float

    def __init__(self, num: int, time_bound: int, alpha: float, beta: float):
        assert num in [1, 2, 3]

        base_dir = tsiligirides_op_dir(num=num)
        if num == 3:
            # In set 3 the budget is padded with 0s to reach 3 digits
            self.instance_file = os.path.join(base_dir, f"tsiligirides_problem_{num}_budget_{time_bound:03}.txt")
        else:
            # In set 1 and 2 the budget is padded with 0s to reach 2 digits
            self.instance_file = os.path.join(base_dir, f"tsiligirides_problem_{num}_budget_{time_bound:02}.txt")

        if not os.path.exists(self.instance_file):
            raise ValueError(f"Cannot find instance {self.instance_file}")

        with open(self.instance_file, 'r') as f:
            budget, _ = f.readline().split()

            assert int(budget) == time_bound

            # Starting depot line:
            depot_x, depot_y, _ = f.readline().split()

            # End depot line: we skip it
            f.readline()

            xs = [float(depot_x)]
            ys = [float(depot_y)]
            ps = [0.0]

            while line := f.readline():
                x, y, p = line.split()
                xs.append(float(x))
                ys.append(float(y))
                ps.append(float(p))

            n_customers = len(xs) - 1
            n_tb = int(n_customers * alpha)
            indices_tb = np.random.choice(np.arange(n_customers), replace=False, size=n_tb)
            indices_tb = [i + 1 for i in indices_tb]
            ls = [0.0] * len(xs)

            for idx in indices_tb:
                ps[idx] *= beta
                ls[idx] = np.random.uniform(low=0.05, high=0.1)

            super().__init__(xs=xs, ys=ys, p=ps, l=ls, time_bound=float(time_bound))

            self.num = num
            self.tb = time_bound
            self.alpha = alpha
            self.beta = beta

    @staticmethod
    def load(filename: str) -> TsiligiridesInstance:
        bn = os.path.splitext(os.path.basename(filename))[0]
        fields = bn.split('-')
        num, tb, alpha, beta = fields[-4], fields[-3], fields[-2], fields[-1]

        instance = Instance.load(filename=filename)
        instance.__class__ = TsiligiridesInstance
        instance.num = num
        instance.tb = tb
        instance.alpha = alpha
        instance.beta = beta

        return instance

    def instance_info_for_filenames(self) -> str:
        return f"hop_tsiligirides-{self.num}-{self.tb}-{self.alpha}-{self.beta}"

    def csv_header(self) -> str:
        return f"{super().csv_header()},instance_type,tsiligirides_op_num,tsiligirides_op_tb,tsiligirides_hop_alpha,tsiligirides_hop_beta"

    def to_csv(self) -> str:
        return f"{super().to_csv()},tsiligirides_hop,{self.num},{self.tb},{self.alpha},{self.beta}"
