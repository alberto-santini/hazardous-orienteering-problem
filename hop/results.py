from dataclasses import dataclass
from typing import Optional
from time import strftime
from hop.instance import Instance
from hop.tour import Tour
import numpy as np

@dataclass
class Results:
    instance: Instance
    algorithm: str
    tour_object: Optional[Tour]
    obj: float
    time_s: float

    def tour(self):
        if self.tour_object is not None:
            return self.tour_object.tour
        else:
            return None

    def csv_header(self) -> str:
        header = 'run_date,algorithm,obj,time_s,'
        header += self.instance.csv_header()

        if self.tour_object is not None:
            header += ',tour,'
            header += 'tour_profit,tour_prob,tour_duration,'
            header += 'pct_time_bound_used,'
            header += 'n_custs_visited,n_tb_custs_visited,'
            header += 'pct_profit_collected,'
            header += 'pct_custs_visited,pct_custs_visited_which_are_tb,'
            header += 'avg_visited_det_profit,'
            header += 'avg_visited_tb_profit,'
            header += 'avg_visited_tb_lambda,'
            header += 'avg_travel_time_after_det_cust,'
            header += 'avg_travel_time_after_tb_cust'

        return header

    def to_csv(self) -> str:
        line = strftime('%Y-%m-%d-%H-%M-%S')
        line += f",{self.algorithm},{self.obj},{self.time_s},"
        line += self.instance.to_csv()

        if self.tour_object is not None:
            line += f",\"{','.join([str(x) for x in self.tour()])}\","
            line += self._csv_tour_stats()

        return line

    def save_csv(self, filename: str) -> None:
        with open(filename, 'w') as f:
            f.write(self.csv_header())
            f.write("\n")
            f.write(self.to_csv())
            f.write("\n")

    def _csv_tour_stats(self) -> str:
        def mean(l):
            if len(l) == 0:
                return 0
            else:
                return np.mean(l)
        
        T = self.tour_object
        inst = self.instance
        visited_det_custs = T.det_customers()
        visited_tb_custs = T.tb_customers()
        visited_det_idx, visited_tb_idx = T.det_tb_indices()

        profit = T.sum_profits
        prob = T.product_probs
        duration = T.duration()
        time_bound = inst.time_bound
        pct_time_bound_used = 100 * duration / time_bound
        n_visited = T.n_customers()
        n_visited_tb = len(visited_tb_custs)
        pct_profit = 100 * T.sum_profits / sum(inst.p)
        pct_visited = 100 * n_visited / (inst.n_vertices - 1)
        pct_visited_tb = 100 * n_visited_tb / n_visited
        avg_profit_visited_det = mean([inst.p[i] for i in visited_det_custs])
        avg_profit_visited_tb = mean([inst.p[i] for i in visited_tb_custs])
        avg_lambda_visited_tb = mean([inst.l[i] for i in visited_tb_custs])
        avg_tt_after_det_cust = mean([T.w[i] for i in visited_det_idx])
        avg_tt_after_tb_cust = mean([T.w[i] for i in visited_tb_idx])

        return f"{profit},{prob},{duration},{pct_time_bound_used},"\
            f"{n_visited},{n_visited_tb},{pct_profit},{pct_visited},{pct_visited_tb},"\
            f"{avg_profit_visited_det},{avg_profit_visited_tb},"\
            f"{avg_lambda_visited_tb},"\
            f"{avg_tt_after_det_cust},{avg_tt_after_tb_cust}"
