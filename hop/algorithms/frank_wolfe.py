from distutils.log import debug
from hop.instance import Instance
from hop.results import Results
from hop.models.linear import LinearModel, LinearModelObjectiveFunction
from numpy.typing import ArrayLike
from typing import Tuple
from time import time
from enum import Enum
from dataclasses import dataclass
import numpy as np

import logging
logging.basicConfig(level=logging.ERROR)


class FrankWolfeStepType(Enum):
    LINE_SEARCH = 1
    DECREASING = 2


@dataclass
class FrankWolfeResults(Results):
    lift_mtz: bool
    add_vi1: bool
    add_vi2: bool
    add_vi3: bool
    add_vi4: bool
    iterations_limit: int
    iterations_used: int

    def csv_header(self) -> str:
        return f"{super().csv_header()},lift_mtz,add_vi1,add_vi2,add_vi3,add_vi4,iterations_limit,iterations_used"

    def to_csv(self) -> str:
        return f"{super().to_csv()},{self.lift_mtz},{self.add_vi1},{self.add_vi2},{self.add_vi3},{self.add_vi4},{self.iterations_limit},{self.iterations_used}"


class FrankWolfe:
    EPS = 0.000001

    def __init__(self, instance: Instance, initial_y: ArrayLike, initial_w: ArrayLike, **kwargs):
        self.instance = instance
        self.current_y = initial_y
        self.current_w = initial_w
        self.p = np.array(self.instance.p)
        self.l = np.array(self.instance.l)
        self.lmask = self.l <= 0
        self.iterations_limit = kwargs.get('iter_limit', 150)
        self.step_type = kwargs.get('step_type', FrankWolfeStepType.DECREASING)

        self.__read_constraints_args(**kwargs)

        psum = np.dot(self.p, self.current_y)
        self.model = LinearModel(
            instance=self.instance,
            obj_type=LinearModelObjectiveFunction.FRANK_WOLFE,
            y_coeff=-self.p / psum,
            w_coeff=self.l,
            integer=False,
            add_vi1=self.add_vi1,
            add_vi2=self.add_vi2,
            add_vi3=self.add_vi3,
            add_vi4=self.add_vi4,
            lift_mtz=self.lift_mtz)

    def __read_constraints_args(self, **kwargs) -> None:
        self.add_vi = kwargs.get('add_vi', False)
        self.add_vi1 = False
        self.add_vi2 = False
        self.add_vi3 = False
        self.add_vi4 = False

        if self.add_vi:
            self.add_vi1 = True
            self.add_vi2 = True
            self.add_vi3 = True
            self.add_vi4 = True

        if 'add_vi1' in kwargs:
            self.add_vi1 = kwargs.get('add_vi1')
        if 'add_vi2' in kwargs:
            self.add_vi2 = kwargs.get('add_vi2')
        if 'add_vi3' in kwargs:
            self.add_vi3 = kwargs.get('add_vi3')
        if 'add_vi4' in kwargs:
            self.add_vi4 = kwargs.get('add_vi4')
  
        self.lift_mtz = kwargs.get('lift_mtz', False)

    def solve(self) -> FrankWolfeResults:
        start = time()

        for iter_num in range(self.iterations_limit):
            logging.debug(f"[FW] Iteration: {iter_num}")
            
            logging.debug(f"[FW] Obj at current point: (log={self.__compute_log_inv_obj(y=self.current_y, w=self.current_w):.3f}, orig={self.__compute_original_obj(y=self.current_y, w=self.current_w):.3f}).")

            new_sol = self.model.solve()
            new_y = new_sol.yvar_np
            new_w = new_sol.wvar_np

            logging.debug(f"[FW] Minimised the scalar product with the gradient via LP.")
            logging.debug(f"[FW] Obj at minimiser: (log={self.__compute_log_inv_obj(y=new_y, w=new_w):.3f}, orig={self.__compute_original_obj(y=new_y, w=new_w):.3f}).")

            if self.__is_local_min(new_y=new_y, new_w=new_w):
                logging.debug('[FW] Current point is a local minimum: quit!')
                break

            logging.debug('[FW] Current point is not a minimum, getting new point.')
            
            if self.step_type == FrankWolfeStepType.LINE_SEARCH:
                self.current_y, self.current_w = self.__line_search(pt1=(self.current_y, self.current_w), pt2=(new_y, new_w))
            else:
                self.current_y, self.current_w = self.__move_step(new_y=new_y, new_w=new_w, iter_num=iter_num)

            logging.debug('[FW] Updating coefficients for the next LP.')
            psum = np.dot(self.p, self.current_y)
            self.model.set_frank_wolfe_coeff(y_coeff=-self.p / psum, w_coeff=self.l, rebuild_obj=True)

        end = time()

        alg_name = 'frankwolfe'
        if self.add_vi1:
            alg_name += '_with_vi1'
        if self.add_vi2:
            alg_name += '_with_vi2'
        if self.add_vi3:
            alg_name += '_with_vi3'
        if self.add_vi4:
            alg_name += '_with_vi4'
        if self.lift_mtz:
            alg_name += '_liftMTZ'

        return FrankWolfeResults(
            instance=self.instance,
            algorithm=alg_name,
            lift_mtz=self.lift_mtz,
            add_vi1=self.add_vi1,
            add_vi2=self.add_vi2,
            add_vi3=self.add_vi3,
            add_vi4=self.add_vi4,
            tour_object=None,
            obj=self.__compute_original_obj(y=self.current_y, w=self.current_w),
            time_s=(end - start),
            iterations_limit=self.iterations_limit,
            iterations_used=iter_num + 1
        )

    def __is_local_min(self, new_y: ArrayLike, new_w: ArrayLike):
        cur_sol = np.concatenate((self.current_y, self.current_w), axis=None)
        new_sol = np.concatenate((new_y, new_w), axis=None)
        grad = self.__get_gradient(y=self.current_y)
        segment = cur_sol - new_sol

        return np.dot(grad, segment) <= self.EPS

    def __get_gradient(self, y: ArrayLike):
        denom = np.dot(self.p, y)
        return np.concatenate((-self.p / denom, self.l), axis=None)

    def __compute_original_obj(self, y: ArrayLike, w: ArrayLike):
        return np.dot(self.p, y) * np.prod(np.exp(- np.multiply(self.l, w)))

    def __compute_log_inv_obj(self, y: ArrayLike, w: ArrayLike):
        return - np.log(np.dot(self.p, y)) + np.dot(self.l, w)

    def __move_step(self, new_y: ArrayLike, new_w: ArrayLike, iter_num: int):
        gamma = 2.0 / (iter_num + 2)
        updated_y =(1 - gamma) * self.current_y + gamma * new_y
        updated_w = (1 - gamma) * self.current_w + gamma * new_w

        logging.debug(f"[FW] > Move step: got two points with objectives (log={self.__compute_log_inv_obj(y=self.current_y, w=self.current_w):.2f}, orig={self.__compute_original_obj(y=self.current_y, w=self.current_w):.2f}) and (log={self.__compute_log_inv_obj(y=new_y, w=new_w):.2f}, orig={self.__compute_original_obj(y=new_y, w=new_w):.2f}).")
        logging.debug(f"[FW] > Move step: gamma = {gamma:.2f}")
        logging.debug(f"[FW] > Move step: objective of the new point is (log={self.__compute_log_inv_obj(y=updated_y, w=updated_w):.2f}, orig={self.__compute_original_obj(y=updated_y, w=updated_w):.2f}).")

        return (updated_y, updated_w)

    def __line_search(self, pt1: Tuple[ArrayLike], pt2: Tuple[ArrayLike]):
        y1, w1 = pt1
        y2, w2 = pt2

        A = np.dot(self.p, y1 - y2)
        B = np.dot(self.l, w1 - w2)
        C = np.dot(self.p, y2)

        assert A != 0
        assert B != 0

        alpha = np.clip(1/B - C/A, a_min=0, a_max=1)

        new_pt_y = alpha * y1 + (1 - alpha) * y2
        new_pt_w = alpha * w1 + (1 - alpha) * w2

        logging.debug(f"[FW] > Line search: got two points with objectives (log={self.__compute_log_inv_obj(y=y1, w=w1):.2f}, orig={self.__compute_original_obj(y=y1, w=w1):.2f}) and (log={self.__compute_log_inv_obj(y=y2, w=w2):.2f}, orig={self.__compute_original_obj(y=y2, w=w2):.2f}).")
        logging.debug(f"[FW] > Line search: alpha wants to be = {1/B - C/A:.2f}")
        logging.debug(f"[FW] > Line search: alpha = {alpha:.2f}")
        logging.debug(f"[FW] > Line search: objective of the new point is (log={self.__compute_log_inv_obj(y=new_pt_y, w=new_pt_w):.2f}, orig={self.__compute_original_obj(y=new_pt_y, w=new_pt_w):.2f}).")

        return (new_pt_y, new_pt_w)
