from hop.instance import Instance
from hop.results import Results
from hop.models.linear import LinearModel, LinearModelObjectiveFunction
from numpy.typing import ArrayLike
from typing import Tuple
from time import time
import numpy as np

import logging
logging.basicConfig(level=logging.ERROR)


class FrankWolfe:
    EPS = 0.000001

    def __init__(self, instance: Instance, add_vi: bool, initial_y: ArrayLike, initial_w: ArrayLike):
        self.instance = instance
        self.current_y = initial_y
        self.current_w = initial_w
        self.p = np.array(self.instance.p)
        self.l = np.array(self.instance.l)
        self.lmask = self.l <= 0

        psum = np.dot(self.p, self.current_y)
        self.model = LinearModel(
            instance=self.instance,
            obj_type=LinearModelObjectiveFunction.FRANK_WOLFE,
            y_coeff=-self.p / psum,
            w_coeff=self.l,
            integer=False,
            add_vi=add_vi)

    def solve(self) -> Results:
        start = time()

        for iter_num in range(100):
            logging.debug(f"[FW] Iteration: {iter_num}")

            new_sol = self.model.solve()
            new_y = new_sol.yvar_np
            new_w = new_sol.wvar_np

            logging.debug('[FW] Got new solution')
            logging.debug(f"[FW] current y: {self.current_y}")
            logging.debug(f"[FW] current w: {self.current_w}")
            logging.debug(f"[FW] f(current) = {self.__compute_log_inv_obj(y=self.current_y, w=self.current_w)}")
            logging.debug(f"[FW] g(current) = {self.__compute_original_obj(y=self.current_y, w=self.current_w)}")
            logging.debug(f"[FW] new y: {new_y}")
            logging.debug(f"[FW] new w: {new_w}")
            logging.debug(f"[FW] f(new): {self.__compute_log_inv_obj(y=new_y, w=new_w)}")
            logging.debug(f"[FW] g(new): {self.__compute_original_obj(y=new_y, w=new_w)}")

            if self.__is_local_min(new_y=new_y, new_w=new_w):
                logging.debug('[FW] Is local minimum for f(): quit!')
                self.current_y, self.current_w = new_y, new_w
                break

            self.current_y, self.current_w = self.__line_search(pt1=(self.current_y, self.current_w), pt2=(new_y, new_w))

            logging.debug('[FW] Got new current solution by line search')
            logging.debug(f"[FW] updated current y: {self.current_y}")
            logging.debug(f"[FW] updated current w: {self.current_w}")
            logging.debug(f"[FW] f(updated current) = {self.__compute_log_inv_obj(y=self.current_y, w=self.current_w)}")
            logging.debug(f"[FW] g(updated current) = {self.__compute_original_obj(y=self.current_y, w=self.current_w)}")

            psum = np.dot(self.p, self.current_y)
            self.model.set_frank_wolfe_coeff(y_coeff=-self.p / psum, w_coeff=self.l, rebuild_obj=True)
            logging.debug('[FW] Updated Frank-Wolfe Model')

        end = time()

        return Results(
            instance=self.instance,
            algorithm='frankwolfe',
            tour_object=None,
            obj=self.__compute_original_obj(y=self.current_y, w=self.current_w),
            time_s=(end - start)
        )

    def __is_local_min(self, new_y: ArrayLike, new_w: ArrayLike):
        cur_sol = np.concatenate((self.current_y, self.current_w), axis=None)
        new_sol = np.concatenate((new_y, new_w), axis=None)

        return np.dot(self.__get_gradient(y=new_y), new_sol - cur_sol) <= self.EPS

    def __get_gradient(self, y: ArrayLike):
        denom = np.dot(self.p, y)
        return np.concatenate((-self.p / denom, self.l), axis=None)

    def __compute_original_obj(self, y: ArrayLike, w: ArrayLike):
        return np.sum(np.dot(self.p, y)) * \
               np.prod(np.exp(- np.multiply(self.l, w)))

    def __compute_log_inv_obj(self, y: ArrayLike, w: ArrayLike):
        return - np.log(np.dot(self.p, y)) + np.dot(self.l, w)

    def __line_search(self, pt1: Tuple[ArrayLike], pt2: Tuple[ArrayLike]):
        y1, w1 = pt1
        y2, w2 = pt2

        A = np.dot(self.p, y1 - y2)
        B = np.dot(self.l, w1 - w2)
        C = np.dot(self.p, y2)

        assert A != 0
        assert B != 0

        alpha = np.clip(1/B - C/A, a_min=0, a_max=1)

        logging.debug(f"[FW] alpha = {alpha}")

        return (
            alpha * y1 + (1 - alpha) * y2,
            alpha * w1 + (1 - alpha) * w2
        )
