from __future__ import annotations
from hop.instance import Instance
from copy import deepcopy
from random import randint, sample, choice
from typing import Sequence, Tuple, List
from colorama import Style
from numpy import random as nprandom
import numpy as np

class Tour:
    infeasible_penalty = 999

    def __init__(self, instance: Instance, **kwargs) -> None:
        self.instance = instance
        self.tour = kwargs.get('tour', [0,0])
        self.__compute_stats()

    def __compute_wpp(self) -> Tuple[List, float, float]:
        assert self.tour[0] == 0
        assert self.tour[-1] == 0

        w = list()
        for idx in range(len(self.tour) - 1, 0, -1):
            t = self.instance.t[self.tour[idx]][self.tour[idx - 1]]

            if len(w) == 0:
                w = [t]
            else:
                w.append(w[-1] + t)

        w = list(reversed(w))

        sum_profits = sum([self.instance.p[i] for i in self.tour[:-1]])
        product_probs = 1
        for idx, iw in enumerate(w):
            product_probs *= np.exp(- iw * self.instance.l[self.tour[idx]])

        return w, sum_profits, product_probs

    def __compute_stats(self) -> None:
        """ Computes:
            * Vector self.w, which contains travel times after each node
            * self.sum_profits, the sum-of-profits component of the obj func
            * self.product_probs, the product-of-probabilities component of the obj func
        """
        self.w, self.sum_profits, self.product_probs = self.__compute_wpp()

    def __valid_stats(self) -> bool:
        w, sum_profits, product_probs = self.__compute_wpp()
        eq = lambda x, y: abs(x - y) < 1e-3
        eqv = lambda u, v: all(eq(x, y) for x, y in zip(u, v))

        if not eqv(w, self.w):
            print('Wrong vector w!')
            print(f"Current w: {self.w}")
            print(f"Correct w: {w}")

        if not eq(sum_profits, self.sum_profits):
            print('Wrong sum of profits!')
            print(f"Current sum: {self.sum_profits}")
            print(f"Correct sum: {sum_profits}")

        if not eq(product_probs, self.product_probs):
            print('Wrong product of probs!')
            print(f"Current prod: {self.product_probs}")
            print(f"Correct prod: {product_probs}")

        return eqv(w, self.w) and eq(sum_profits, self.sum_profits) and eq(product_probs, self.product_probs)

    def __tour_without_index(self, index: int) -> List[int]:
        """ Gives a copy of self.tour, without the entry at index `index`. """
        return self.tour[:index] + self.tour[index+1:]

    def __tour_with_customer_after_position(self, customer: int, position: int) -> List[int]:
        """ Gives a copy of self.tour, with cutomer `customer` inserted after
            the entry currently at position `position`.
        """
        return self.tour[:position+1] + [customer] + self.tour[position+1:]

    def __tour_with_swapped_indices(self, i: int, j: int) -> List[int]:
        """ Gives a copy of self.tour, with entries at positions `i` and `j`
            swapped.
        """
        tour = deepcopy(self.tour)
        tour[i], tour[j] = tour[j], tour[i]
        return tour

    def __without_index(self, index: int) -> Tour:
        """ Returns a new Tour, without the customer at position `index` in
            the current tour.
        """
        assert index >= 1
        assert index < len(self.tour) - 1
        assert self.__valid_stats()

        new_tour = deepcopy(self)

        cust = self.tour[index]
        new_tour.sum_profits -= self.instance.p[cust]
        new_tour.product_probs /= np.exp(- self.w[index] * self.instance.l[cust])

        pred = self.tour[index - 1]
        succ = self.tour[index + 1]
        time_diff = self.instance.t[pred][cust] + self.instance.t[cust][succ] - self.instance.t[pred][succ]

        for idx, icust in enumerate(self.tour[:index]):
            new_tour.w[idx] -= time_diff
            new_tour.product_probs *= np.exp(time_diff * self.instance.l[icust])

        del new_tour.w[index]
        new_tour.tour = self.__tour_without_index(index=index)

        assert new_tour.n_customers() == self.n_customers() - 1
        assert new_tour.__valid_stats()

        return new_tour

    def __without_best_index(self) -> Tour:
        """ Returns a new tour with one fewer customer than the current tour.
            The customer removed is the one which would produce the best
            possible objective function.
        """
        return max(
            [
                self.__without_index(index=index)
                for index in self.customer_indices()
            ],
            key=lambda t: t.hierarchical_objval()
        )

    def __with_customer_after_position(self, customer: int, position: int) -> Tour:
        """ Returns a new tour, with customer `customer` inserted after the
            entry which is now at position `position` in the tour.
        """
        assert customer not in self.tour
        assert position >= 0
        assert position < len(self.tour) - 1
        assert self.__valid_stats()

        new_tour = deepcopy(self)

        pred = self.tour[position]
        succ = self.tour[position + 1]
        time_diff = self.instance.t[pred][customer] + self.instance.t[customer][succ] - self.instance.t[pred][succ]
        new_tour.sum_profits += self.instance.p[customer]

        for idx, curr in enumerate(self.tour[:position+1]):
            new_tour.w[idx] += time_diff
            new_tour.product_probs /= np.exp(time_diff * self.instance.l[curr])

        time_after_cust = self.instance.t[customer][succ]
        if position + 1 < len(self.w):
            # If not inserting as the very last customer...
            time_after_cust += self.w[position + 1]

        new_tour.w.insert(position + 1, time_after_cust)
        new_tour.product_probs *= np.exp(- new_tour.w[position+1] * self.instance.l[customer])
        new_tour.tour = self.__tour_with_customer_after_position(customer=customer, position=position)

        assert new_tour.n_customers() == self.n_customers() + 1
        assert new_tour.__valid_stats()

        return new_tour

    def __with_customer_at_best_position(self, customer: int) -> Tour:
        """ Returns a new tour, with customer `customer` inserted in the
            best possible position. The new tour could have better or worse
            objective value than the current tour.
        """
        assert customer not in self.tour

        return max(
            [
                self.__with_customer_after_position(customer=customer, position=position)
                for position in range(self.n_customers() + 1)
            ], key=lambda t: t.hierarchical_objval())

    def n_customers(self) -> int:
        """ Number of customers in the tour (does not include the depot). """
        assert len(self.tour) >= 2
        assert self.tour[0] == 0
        assert self.tour[-1] == 0

        return len(self.tour) - 2

    def customer_indices(self) -> Sequence[int]:
        """ Returns valid indices for customers in self.tour.
            It does not include the indices of the depot.
        """
        assert len(self.tour) >= 2
        assert self.tour[0] == 0
        assert self.tour[-1] == 0

        return range(1, len(self.tour) - 1)

    def customers(self) -> List[int]:
        """ Returns the set of visited customers (as an ordered list,
            whose order is the visit order).
        """
        assert len(self.tour) >= 2
        assert self.tour[0] == 0
        assert self.tour[-1] == 0

        return self.tour[1:-1]

    def det_customers(self) -> List[int]:
        """ Returns the set of deterministic customers visited by the tour. """
        return [i for i in self.customers() if self.instance.l[i] == 0]

    def tb_customers(self) -> List[int]:
        """ Returns the set of time-bomb customers visited by the tour. """
        return [i for i in self.customers() if self.instance.l[i] > 0]

    def duration(self) -> float:
        """ Tour time duration. """
        return self.w[0]

    def objval(self) -> float:
        """ Tour objective function value. """
        return self.sum_profits * self.product_probs

    def hierarchical_objval(self) -> float:
        return self.objval(), -self.duration()

    def fitness(self) -> float:
        """ Tour profit minus a penalty for infeasibility. """
        if self.duration() > self.instance.time_bound:
            return self.objval() - self.infeasible_penalty
        else:
            return self.objval()

    def hierarchical_fitness(self) -> Tuple[float, float]:
        return self.fitness(), -self.duration()

    def det_tb_indices(self) -> Tuple[List[int], List[int]]:
        det_indices = list()
        tb_indices = list()

        for idx in self.customer_indices():
            if self.instance.l[self.tour[idx]] > 0:
                tb_indices.append(idx)
            else:
                det_indices.append(idx)

        return det_indices, tb_indices

    def remove_best(self, n: int = 1) -> Tour:
        """ Returns a new tour with exactly `n` fewer customer than the
            current tour. Customers are removed one-by-one. Each time a
            customer is to be removed, it is chosen as the one which gives
            the best objective function after being removed. If the tour has
            fewer than `n` customers, it is emptied.
        """
        assert n > 0

        if n >= self.n_customers():
            return Tour(instance=self.instance)

        tour = self.__without_best_index()
        for _ in range(n-1):
            tour = tour.__without_best_index()

        assert tour.n_customers() == self.n_customers() - n

        return tour

    def remove_best_if_improving_else_random(self) -> Tour:
        """ Returns a new tour with exactly one fewer customer than the
            current tour. If there is at least one customer whose removal
            improves over the current tour, it removes the customer causing
            the best improvement. If there is no customer which, when removed,
            improves over the current tour, it removes a random customer.
            This method, therefore, focuses more on diversification compared
            with `Tour::best_removal`.
        """
        tour = self.remove_best()

        if tour.hierarchical_objval() > self.hierarchical_objval():
            return tour
        else:
            return self.remove_random()

    def remove_random(self, n: int = 1) -> Tour:
        """ Returns a new tour with exactly `n` fewer customers than the
            current tour. It removes a random customer. If `n` is larger
            or equal to the current number of customers, it gives an empty
            tour.
        """
        assert n > 0

        if n >= self.n_customers():
            return Tour(instance=self.instance)

        tour = self.__without_index(randint(1, self.n_customers()))

        for _ in range(n-1):
            tour = tour.__without_index(randint(1, tour.n_customers()))

        assert tour.n_customers() == self.n_customers() - n

        return tour

    def insert_best_customer_best_position(self) -> Tour:
        """ Returns a new tour in which the best of the remaining customers
            (customers which are not currently in the tour) is inserted
            in the best possible position in the current tour. If all
            customers are already in the tour, it returns a copy of the
            current tour.
        """
        remaining = [i for i in range(1, self.instance.n_vertices) if i not in self.tour]

        if len(remaining) == 0:
            return deepcopy(self)

        return max(
            [
                self.__with_customer_at_best_position(customer=i)
                for i in remaining
            ],
            key=lambda t: t.hierarchical_objval()
        )

    def insert_random_customer_best_position(self) -> Tour:
        """ Returns a new tour in which one of the remaining customers
            (customers which are not currently in the tour) is inserted
            in the best possible position in the current tour. If all
            customers are already in the tour, it returns a copy of the
            current tour.
        """
        remaining = [i for i in range(1, self.instance.n_vertices) if i not in self.tour]

        if len(remaining) == 0:
            return deepcopy(self)

        customer = choice(remaining)

        return self.__with_customer_at_best_position(customer=customer)

    def insert_random_customer_random_position(self) -> Tour:
        """ Returns a new tour in which one of the remaining customers
            (customers which are not currently in the tour) is inserted
            at a random position in the current tour. If all customers are
            already in the tour, it returns a copy of the current tour.
        """
        remaining = [i for i in range(1, self.instance.n_vertices) if i not in self.tour]

        if len(remaining) == 0:
            return deepcopy(self)

        customer = choice(remaining)
        position = choice(range(0, self.n_customers() + 1))

        return self.__with_customer_after_position(customer=customer, position=position)

    def swap_indices(self, i: int, j: int) -> Tour:
        """ Returns a new tour in which customers at positions `i` and `j`
            are swapped.
        """
        assert i in self.customer_indices()
        assert j in self.customer_indices()

        return Tour(instance=self.instance, tour=self.__tour_with_swapped_indices(i, j))

    def swap_earlier_tb_with_later_det(self) -> Tour:
        """ Returns another tour with exactly one pair of swapped customers,
            compared to the current tour (see below for a possible exception
            to this definition).
            It returns the best tour obtained swapping a time-bomb customer
            with a deterministic customer, if the time-bomb customer precedes
            the deterministic one in the tour.
            If there are no customers which satisfy the above condition, it
            returns a copy of the current tour.
        """
        det_indices, tb_indices = self.det_tb_indices()

        try:
            return max(
                [
                    self.swap_indices(det_index, tb_index)
                    for det_index, tb_index in zip(det_indices, tb_indices)
                    if det_index > tb_index
                ], key=lambda t: t.hierarchical_objval())
        except ValueError:
            return deepcopy(self)

    def swap_earlier_more_tb_with_later_less_tb(self) -> Tour:
        """ Returns another tour with exactly one pair of swapped customers,
            compared to the current tour (see below for a possible exception
            to this definition).
            It returns the best tour obtained swapping a time-bomb customer i
            with a time-bomb customer j, if i precedes j in the tour and i is
            "more time-bomb" than j, i.e., if the exponential parameter of i
            is larger than that of j.
            If there are no customers which satisfy the above condition, it
            returns a copy of the current tour.
        """
        _, tb_indices = self.det_tb_indices()

        try:
            return max(
                [
                    self.swap_indices(i, j)
                    for idx, i in enumerate(tb_indices)
                    for j in tb_indices[idx+1:]
                    if i < j and self.instance.l[self.tour[i]] > self.instance.l[self.tour[j]]
                ], key=lambda t: t.hierarchical_objval()
            )
        except ValueError:
            return deepcopy(self)

    def swap_det_with_det(self) -> Tour:
        """ Returns another tour with exactly one pair of swapped customers,
            compared to the current tour (see below for a possible exception
            to this definition).
            It returns the best tour obtained swapping two deterministic
            customers. If there are fewer than two deterministic customers, it
            returns a copy of the current tour.
        """
        det_indices, _ = self.det_tb_indices()

        if len(det_indices) < 2:
            return deepcopy(self)

        return max(
            [
                self.swap_indices(i, j)
                for idx, i in enumerate(det_indices)
                for j in det_indices[idx+1:]
            ], key=lambda t: t.hierarchical_objval()
        )

    def swap_random_with_random(self) -> Tour:
        """ If the current tour has at least two customers, it returns
            a new tour with a random pair of swapped customers. Otherwise
            it returns a copy of the current tour.
        """
        if self.n_customers() < 2:
            return deepcopy(self)

        i, j = sample(self.customer_indices(), k=2)

        return self.swap_indices(i, j)

    def make_feasible(self) -> None:
        """ Removes customers from the tour, until it is time-feasible.
            At each iteration, it greedily removes the customer which gives the
            best objective value. (This procedure could actually strictly
            improve the objective value!)
        """
        while self.duration() > self.instance.time_bound:
            best_tour = self.__without_index(index=1)
            for idx in range(2, len(self.tour) - 1):
                new_tour = self.__without_index(index=idx)
                if new_tour.objval() > best_tour.objval():
                    best_tour = new_tour

            # Replace self
            self.__dict__.update(best_tour.__dict__)

    def __str__(self) -> str:
        def b(x: int) -> str:
            return f"{Style.BRIGHT}{x}{Style.RESET_ALL}" if self.instance.l[x] > 0 else str(x)

        s_tour = [b(i) for i in self.tour]
        s = '[' + ', '.join(s_tour) + ']'
        s += f" obj: {self.objval():.3f}, Σ: {self.sum_profits:.1f},"
        s += f" Π: {self.product_probs:.3f}, t: {self.duration():.1f}/{self.instance.time_bound:.1f}"

        return s

    def __hash__(self):
        assert len(self.tour) >= 2
        return hash(tuple(self.tour[1:-1]))

    def __eq__(self, other):
        assert len(self.tour) >= 2
        assert len(other.tour) >= 2
        return self.tour[1:-1] == other.tour[1:-1]

    @staticmethod
    def get_random(instance: Instance):
        t = nprandom.permutation(range(1, instance.n_vertices))
        tour = Tour(instance=instance, tour=[0] + list(t) + [0])
        tour.make_feasible()
        return tour
