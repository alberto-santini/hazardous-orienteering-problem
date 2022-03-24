from hop.instance import Instance
from hop.results import Results
from hop.tour import Tour
from hop.algorithms.labelling.strong import StrongLabel, strong_dominance
from hop.algorithms.labelling.relaxed import RelaxedLabel, relaxed_dominance
from hop.algorithms.labelling.relaxedtc import RelaxedTCLabel, relaxedtc_dominance
from collections import defaultdict
from dataclasses import dataclass
from colorama import Fore
from typing import Type, Callable, TypeVar, List
from time import time

import logging
logging.basicConfig(level=logging.ERROR)

__all__ = []

LabelClass = Type
Label = TypeVar('Label')
DominanceFunc = Callable[[Label, Label], bool]
ExcludeTrivialFunc = Callable[[Label], bool]


@dataclass
class LabellingResults(Results):
    non_dominated_labels: List[Label]
    still_active_labels: List[Label]
    best_label: Label

    def algorithm_completed(self) -> bool:
        return len(self.still_active_labels) == 0

    def csv_header(self) -> str:
        return f"{super().csv_header()},algorithm_completed,n_still_active_labels"

    def to_csv(self) -> str:
        return f"{super().to_csv()},{self.algorithm_completed()},{len(self.still_active_labels)}"


def __labelling_algorithm(i: Instance, label_class: LabelClass, dominance: DominanceFunc, exclude_trivial: ExcludeTrivialFunc, time_limit: float = 3600.0):
    labels = defaultdict(list)
    labels[0] = [label_class(i)]
    n_active_labels = 1
    start = time()
    last_print = time()

    def __get_any_label():
        for stored_labels in labels.values():
            for lbl in stored_labels:
                if lbl.unexplored:
                    return lbl

    def __log_labels():
        logging.debug(f"{Fore.BLUE}# of active labels: {n_active_labels}{Fore.RESET}")
        for v, lbls in labels.items():
            for lbl in lbls:
                if lbl.unexplored:
                    logging.debug(f"{Fore.GREEN}", end='')
                else:
                    logging.debug(f"{Fore.RED}", end='')
                logging.debug(f"{v} => {lbl}{Fore.RESET}")

    while (current_time := time()) - start <= time_limit:
        if n_active_labels == 0:
            break

        __log_labels()

        if current_time - last_print > 10:
            print(f"Active labels: {n_active_labels} - Elapsed time: {current_time - start:.2f}s")
            last_print = current_time
            
        label = __get_any_label()

        logging.debug(f"Extending {label}")

        for j in range(i.n_vertices):            
            new_label = label.extend(j)

            if new_label is None:
                # Unfeasible extension
                continue

            logging.debug(f"Feasible extension to {j}: {new_label}")

            # To avoid unnecessary (large) copies, we iterate over
            # label[j] in reverse order using indices. In this way,
            # we can remove elements while we iterate without invalidating
            # following indices.
            for old_label_id in range(len(labels[j]) - 1, -1, -1):
                old_label = labels[j][old_label_id]

                if dominance(new_label, old_label):
                    # New label dominates old label

                    logging.debug(f"{Fore.YELLOW}New label {new_label} dominates old label {old_label}{Fore.RESET}")

                    if old_label.unexplored:
                        n_active_labels -= 1

                    logging.debug(f"\t{Fore.BLUE}# of active labels: {n_active_labels}{Fore.RESET}")

                    del labels[j][old_label_id]

                    continue

                if dominance(old_label, new_label):
                    # Old label dominates new label

                    logging.debug(f"{Fore.YELLOW}Old label {old_label} dominates new label {new_label}{Fore.RESET}")

                    break
            else:
                # New label is undominated
                labels[j].append(new_label)

                logging.debug(f"{Fore.CYAN}New label {new_label} is not dominated{Fore.RESET}")

                # new_label could be considered not to explore (unexplored == False) if it's a
                # label back to the depot. We increase the counter of active labels only if
                # new_label must be explored in the future (unexplored == True).
                if new_label.unexplored:
                    n_active_labels += 1

                logging.debug(f"{Fore.BLUE}# of active labels: {n_active_labels}{Fore.RESET}")

        label.unexplored = False
        n_active_labels -= 1

        logging.debug(f"Finished extending label {label}")
        logging.debug(f"{Fore.BLUE}# of active labels: {n_active_labels}{Fore.RESET}")

    end = time()

    non_dominated = [label for label in labels[0] if exclude_trivial(label)]
    still_active_labels = [l for stored_labels in labels.values() for l in stored_labels if l.unexplored == True]
    best_label = max(non_dominated, key=lambda l: l.obj)

    logging.debug('Undominated labels:')
    for lbl in non_dominated:
        logging.debug(f"\t{lbl}")
    logging.debug(f"Best label: best_label")
    logging.debug(f"Labels left to explore: {len(still_active_labels)}")

    return LabellingResults(
        instance=i,
        algorithm=f"labelling_{label_class.__name__}",
        time_s=(end - start),
        obj=best_label.obj,
        tour_object=Tour(instance=i, tour=best_label.get_path()),
        non_dominated_labels=non_dominated,
        best_label=best_label,
        still_active_labels=still_active_labels
    )


def strong_labelling(i: Instance, time_limit: float = 3600):
    return __labelling_algorithm(i, label_class=StrongLabel, dominance=strong_dominance, exclude_trivial=lambda l: sum(l.visited) > 1, time_limit=time_limit)


def relaxed_labelling(i: Instance, time_limit: float = 3600):
    return __labelling_algorithm(i, label_class=RelaxedLabel, dominance=relaxed_dominance, exclude_trivial=lambda l: l.n_visited > 1, time_limit=time_limit)


def relaxedtc_labelling(i: Instance, time_limit: float = 3600):
    return __labelling_algorithm(i, label_class=RelaxedTCLabel, dominance=relaxedtc_dominance, exclude_trivial=lambda l: l.n_visited > 1, time_limit=time_limit)
