from hop.instance.tsiligirides import TsiligiridesInstance
from hop.utils import tsiligirides_hop_dir
import os

time_bounds = {
    1: [5, 10, 15, 20, 25, 30, 35, 40, 46, 50, 55, 60, 65, 70, 73, 75, 80, 85],
    2: [15, 20, 23, 25, 27, 30, 32, 35, 38, 40, 45],
    3: [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110]
}

alpha = [.10, .20, .30, .40]

beta = [2, 3, 4, 5]

if __name__ == '__main__':
    destination_folder = tsiligirides_hop_dir()

    for num in [1, 2, 3]:
        for tb in time_bounds[num]:
            for a in alpha:
                for b in beta:
                    i = TsiligiridesInstance(num=num, time_bound=tb, alpha=a, beta=b)
                    instance_file = os.path.join(destination_folder, f"hop_tsiligirides-{num}-{tb}-{a}-{b}.json")
                    i.save(filename=instance_file)
