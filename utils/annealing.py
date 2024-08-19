from argparse import Namespace


class KLAnnealer:
    def __init__(self, args: Namespace, current_epoch=0):
        self.offset = current_epoch
        self.num_epoch = args.epochs
        self.kl_anneal_type = args.kl_anneal_type
        self.kl_anneal_ratio = args.kl_anneal_ratio
        self.kl_anneal_cycle = args.kl_anneal_cycle
        self.beta = self.frange_cycle_linear(
            n_iter=self.offset,
            n_cycle=self.kl_anneal_cycle,
            ratio=self.kl_anneal_ratio,
        )

    def update(self):
        self.offset += 1
        self.beta = self.frange_cycle_linear(
            n_iter=self.offset,
            n_cycle=self.kl_anneal_cycle,
            ratio=self.kl_anneal_ratio,
        )

    def get_beta(self):
        return self.beta

    def frange_cycle_linear(self, n_iter: int, start=0.0, stop=1.0, n_cycle=1, ratio=1):
        if self.kl_anneal_type == "Cyclical":
            idx = n_iter % n_cycle
            step = idx * (stop - start) / (n_cycle * ratio)
            return min(start + step, 1.0)
        elif self.kl_anneal_type == "Monotonic":
            return min(start + n_iter * (stop - start) / (n_cycle * ratio), 1.0)
        else:
            # Fallback to no annealing
            return 1.0
