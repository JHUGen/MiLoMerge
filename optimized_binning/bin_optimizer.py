import warnings
import numpy as np
import tqdm
import numba as nb
import dask.array as da
import time

@nb.njit(fastmath=True, cache=True)
def mlm_driver_comp_to_all(n, counts, weights, b, b_prime):

    metric_val = 0
    for h in np.arange(n):
        for h_prime in nb.prange(h+1, n):
            t1, t2 = counts[h][b]*weights[h][h_prime], counts[h_prime][b_prime]*weights[h][h_prime]
            t3, t4 = counts[h][b_prime]*weights[h][h_prime], counts[h_prime][b]*weights[h][h_prime] 

            metric_val += (t1*t2)**2 + (t3*t4)**2 - 2*t1*t2*t3*t4

    return metric_val

class Merger():
    def __init__(
            self,
            bin_edges,
            *counts,
            weights=None,
            comp_to_first=False
        ) -> None:

        it = iter(counts)
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            errortext = [str(len(count)) for count in counts]
            errortext = " ".join(errortext)
            errortext = 'Not all counts have same length! Lengths are: ' + errortext
            raise ValueError('\n'+errortext)

        if len(counts[0]) != len(bin_edges) - 1:
            errortext = f"len(counts) = {len(counts[0])} != len(bin_edges) - 1 = {len(bin_edges)-1}"
            raise ValueError('\n'+errortext)

        if weights is not None and len(weights) != len(counts):
            errortext = "The # of weight values and the # of hypotheses should be the same!"
            errortext += f'\nThere are {len(counts)} hypotheses and {len(weights)} weight values'
            raise ValueError('\n'+errortext)

        if weights is None:
            weights = np.ones(len(counts), dtype=float)
        else:
            weights = np.array(weights)

        self._merger_type = None

        self.weights = np.outer(weights, weights)
        self.n_hypotheses = len(counts)
        self.n_items = self.original_n_items = len(counts[0])
        #self.n is the number of hypotheses, self.n_items is the current number of bin_edges

        self.comp_to_first = comp_to_first

        self.counts = np.vstack(counts)
        self.counts /= self.counts.sum(axis=1)[:, None]

        self.bin_edges = np.array(bin_edges)

    @staticmethod
    @nb.njit(fastmath=True, cache=True, nogil=True)
    def __mlm_driver_comp_to_first(n, counts, weights, b, b_prime):

        metric_val = 0
        for h_prime in nb.prange(1, n):
            t1, t2 = counts[0][b]*weights[0][h_prime], counts[h_prime][b_prime]*weights[0][h_prime]
            t3, t4 = counts[0][b_prime]*weights[0][h_prime], counts[h_prime][b]*weights[0][h_prime] 

            metric_val += (t1*t2)**2 + (t3*t4)**2 - 2*t1*t2*t3*t4

        return metric_val

    @staticmethod
    @nb.njit(fastmath=True, cache=True, nogil=True)
    def __mlm_driver_comp_to_all(n, counts, weights, b, b_prime):

        metric_val = 0
        for h in np.arange(n):
            for h_prime in nb.prange(h+1, n):
                t1, t2 = counts[h][b]*weights[h][h_prime], counts[h_prime][b_prime]*weights[h][h_prime]
                t3, t4 = counts[h][b_prime]*weights[h][h_prime], counts[h_prime][b]*weights[h][h_prime] 

                metric_val += (t1*t2)**2 + (t3*t4)**2 - 2*t1*t2*t3*t4

        return metric_val

    def _mlm(self, b, b_prime):
        if self.comp_to_first:
            return self.__mlm_driver_comp_to_first(
                self.n_hypotheses, self.counts,
                self.weights, b, b_prime
            )

        return self.__mlm_driver_comp_to_all(
            self.n_hypotheses, self.counts,
            self.weights, b, b_prime
        )

    def __repr__(self) -> str:
        return f"Merger of type {self._merger_type} merging {self.original_n_items}"

class MergerLocal(Merger):
    def __init__(
            self,
            bin_edges,
            *counts,
            weights=None,
            comp_to_first=False
        ) -> None:
        super().__init__(bin_edges, *counts, weights=weights, comp_to_first=comp_to_first)
        if len(self.bin_edges.shape) > 1:
            raise ValueError("LOCAL MERGING CAN ONLY HANDLE 1-DIMENSIONAL ARRAYS")

        self._merger_type = "Local"

    def _merge(self, i, j):
        if i == j + 1:
            if i > self.n_items - 1 or i < 0:
                raise ValueError("UNHANDLED EDGE CASE WHILE MERGING")
            temp_counts = np.concatenate(
                (
                    self.counts[:, :j],
                    np.array([self.counts[:, j] + self.counts[:, i]]).T,
                    self.counts[:, i+1:]
                ),
                axis=1
            )
            temp_edges = np.concatenate((self.bin_edges[:i], self.bin_edges[i+1:]))

        elif i == j - 1:
            if i > self.n_items - 1 or i < 1:
                raise ValueError("UNHANDLED EDGE CASE WHILE MERGING")
            temp_counts = np.concatenate(
                (
                    self.counts[:, :i],
                    np.array([self.counts[:, i] + self.counts[:, j]]).T,
                    self.counts[:, j+1:]
                ),
                axis=1
            )
            temp_edges = np.concatenate((self.bin_edges[:i+1], self.bin_edges[i+2:]))

        elif i == j:
            pass

        else:
            raise ValueError("This is local binning! Can only merge ahead or behind!")

        return (temp_counts, temp_edges)

    def run(self, target_bin_number=1):
        if self.n_items <= target_bin_number:
            warnings.warn("Merging is pointless! Number of bins already >= target")

        pbar = tqdm.tqdm(
            total=self.n_items - target_bin_number,
            desc="Binning locally:", leave=True, position=0
            )
        while self.n_items > target_bin_number:
            best_combo = None
            best_score = np.inf

            for i in range(1, self.n_items - 1):
                score = self._mlm(i, i+1)

                if score < best_score:
                    best_combo = (i, i+1)
                    best_score = score

            score = self._mlm(1, 0)
            if score < best_score:
                best_combo = (1, 0)

            self.counts, self.bin_edges = self._merge(*best_combo)

            self.n_items -= 1
            pbar.update(1)

        return self.bin_edges



class MergerNonlocal(Merger):
    def __init__(
            self,
            bin_edges,
            *counts,
            weights=None,
            comp_to_first=False,
            tracker_size=50
        ) -> None:

        unrolled_counts = list(map(np.ndarray.ravel, map(np.array, counts)))
        unrolled_bins = np.arange(len(unrolled_counts[0]) + 1)

        super().__init__(
            unrolled_bins, *unrolled_counts,
            weights=weights, comp_to_first=comp_to_first
            )

        self._merger_type = "Non-local"

        self.physical_bins = da.array(bin_edges)

        if len(self.physical_bins.shape) > 1:
            self.n_observables = bin_edges.shape[0]
        else:
            self.n_observables = 1

        self.__cur_iteration_tracker = {}
        self.scores = np.zeros((self.n_items, self.n_items), dtype=np.float64)
        self.things_to_recalculate = tuple(range(self.n_items))

    def _merge(self, i, j):
        k = 0
        old_1 = old_2 = None
        hit_first = False
        for c in np.arange(self.n_items):
            if c not in (i, j):
                if c != k:
                    self.counts[:, k] = self.counts[:, c]
                    self.scores[:, k], self.scores[k] = self.scores[:, c], self.scores[c]
                
                k += 1
            elif hit_first:
                old_2 = self.counts[:, c].copy()
            else:
                old_1 = self.counts[:, c].copy()
                hit_first = True

        self.counts[:, k] = old_1 + old_2
        self.counts = self.counts[:, :-1]
        
        self.scores[k:] = np.inf
        self.scores[:,k:] = np.inf

        self.things_to_recalculate = (k, )


    def _closest_pair(self):
        for i in np.arange(self.n_items, dtype=np.int32):
            for j in self.things_to_recalculate:
                if i == j:
                    self.scores[i][j] = np.inf
                    continue
                self.scores[i][j] = self._mlm(i, j)

        smallest_distance_index = np.unravel_index(self.scores.argmin(), self.scores.shape)

        return smallest_distance_index

    def run(self, target_bin_number=1):

        pbar = tqdm.tqdm(
            total=self.n_items - target_bin_number,
            desc="Binning non-locally:", leave=True, position=0
            )
        while self.n_items > target_bin_number:
            min_1, min_2 = self._closest_pair()
            self._merge(min_1, min_2)

            self.n_items -= 1
            pbar.update(1)

        return self.counts, self.bin_edges[:self.n_items + 1]
