import warnings
import numpy as np
import awkward as ak
import tqdm
import numba as nb
import copy
import multiprocessing

class Split_Node(object):
    def __init__(self, str_cut, parent=None) -> None:
        if parent is not None:
            self.parent = parent
            self.parent.children.append(self)
        else:
            self.parent = None

        self.str_cut = str_cut
        self.fraction = 0

        self.children = []

    def is_leaf(self):
        return len(self.children) == 0
    
    def is_root(self):
        return self.parent is None
    
    def get_all_cuts_as_str(self):
        if self.parent is None:
            return self.str_cut
        else:
            overall_cuts = []
            cur_node = self
            while cur_node.parent is not None:
                overall_cuts.append(f"({cur_node.str_cut})")
                cur_node = cur_node.parent
            return " & ".join(reversed(overall_cuts))
    
    def apply_all_cuts(self, array_name):
        overall_cuts = []
        cur_node = self
        while cur_node.parent is not None:
            if ">" in cur_node.str_cut:
                observable, edge = cur_node.str_cut.strip().split(">")
                overall_cuts.append(f"({array_name}.{observable} > {edge})")
            elif "<=" in cur_node.str_cut:
                observable, edge = cur_node.str_cut.strip().split("<=")
                overall_cuts.append(f"({array_name}.{observable} <= {edge})")

            cur_node = cur_node.parent

        if not len(overall_cuts):
            return None
        return " & ".join(overall_cuts)
            
    
    def get_leaves(self):
        if self.is_leaf():
            return [self]
        else:
            return sum([i.get_leaves() for i in self.children], start=[])
    
    def __repr__(self):
        return self.str_cut
    
    def to_dict(self, parent=None, start=0, tree=dict()):
        if self.is_node():
            tree[start] = {"cut":self.str_cut, "parent":None, "children":[]}
            parent = start
            for node in self.children:
                start += 1
                tree[parent]["children"].append(start)
                node.to_dict(parent, start, tree)
        else:
            tree[start] = {"cut":self.str_cut, "parent":None, "children":[]}
            parent = start
            for node in self.children:
                start += 1
                tree[parent]["children"].append(start)
                node.to_dict(parent, start, tree)
        return tree
    

class Splitter(object):
    def __init__(
        self,
        *data,
        weights=None,
        hypothesis_weights=None,
        comp_to_first_only = False
        ) -> None:

        if not np.all(np.array([len(data[n]) for n in range(len(data))]) == len(data[0])):
            raise ValueError("All dictionaries must have the same length!")

        self.observables = tuple(data[0].keys())
        self.n_observables = len(self.observables)
        self.n_hypotheses = len(data)
        self.maxima_and_minima = np.zeros((self.n_observables, 2))
        
        if comp_to_first_only:
            self.h_value = np.array([0], dtype=float)
        else:
            self.h_value = np.arange(self.n_hypotheses)

        if weights is None:
            weights = [np.fill(1) for i in self.n_observables]
        for n, weight in enumerate(weights):
            data[n]["w"] = weight

        if len(weights) != len(data):
            raise ValueError("data and weights should be the same length!")

        if hypothesis_weights is None:
            hypothesis_weights = np.ones(self.n_hypotheses, dtype=np.float64)
        else:
            hypothesis_weights = np.array(hypothesis_weights, dtype=np.float64)
        self.hypothesis_weights = np.outer(hypothesis_weights, hypothesis_weights)

        self.data = {observable:None for observable in self.observables + ("w",)}
        for observable in self.observables + ("w",):
            self.data[observable] = ak.Array(
                np.column_stack(
                    [obs_dict[observable] for obs_dict in data]
                )
            )
        self.data = ak.from_regular(ak.Array(self.data))
        self.total = ak.sum(self.data.w, axis=0)
        
        del data, weights, weight, observable
        
        self.decision_tree = None

    def _score(self, mask_1, mask_2):
        b1 = ~ak.is_none(ak.mask(self.data, mask_1), axis=1)
        
        b2 = ~ak.is_none(ak.mask(self.data, mask_2), axis=1)
        
        metric_val = 0
        for h in self.h_value:
            for h_prime in np.arange(h+1, self.n_hypotheses):
                t1 = ak.sum(self.data[:,h][b1[:,h]].w)/self.total[h]
                t2 = ak.sum(self.data[:,h_prime][b2[:,h_prime]].w)/self.total[h_prime]
                t3 = ak.sum(self.data[:,h][b2[:,h]].w)/self.total[h]
                t4 = ak.sum(self.data[:,h_prime][b1[:,h]].w)/self.total[h_prime]
                
                hyp_weight = self.hypothesis_weights[h][h_prime]**4
                
                metric_val += ((t1*t2)**2 + (t3*t4)**2 - 2*t1*t2*t3*t4)*hyp_weight

        return metric_val

    def check_split(self, n, possible_parent, big_items, stat_limit=0.025):
        best_score = -1
        cut = None
        cut_2 = None
        cut_str = ""
        
        eval_str = possible_parent.apply_all_cuts("self.data")
        if eval_str is None:
            parent_cuts_mask = True
        else:
            parent_cuts_mask = eval(eval_str)
        
        for observable, possible_edge_list in big_items:
            for edge in possible_edge_list:
                new_cut = f"(self.data.{observable} > {edge})"
                new_cut_2 = f"(self.data.{observable} <= {edge})"
                cuts = (parent_cuts_mask) & eval(new_cut)
                cuts_2 = (parent_cuts_mask) & eval(new_cut_2)
                iteration_cuts_as_str_temp = (
                    f"{observable} > {edge:.2f}",
                    f"{observable} <= {edge:.2f}"
                )

                # if not ak.any(cuts) or not ak.any(cuts_2):
                #     continue
                # elif all([i in possible_parent.get_all_cuts_as_str() for i in iteration_cuts_as_str_temp]):
                #     continue
                check1 = (ak.sum(self.data.w[cuts], axis=0)/self.total < stat_limit)
                check2 = (ak.sum(self.data.w[cuts_2], axis=0)/self.total < stat_limit)

                check_val = check1 | check2

                print("eval:", eval_str)
                print("trying:", iteration_cuts_as_str_temp)
                print("check1:", check1, ak.sum(self.data.w[cuts], axis=0)/self.total)
                print("check2:", check2, ak.sum(self.data.w[cuts_2], axis=0)/self.total)
                
                print("check_val OR len(check_val)", check_val, not len(check_val))

                if ak.any(check_val) or not len(check_val):
                    print("SKIPPING")
                    continue
                score_temp = self._score(cuts, cuts_2)
                if score_temp > best_score:
                    cut = cuts
                    cut_2 = cuts_2
                    cut_str = iteration_cuts_as_str_temp
                    best_score = score_temp

        print("The best cut for the parent is", cut_str)
        print()
        print()
        
        return n, best_score, cut_str, cut, cut_2


    def split(
        self,
        n_bins_wanted,
        granularity=10,
        stat_limit=0.025
    ):
        possible_edges = {
            key:np.linspace(ak.min(self.data[key]), ak.max(self.data[key]), granularity + 2)[1:-1] 
            for key in self.observables
        }
        
        self.decision_tree = Split_Node(ak.Array([True]), "ROOT")
        n_bins = 1
        pbar = tqdm.tqdm(
            total=n_bins_wanted, desc="Splitting Bins", 
            leave=True, initial=0, colour="green"
        )
        while n_bins < n_bins_wanted:    
            max_score = -1
            best_parent = None
            iteration_cut_as_str = None
            iteration_mask_1 = None
            iteration_mask_2 = None
            
            with multiprocessing.Pool() as pool:
                starmap_input = []
                leaves = self.decision_tree.get_leaves()
                leaf_key = {i:node for i, node in enumerate(leaves)}
                for i, possible_parent in leaf_key.items():
                    starmap_input += [
                        (i, possible_parent, list(possible_edges.items()), stat_limit)
                    ]
                
                for n, score_temp, cut_str, cut, cut_2 in pool.starmap_async(
                    self.check_split, starmap_input
                ).get():
                    if score_temp > max_score:
                        iteration_cut_as_str = cut_str
                        iteration_mask_1 = cut
                        iteration_mask_2 = cut_2
                        best_parent = leaf_key[n]
                        max_score = score_temp

            if max_score < 0:
                print(f"Prematurely finished at {n_bins}! No new bins found!")
                return

            iteration_cut_as_str = list(iteration_cut_as_str)
            # print("Picking", iteration_cut_as_str)
            
            print("THE TWO BEST CUTS APPLIED:", iteration_cut_as_str)
            print()
            print()
            Split_Node(iteration_cut_as_str[0], best_parent)
            Split_Node(iteration_cut_as_str[1], best_parent)
            n_bins += 1
            pbar.update(1)
    
    def get_histogram(self, total_integral=1):
        leaves = self.decision_tree.get_leaves()
        bin_counts = np.zeros((len(leaves), self.n_hypotheses))
        for n, cut_up_bin in enumerate(leaves):
            mask = eval(cut_up_bin.apply_all_cuts("self.data"))

            bin_counts[n] = ak.sum(self.data[mask].w, axis=0)/self.total*total_integral
        return bin_counts, np.arange(bin_counts.shape[0] + 1)


