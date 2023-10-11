import numpy as np

def print_msg_box(msg, indent=1, width=0, title=""):
    """returns message-box with optional title.
    Ripped from https://stackoverflow.com/questions/39969064/how-to-print-a-message-box-in-python
    
    Parameters
    ----------
    msg : str
        The message to use
    indent : int, optional
        indent size, by default 1
    width : int, optional
        box width, by default 0
    title : str, optional
        box title, by default ""
    """
    
    lines = msg.split('\n')
    space = " " * indent
    if not width:
        width = max(map(len, lines))
    box = f'╔{"═" * (width + indent * 2)}╗\n'  # upper_border
    if title:
        box += f'║{space}{title:<{width}}{space}║\n'  # title
        box += f'║{space}{"-" * len(title):<{width}}{space}║\n'  # underscore
    box += ''.join([f'║{space}{line:<{width}}{space}║\n' for line in lines])
    box += f'╚{"═" * (width + indent * 2)}╝'  # lower_border
    return box

def merge_bins(target, bins, *counts, **kwargs):
    """Merges a set of bins that are given based off of the counts provided
    Eliminates any bin with a corresponding count that is less than the target
    Useful to do merge_bins(\*np.histogram(data), ...)
    
    
    Parameters
    ----------
    counts : numpy.ndarray
        The counts of a histogram
    bins : numpy.ndarray
        The bins of a histogram
    target : int, optional
        The target value to achieve - any counts below this will be merged, by default 0
    ab_val : bool, optional
        If on, the target will consider the absolute value of the counts, not the actual value, by default True
    drop_first : bool, optional
        If on, the function will not automatically include the first bin edge, by default False

    Returns
    -------
    Tuple(numpy.ndarray, numpy.ndarray)
        A np.histogram object with the bins and counts merged

    Raises
    ------
    ValueError
        If the bins and counts are not sized properly the function will fail
    """
    
    drop_first = kwargs.get('drop_first',False)
    ab_val = kwargs.get('ab_val', True)
    
    new_counts = []
    [new_counts.append([]) for _ in counts]
    
    counts = np.vstack(counts)
    
    if any([len(bins) != len(count) + 1 for count in counts]):
        errortext = "Length of bins is {:.0f}, lengths of counts are ".format(len(bins))
        errortext += " ".join([str(len(count)) for count in counts])
        errortext += "\nlen(bins) should be len(counts) + 1!"
        raise ValueError("\n" + errortext)
    
    
    if not drop_first:
        new_bins = [bins[0]] #the first bin edge is included automatically if not explicitly stated otherwise
    else:
        new_bins = []
    
    if ab_val:
        counts = np.abs(counts)
    
    
    i = 0
    while i < len(counts[0]):
        summation = np.zeros(len(counts))
        start = i
        # print("starting iteration at:", i)
        # print("Current running sum is ( {:.3f}, {:.3f} )".format(np.sum(new_counts[0]), np.sum(new_counts[1])))
        # print("Running sum should be ( {:.3f}, {:.3f} )".format(*np.sum(counts[:,:i], axis=1)))
        while np.any(summation <= target) and (i < len(counts[0])):
            summation += counts[:,i]
            i += 1
        # print("Merged counts", start, "through", i-1)
        
        if drop_first and len(new_bins) == 0:
            first_bin = max(i - 1, 0)
            new_bins += [bins[first_bin]]
            
        if not( np.any(summation <= target) and (i == len(counts[0])) ):
            for k in range(len(counts)):
                new_counts[k] += [np.sum(counts[k][start:i])]
            new_bins += [bins[i]]
        else:
            for k in range(len(counts)):
                new_counts[k][-1] += np.sum(counts[k][start:i])
            new_bins[-1] = bins[i]
        # print("Current running sum is ( {:.3f}, {:.3f} )".format(np.sum(new_counts[0]), np.sum(new_counts[1])))
        # print("Running sum should be ( {:.3f}, {:.3f} )".format(*np.sum(counts[:,:i], axis=1)))
        # print()
        # print()
    return np.vstack(new_counts), np.array(new_bins)

class Grim_Brunelle_merger(object):#Professor Nathan Brunelle!
    #https://engineering.virginia.edu/faculty/nathan-brunelle
    def __init__(self, bins, *counts, stats_check=True, subtraction_metric=True, weights=None) -> None:
        """This class is the base object for bin merging. It merges bins locally (by outting adjacent bins together).
        Please only input 1 dimensional histograms! Should you need to operate upon a multidimensional histogram - please unroll them first.

        Parameters
        ----------
        bins : numpy.ndarray
            These are your bin edges
        counts : numpy.ndarray
            These are a set of bin counts put in as args. Place an unlimited number of them - each one corresponds to a different hypothesis. 
            Both bins and counts can be created using numpy.histogram()
        stats_check : bool, optional
            If you want to insert a statistics check enable this option, by default True
        subtraction_metric : bool, optional
            If you want to use the metric that does subtraction. If false, it uses version of the metric that uses division, by default True
        weights : numpy.ndarray, optional
            A list of weights of the same length as the number of counts put in for each hypothesis. If None all weights are 1, by default None

        Raises
        ------
        ValueError
            If your bin counts are not all the same length, raise an error
        ValueError
            If len(counts) != len(bins) - 1 then raise an error
        ValueError
            If len(weights) != len(counts) then raise an error
        """
        
        it = iter(counts)
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            errortext = [str(len(count)) for count in counts]
            errortext = " ".join(errortext)
            errortext = 'Not all counts have same length! Lengths are: ' + errortext
            errortext = print_msg_box(errortext, title="ERROR")
            raise ValueError('\n'+errortext)
        
        if len(counts[0]) != len(bins) - 1:
            errortext = "Invalid lengths! {:.0f} != {:.0f} + 1".format(len(counts[0]), len(bins))
            errortext = print_msg_box(errortext, title="ERROR")
            raise ValueError('\n'+errortext)

        if weights != None and len(weights) != len(counts):
            errortext = "If using weights, the number of weight values and the number of hypotheses should be the same!"
            errortext += '\nCurrently there are {:.0f} hypotheses and {:.0f} weight value(s)'.format(len(counts), len(weights))
            errortext = print_msg_box(errortext)
            raise ValueError('\n'+errortext)
        
        if weights == None:
            weights = np.ones(len(counts), dtype=float)
        else:
            weights = np.array(weights)
        
        self.weights = np.outer(weights, weights) #generates a matrix from the outer product of two vectors

        self.n = len(counts)
        self.subtraction_metric=subtraction_metric

        self.original_bins = bins.copy()
        self.original_counts = np.vstack(counts)
        self.original_counts = self.original_counts.T
        self.original_counts /= np.abs(self.original_counts).sum(axis=0)
        self.original_counts = self.original_counts.T
        
        stats_for_mean = np.concatenate(counts)
        
        if not stats_check:
            self.merged_counts, self.post_stats_merge_bins = self.original_counts.copy(), self.original_bins.copy()
        else:
            self.merged_counts, self.post_stats_merge_bins = merge_bins(0.05*np.mean( stats_for_mean ), 
                                            bins, 
                                            *self.original_counts.copy()
                                            )

        self.n_items = len(self.merged_counts[0])
        
        self.local_edges = self.post_stats_merge_bins.copy()
        
        self.counts_to_merge = self.merged_counts.copy()
        self.counts_to_merge = self.merged_counts.astype(float)
    
    def reset(self):
        """Resets the state
        """
        self.counts_to_merge = self.merged_counts.copy()
        self.local_edges = self.post_stats_merge_bins.copy()
        self.n_items = len(self.merged_counts[0])
    
    def __MLM__(self, b, bP):
        """The distance function MLM

        Parameters
        ----------
        b : int
            index 1 for the counts
        bP : int
            index 2 for the counts

        Returns
        ------
        float
            The distance metric between the two indices
        """
        
        counts = self.counts_to_merge.copy()
        numerator = 0
        denomenator = 0
        
        for h in range(self.n):
            for hP in range(h+1, self.n):
                mat = np.array([
                    [ counts[h][b], counts[h][bP] ],
                    [ counts[hP][b], counts[hP][bP] ]
                ], dtype=float)
                
                mat *= self.weights[h][hP]
                # print("tests")
                # print(mat)
                # print(counts[h][b], counts[h][bP], counts[hP][b], counts[hP][bP])
                # print(2*np.prod(mat), 2*np.prod([counts[h][b], counts[h][bP], counts[hP][b], counts[hP][bP]]))
                # print()
                if self.subtraction_metric:
                    numerator += (mat[0][0]*mat[1][1])**2 + (mat[0][1]*mat[1][0])**2 - 2*np.prod(mat)
                else:
                    numerator += (mat[0][0]*mat[1][1])**2 + (mat[0][1]*mat[1][0])**2
                    denomenator += np.prod(mat)
                    
        if self.subtraction_metric:
            return numerator#/(2*denomenator)
        else:
            return numerator/(2*denomenator)
    
    def __merge__(self, i, j):
        """Merges bins together by bin count index

        Parameters
        ----------
        i : int
            index i
        j : int
            index j

        Returns
        -------
        Tuple([numpy.ndarray, numpy.ndarray])
            Returns a numpy histogram tuple of locally merged bins

        Raises
        ------
        ValueError
            Raises a ValueError if the edges are merged upon
        ValueError
            Raises an error if you try to merge non-adjacent bins
        """
        temp_counts = self.counts_to_merge.copy()
        temp_edges = self.local_edges.copy()
        
        if i == j + 1: #merges count i with the count behind it
            if i > self.n_items - 1 or i < 0:
                raise ValueError("YOU IDIOT WHY WOULD YOU MERGE ON THE EDGES THAT LOSES THE RANGE\nTrying to merge {:.0f} with {:.0f}".format(i,j))
            temp_counts = np.concatenate( (self.counts_to_merge[:,:j], np.array([self.counts_to_merge[:,j] + self.counts_to_merge[:,i]]).T, self.counts_to_merge[:,i+1:]), axis=1 )
            temp_edges = np.concatenate( (self.local_edges[:i], self.local_edges[i+1:]) )
        elif i == j - 1: #merges count i with the count in front of it
            if i > self.n_items - 1 or i < 1:
                raise ValueError("YOU IDIOT WHY WOULD YOU MERGE ON THE EDGES THAT LOSES THE RANGE\nTrying to merge {:.0f} with {:.0f}".format(i,j))  
            temp_counts = np.concatenate( (self.counts_to_merge[:,:i], np.array([self.counts_to_merge[:,i] + self.counts_to_merge[:,j]]).T, self.counts_to_merge[:,j+1:]), axis=1 )
            temp_edges = np.concatenate( (self.local_edges[:i+1], self.local_edges[i+2:]) )
        elif i == j: #does nothing
            pass
        else:
            raise ValueError("This is local binning! Can only merge ahead or behind!")
        
        return (temp_counts, temp_edges)
    
    def run(self, target_bin_number):
        """runs the local bin merging

        Parameters
        ----------
        target_bin_number : int
            the number of bins you want

        Returns
        -------
        Tuple(numpy.ndarray, numpy.ndarray)
            a numpy histogram of the new counts and bins
        """
        
        while self.n_items > target_bin_number:
            combinations = {}
            scores = {}
            
            temp_counts = np.zeros(shape=(self.counts_to_merge.shape[0], self.counts_to_merge.shape[1] - 1), dtype=float)
            
            for i in range(1, self.n_items - 1): #don't merge edge bins/counts!
                score = self.__MLM__(i, i+1)
                combinations[i] = (i, i+1)
                scores[i] = score
            
            score = self.__MLM__(1, 0) #try merging counts 1 with count 0 (backwards)
            combinations[0] = (1, 0)
            scores[0] = score
            i1, i2 = combinations[ min(scores, key=scores.get) ]
            # print(scores)
            # print(combinations, '\n')
            # for k in range(self.n):
            temp_counts, temp_bins = self.__merge__(i1, i2)
            
            self.counts_to_merge, self.local_edges = temp_counts, temp_bins
            
            self.n_items -= 1
            
        return self.counts_to_merge, self.local_edges
    
    def run_local_faster(self, target_bin_number):
        """attempts to run faster - needs to be debugged. WIP

        Parameters
        ----------
        target_bin_number : int
            the number of bins you want

        Returns
        -------
        Tuple(numpy.ndarray, numpy.ndarray)
            a numpy histogram of the new counts and bins
        """
        
        things_to_recalculate = set(list(range(1, self.n_items - 1)))
        combinations = {}
        scores = {}
        while self.n_items > target_bin_number:
            # print(things_to_recalculate)
            # print(scores)
            # print(combinations)
            
            temp_counts = np.zeros(shape=(self.counts_to_merge.shape[0], self.counts_to_merge.shape[1] - 1), dtype=float)
            for i in things_to_recalculate: #don't merge edge bins/counts!
                if i == 1:
                    score = self.__MLM__(1, 0) #try merging counts 1 with count 0 (backwards)
                    combinations[0] = (1, 0)
                    scores[0] = score
                score = self.__MLM__(i, i+1)
                combinations[i] = (i, i+1)
                scores[i] = score
            
            item_to_delete = min(scores, key=scores.get)
            i1, i2 = combinations[item_to_delete]
            
            # print(i1, i2, self.n_items, '\n')
            if i1 == self.n_items - 2:
                things_to_recalculate = set([i-1])
            else:
                things_to_recalculate = set((i1-1, i1))
            # print(scores)
            # print(combinations)
            for index in scores.keys():
                if index > i1:
                    scores[index-1] = scores[index]
                    scores[index] = np.inf
                    
                    old_i1, old_i2 = combinations[index]
                    combinations[index-1] = (old_i1 - 1, old_i2 - 1)
                    combinations[index] = (np.nan, np.nan)
            
            temp_counts, temp_bins = self.__merge__(i1, i2)
            
            self.counts_to_merge, self.local_edges = temp_counts, temp_bins
            
            self.n_items -= 1
            
        return self.counts_to_merge, self.local_edges


class Grim_Brunelle_with_standard_model(Grim_Brunelle_merger):
    def __init__(self, bins, *counts, stats_check=True, subtraction_metric=True, weights=None) -> None:
        """This class  merges bins locally (by outting adjacent bins together) by comparing each item to a "Standard Model" sample.
        The first set of counts that are input will be treated as the standard model sample.
        Please only input 1 dimensional histograms! Should you need to operate upon a multidimensional histogram - please unroll them first.

        Parameters
        ----------
        bins : numpy.ndarray
            These are your bin edges
        counts : numpy.ndarray
            These are a set of bin counts put in as args. Place an unlimited number of them -  each one corresponds to a different hypothesis. 
            The first array that is input will be the "Standard Model" sample.
            Both bins and counts can be created using numpy.histogram()
        stats_check : bool, optional
            If you want to insert a statistics check enable this option, by default True
        subtraction_metric : bool, optional
            If you want to use the metric that does subtraction. If false, it uses version of the metric that uses division, by default True
        weights : numpy.ndarray, optional
            A list of weights of the same length as the number of counts put in for each hypothesis. If None all weights are 1, by default None

        Raises
        ------
        ValueError
            If your bin counts are not all the same length, raise an error
        ValueError
            If len(counts) != len(bins) - 1 then raise an error
        ValueError
            If len(weights) != len(counts) then raise an error
        """
        super().__init__(bins, *counts, stats_check=stats_check, subtraction_metric=subtraction_metric, weights=weights)
        
        self.weights = np.array(weights)
        
        
    def __MLM__(self, b, bP):
        """The distance function MLM - edited to perform only comparisons with the "Standard Model" sample.

        Parameters
        ----------
        b : int
            index 1 for the counts
        bP : int
            index 2 for the counts

        Returns
        ------
        float
            The distance metric between the two indices
        """
        
        counts = self.counts_to_merge.copy()
        numerator = 0
        denomenator = 0
        
        h = 0
        for hP in range(1, self.n):
            mat = np.array([
                [ counts[h][b], counts[h][bP] ],
                [ counts[hP][b], counts[hP][bP] ]
            ], dtype=float)
            
            mat *= self.weights[hP]
            # print("tests")
            # print(mat)
            # print(counts[h][b], counts[h][bP], counts[hP][b], counts[hP][bP])
            # print(2*np.prod(mat), 2*np.prod([counts[h][b], counts[h][bP], counts[hP][b], counts[hP][bP]]))
            # print()
            if self.subtraction_metric:
                numerator += (mat[0][0]*mat[1][1])**2 + (mat[0][1]*mat[1][0])**2 - 2*np.prod(mat)
            else:
                numerator += (mat[0][0]*mat[1][1])**2 + (mat[0][1]*mat[1][0])**2
                denomenator += np.prod(mat)
                    
        if self.subtraction_metric:
            return numerator#/(2*denomenator)
        else:
            return numerator/(2*denomenator)


class Grim_Brunelle_nonlocal(Grim_Brunelle_merger):
    def __init__(self, bins, *counts, stats_check=True, subtraction_metric=True, weights=None) -> None:
        """This class is the base object for bin merging. 
        It merges bins nonlocally - so the bins no longer contain any physical meaning afterwards.
        Please only input 1 dimensional histograms! Should you need to operate upon a multidimensional histogram - please unroll them first.

        Parameters
        ----------
        bins : numpy.ndarray
            These are your bin edges
        counts : numpy.ndarray
            These are a set of bin counts put in as args. Place an unlimited number of them - each one corresponds to a different hypothesis. 
            Both bins and counts can be created using numpy.histogram()
        stats_check : bool, optional
            If you want to insert a statistics check enable this option, by default True
        subtraction_metric : bool, optional
            If you want to use the metric that does subtraction. If false, it uses version of the metric that uses division, by default True
        weights : numpy.ndarray, optional
            A list of weights of the same length as the number of counts put in for each hypothesis. If None all weights are 1, by default None

        Raises
        ------
        ValueError
            If your bin counts are not all the same length, raise an error
        ValueError
            If len(counts) != len(bins) - 1 then raise an error
        ValueError
            If len(weights) != len(counts) then raise an error
        """
        super().__init__(bins, *counts, stats_check=stats_check, subtraction_metric=subtraction_metric, weights=weights)
    
        self.tracker = [ {} ]
        for i in range(self.n_items):
            self.tracker[0][i] = (i, i) #record where it started, and where it ended
    
    def __trace__(self, i):
        """WIP. Trying to trace the placement of where bins go

        Parameters
        ----------
        i : int
            index i

        Returns
        -------
        int
            what index i used to be
        """
        # print(self.tracker)
        og_i = i
        while self.tracker[-1][i][1] != i:
            # print("Trying to match", i, "with", self.tracker[i])
            i = self.tracker[-1][i][1]
        print(og_i, "points to", i)
        print('\n')
        return i
    
    def __merge__(self, i, j, track=False):
        """Merges bins together by bin count index

        Parameters
        ----------
        i : int
            index i
        j : int
            index j
        track : bool, optional
            Whether you would like to track the placement of where the original bins go in the nonlocal case, by default False

        Returns
        -------
        Tuple([numpy.ndarray, numpy.ndarray])
            Returns a numpy histogram tuple of nonlocally merged bin edges and their counts. The bin edges are nonphysical, so are just a range of indices equally spaced.
        """
        merged_counts = np.zeros(shape=(self.n, self.n_items - 1), dtype=float)
        k = 0
        for n in range(self.n_items):
            if n != i and n != j:
                if track:
                    self.tracker[n] = k
                merged_counts[:,k] = self.counts_to_merge[:,n]
                k += 1
        
        if track:
            self.tracker[i] = k
            self.tracker[j] = k
        
        merged_counts[:,k] = self.counts_to_merge[:,i] + self.counts_to_merge[:,j] #shove everything into the final bin
        self.n_items -= 1
        
        self.counts_to_merge = merged_counts
        
        return self.counts_to_merge

    def __closest_pair__(self):
        """A simple function to find the closest pair of points between any two bins nonlocally

        Returns
        -------
        Tuple([float, int, int])
            A tuple of values containing the smallest distance, and the indices that create that distance

        Raises
        ------
        ValueError
            If there is ever a nan returned throw an error. This is only possible when using the division version of the metric.
        """
        # print("USING INDICES:", indices, "With BRUTE_FORCE=", brute_force)
        smallest_distance = (np.inf, None, None)
        for i in range(self.n_items):
            for j in range(i):
                temp_dist = self.__MLM__(i, j)
                if temp_dist < smallest_distance[0]:
                    smallest_distance = (temp_dist, i, j)
        
        if not np.isfinite(smallest_distance[0]):
            raise ValueError("Distance function has produced nan/inf at some point with value" + str(smallest_distance[0]))
        
        return smallest_distance
    
    def run(self, target_bin_number, track=False):
        """runs the nonlocal binning

        Parameters
        ----------
        target_bin_number : int
            the number of bins you want
        track : bool, optional
            Whether you would like to track the placement of where the original bins go in the nonlocal case, by default False

        Returns
        -------
        Tuple(numpy.ndarray, numpy.ndarray)
            a numpy histogram of the new counts and bins
        """
        
        # print("RUNNING NONLOCAL", self.n_items, target_bin_number)
        while self.n_items > target_bin_number:
            distance, i, j = self.__closest_pair__()
            self.__merge__(i,j, track=track)
    
        return self.counts_to_merge, np.array(range(self.n_items+1))
    
class Grim_Brunelle_nonlocal_with_standard_model(Grim_Brunelle_nonlocal):
    def __init__(self, bins, *counts, stats_check=True, subtraction_metric=True, weights=None) -> None:
        """This class will perform nonlocal bin merging by comparing each item to a "Standard Model" sample.
        The first set of counts that are input will be treated as the standard model sample.
        It merges bins nonlocally - so the bins no longer contain any physical meaning afterwards.
        Please only input 1 dimensional histograms! Should you need to operate upon a multidimensional histogram - please unroll them first.

        Parameters
        ----------
        bins : numpy.ndarray
            These are your bin edges
        counts : numpy.ndarray
            These are a set of bin counts put in as args. Place an unlimited number of them -  each one corresponds to a different hypothesis. 
            The first array that is input will be the "Standard Model" sample.
            Both bins and counts can be created using numpy.histogram()
        stats_check : bool, optional
            If you want to insert a statistics check enable this option, by default True
        subtraction_metric : bool, optional
            If you want to use the metric that does subtraction. If false, it uses version of the metric that uses division, by default True
        weights : numpy.ndarray, optional
            A list of weights of the same length as the number of counts put in for each hypothesis. If None all weights are 1, by default None

        Raises
        ------
        ValueError
            If your bin counts are not all the same length, raise an error
        ValueError
            If len(counts) != len(bins) - 1 then raise an error
        ValueError
            If len(weights) != len(counts) then raise an error
        """
        super().__init__(bins, *counts, stats_check=stats_check, subtraction_metric=subtraction_metric, weights=weights)
        self.weights = np.array(weights)
        
        
    def __MLM__(self, b, bP):
        """The distance function MLM

        Parameters
        ----------
        b : int
            index 1 for the counts
        bP : int
            index 2 for the counts

        Returns
        ------
        float
            The distance metric between the two indices
        """
        
        counts = self.counts_to_merge.copy()
        numerator = 0
        denomenator = 0
        
        h = 0
        for hP in range(1, self.n):
            mat = np.array([
                [ counts[h][b], counts[h][bP] ],
                [ counts[hP][b], counts[hP][bP] ]
            ], dtype=float)
            
            mat *= self.weights[hP]
            # print("tests")
            # print(mat)
            # print(counts[h][b], counts[h][bP], counts[hP][b], counts[hP][bP])
            # print(2*np.prod(mat), 2*np.prod([counts[h][b], counts[h][bP], counts[hP][b], counts[hP][bP]]))
            # print()
            if self.subtraction_metric:
                numerator += (mat[0][0]*mat[1][1])**2 + (mat[0][1]*mat[1][0])**2 - 2*np.prod(mat)
            else:
                numerator += (mat[0][0]*mat[1][1])**2 + (mat[0][1]*mat[1][0])**2
                denomenator += np.prod(mat)
                    
        if self.subtraction_metric:
            return numerator#/(2*denomenator)
        else:
            return numerator/(2*denomenator)
