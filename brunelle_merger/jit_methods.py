import numba as nb
import numpy as np

@nb.njit
def _MLM(n, counts, weights, b , bP, subtraction_metric, SM_version):
    numerator = 0
    denomenator = 0
    
    initial_range = np.arange(n) if SM_version else np.arange(1)
    
    for h in initial_range:
        for hP in np.arange(h+1, n):
            mat = np.array([
                [ counts[h][b], counts[h][bP] ],
                [ counts[hP][b], counts[hP][bP] ]
            ], dtype=np.float64)
            
            mat *= weights[h][hP]
            # print("tests")
            # print(mat)
            # print(counts[h][b], counts[h][bP], counts[hP][b], counts[hP][bP])
            # print(2*np.prod(mat), 2*np.prod([counts[h][b], counts[h][bP], counts[hP][b], counts[hP][bP]]))
            # print()
            if subtraction_metric:
                numerator += (mat[0][0]*mat[1][1])**2 + (mat[0][1]*mat[1][0])**2 - 2*np.prod(mat)
            else:
                numerator += (mat[0][0]*mat[1][1])**2 + (mat[0][1]*mat[1][0])**2
                denomenator += np.prod(mat)
                
    if subtraction_metric:
        return numerator
    else:
        return numerator/(2*denomenator)