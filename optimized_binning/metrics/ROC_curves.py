import numpy as np
import numba as nb

@nb.njit(nb.float64(nb.float64[:], nb.float64[:]), fastmath=True, cache=True)
def ROC_score(hypo_1, hypo_2):
    hypo_1 = hypo_1.copy()/hypo_1.sum()
    hypo_2 = hypo_2.copy()/hypo_2.sum()

    raw_ratio = hypo_1.copy()/hypo_2
    ratio_indices = np.argsort(raw_ratio)

    length = len(ratio_indices) + 1

    TPR = np.zeros(length)
    FPR = np.zeros(length)

    for n in nb.prange(length):
        above_cutoff = ratio_indices[n:]
        below_cutoff = ratio_indices[:n]

        TPR[n] = hypo_1[above_cutoff].sum()/(
            hypo_1[above_cutoff].sum() + hypo_1[below_cutoff].sum()
            ) #gets the indices listed

        FPR[n] = hypo_2[below_cutoff].sum()/(
            hypo_2[above_cutoff].sum() + hypo_2[below_cutoff].sum()
            )

    return np.trapz(TPR, FPR)

@nb.njit(fastmath=True, cache=True)
def ROC_curve(sample1, sample2):
    ratios = np.argsort(sample2/sample1)[::-1]
    
    PAC = np.cumsum(sample1[ratios])
    PBC = sample1.sum() - PAC
    NAC = np.cumsum(sample2[ratios])
    NBC = sample2.sum() - NAC

    TPR = PAC/(PAC + PBC) #vectorized calculation
    FPR = NAC/(NAC + NBC)


    return TPR, FPR, np.abs(np.trapz(FPR, TPR))

@nb.njit(nb.types.Tuple((nb.float64, nb.float64))(nb.float64[:], nb.float64[:], nb.float64, nb.float64), fastmath=True, cache=True, parallel=True, nogil=True)
def ROC_score_with_error_analytic(hypo_1, hypo_2, integral_1, integral_2):
    hypo_1 = hypo_1.copy()/hypo_1.sum()
    hypo_2 = hypo_2.copy()/hypo_2.sum()

    raw_ratio = hypo_1/hypo_2
    ratio_indices = np.argsort(raw_ratio)

    length = len(ratio_indices)

    TPR = np.zeros(length)
    FPR = np.zeros(length)
    BGGI = 0
    BIIG = 0

    for n in nb.prange(length):
        above_cutoff = ratio_indices[n+1:]
        cutoff = ratio_indices[n]
        below_cutoff = ratio_indices[:n]

        PAC = hypo_1[above_cutoff].sum()
        PBC = hypo_1[below_cutoff].sum()
        TPR[n] = PAC/(
            PAC + PBC
            ) #gets the indices listed

        NAC = hypo_2[above_cutoff].sum()
        NBC = hypo_2[below_cutoff].sum()
        FPR[n] = NAC/(
            NAC + NBC
            )

        BGGI += hypo_2[cutoff]*(PAC**2 + PAC*hypo_1[cutoff] + hypo_1[cutoff]**2/3)
        BIIG += hypo_1[cutoff]*(NBC**2 + NBC*hypo_2[cutoff] + hypo_2[cutoff]**2/3)

    val = -np.trapz(TPR, FPR)

    err = (1/(integral_1*integral_2))*(val*(1 - val) + (integral_1 - 1)*(BGGI - val**2) + (integral_2 - 1)*(BIIG - val**2))
    err = np.sqrt(err)


    return val, err

#nb.types.Tuple(nb.float64[:], nb.float64[:], nb.float64)(nb.float64[:], nb.float64[:])
@nb.njit(fastmath=True, cache=True)
def length_scale_ROC(sample1, sample2):
    if np.any(sample2 < 0):
        if np.any(sample1 < 0):
            raise ValueError("Need 1 positive-definite sample!")
        negative_counts = sample2
        positive_counts = sample1
    else:
        negative_counts = sample1#/np.abs(sample1).sum()
        positive_counts = sample2#/np.abs(sample2).sum()

    ratios = np.argsort(positive_counts/negative_counts)
    TPR = np.zeros(len(negative_counts) + 1)
    FPR = np.zeros(len(negative_counts) + 1)

    TPR[1:] = np.cumsum(negative_counts[ratios])
    FPR[1:] = np.cumsum(positive_counts[ratios])

    length = np.sqrt(np.diff(TPR)**2 + np.diff(FPR)**2).sum() #vectorized distance formula

    return TPR, FPR, length

