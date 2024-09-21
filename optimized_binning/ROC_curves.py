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


def ROC_curve(sample1, sample2):
    """This function produces a ROC curve from an attribute like phi, cos(theta1), D_{0-}, etc.

    Parameters
    ----------
    sample1 : numpy.ndarray
        The first data sample for your attribute. This is your "True" data
    sample2 : numpy.ndarray
        The second data sample for your attribute. This if your "False" data
    

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray, float)
        returns the true rate, the false rate, and the area under the curve (assuming true rate is the x value)
    """
    
    hypo1_counts = np.array(sample1)
    hypo2_counts = np.array(sample2)
    
    sample_sum1 = sample1.sum()
    sample_sum1_sq = sample_sum1**2
    sample_sum2 = sample2.sum()
    sample_sum2_sq = sample_sum2**2
    
    Tx_Ty = sample_sum1*sample_sum2
    TxSq_Ty = sample_sum1_sq*sample_sum2
    Tx_TySq = sample_sum1*sample_sum2_sq
    
    ratios = sorted(
        list(enumerate(hypo2_counts/hypo1_counts)), key=lambda x: x[1], reverse=True
    )
    # print(ratios)
    
    ratios = np.array(ratios)[:,0].astype(int) #gets the bin indices only for the ordered ratio pairs
    ratios = ratios[np.isfinite(ratios)]
    # print(ratios)
    # print()
    length = len(ratios) + 1
    
    PAC = np.zeros(length) #"positive" above cutoff
    PBC = np.zeros(length) #"positive" below cutoff
    NAC = np.zeros(length) #"negative" above cutoff
    NBC = np.zeros(length) #"negative" below cutoff
    
    err = 0
    for i in range(length):
        above_cutoff = ratios[i:]
        below_cutoff = ratios[:i]
        
        PAC[i] = hypo1_counts[above_cutoff].sum() #gets the indices listed
        PBC[i] = hypo1_counts[below_cutoff].sum()
        
        NAC[i] = hypo2_counts[above_cutoff].sum()
        NBC[i] = hypo2_counts[below_cutoff].sum()
        
        if i > 0:
            accum_2 = hypo2_counts[below_cutoff].sum()
            alpha_j = (hypo1_counts*accum_2).sum()
            err += hypo1_counts[ratios[i - 1]]*(accum_2/Tx_Ty - alpha_j/TxSq_Ty)**2 + hypo2_counts[ratios[i - 1]]*(sample_sum1/Tx_Ty - alpha_j/Tx_TySq)**2
        # else:
        #     S = hypo1_counts[ratios[n]]*(hypo2_counts[below_cutoff].sum())**2

        # for bin_index in above_cutoff: #The above lines are the same as this commented code but vectorized
        #     PAC += g1_phi_counts[bin_index]
        #     NAC += g4_phi_counts[bin_index]
        
        # for bin_index in below_cutoff:
        #     PBC += g1_phi_counts[bin_index]
        #     NBC += g4_phi_counts[bin_index]
        # TPR.append(1 - PAC/(PAC + PBC))
        # FPR.append(1 - NAC/(NAC + NBC))
    
    PAC, PBC = PAC/sample_sum1, PBC/sample_sum1
    NAC, NBC = NAC/sample_sum2, NBC/sample_sum2
    
    TPR = PAC/(PAC + PBC) #vectorized calculation
    FPR = NAC/(NAC + NBC)
    
    
    return TPR, FPR, np.abs(np.trapz(FPR, TPR)), err

@nb.njit(nb.types.Tuple((nb.float64, nb.float64))(nb.float64[:], nb.float64[:]), fastmath=True, cache=True, parallel=True, nogil=True)
def ROC_score_with_error(hypo_1, hypo_2):
    hypo_1 = hypo_1.copy()/hypo_1.sum()
    hypo_2 = hypo_2.copy()/hypo_2.sum()

    raw_ratio = hypo_1/hypo_2
    ratio_indices = np.argsort(raw_ratio)

    length = len(ratio_indices)

    TPR = np.zeros(length)
    FPR = np.zeros(length)

    for n in nb.prange(length):
        above_cutoff = ratio_indices[n+1:]
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
    
    val = -np.trapz(TPR, FPR)

    
    print("Building fake data...")
    
    hypo_1 *= 10000
    hypo_1 = hypo_1.astype(np.int64)
    hypo_2 *= 10000
    hypo_2 = hypo_2.astype(np.int64)
    
    fake_data_1 = np.empty(hypo_1.sum())
    k1 = 0
    fake_data_2 = np.empty(hypo_2.sum())
    k2 = 0
    bins = np.array([i for i in range(0, len(hypo_1) + 1)], dtype=np.uint64)
    for n, (count_1, count_2) in enumerate(zip(hypo_1, hypo_2)):
        next_1 = k1 + count_1
        fake_data_1[k1:next_1] = np.full(count_1, n+1)
        k1 = next_1

        next_2 = k2 + count_2
        fake_data_2[k2:next_2] = np.full(count_2, n+1)
        k2 = next_2

    print("starting error calculation...")
    accumulator = np.empty(2000, dtype=np.float64)
    for i in nb.prange(2000):
        G_prime = np.random.choice(fake_data_1, size=len(fake_data_1))
        I_prime = np.random.choice(fake_data_2, size=len(fake_data_2))
        
        G_prime_counts, _ = np.histogram(G_prime, bins)
        I_prime_counts, _ = np.histogram(I_prime, bins)
        accumulator[i] = ROC_score(G_prime_counts.astype(np.float64), I_prime_counts.astype(np.float64))
    err = np.std(accumulator)
        

    return val, err

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
