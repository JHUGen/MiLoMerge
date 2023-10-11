import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

def giga_ROC(sample1, sample2):
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)
    
    hypo1_counts = sample1.copy()/np.abs(sample1).sum()
    hypo2_counts = sample2.copy()/np.abs(sample2).sum()
    
    division_terms = hypo2_counts/hypo1_counts
    
    division_terms[~np.isfinite(division_terms)] = 0
    
    ratios = np.array(
        sorted(
            list(enumerate(division_terms)), key=lambda x: x[1]
        )
    )
    
    
    ratio_values =  ratios[:,1].astype(int) #gets the values only for the ordered ratio pairs
    indices = ratios[:,0].astype(int) #gets the bin indices only for the ordered ratio pairs
    
    
    length = len(indices)
    
    
    PAC = np.zeros(length + 1) #"positive" above cutoff
    PAC_numerator = np.zeros(length + 1)
    PBC = np.zeros(length + 1) #"positive" below cutoff
    NAC = np.zeros(length + 1) #"negative" above cutoff
    NAC_numerator = np.zeros(length + 1)
    NBC = np.zeros(length + 1) #"negative" below cutoff
    
    first_positive = np.searchsorted(ratio_values, 0, side='left')
    # print("div terms:", division_terms)
    # negative_terms = indices[:first_positive]
    # positive_terms = indices[first_positive:]
    
    # print(negative_terms)
    
    # print(ratios)
    
    for n in range(length + 1):
        above_cutoff = indices[n:]
        below_cutoff = indices[:n]
        
        PAC[n] = np.abs(hypo1_counts[above_cutoff]).sum() #gets the indices listed
        PAC_numerator[n] = hypo1_counts[below_cutoff].sum()
        PBC[n] = np.abs(hypo1_counts[below_cutoff]).sum()
        
        NAC[n] = np.abs(hypo2_counts[above_cutoff]).sum()
        NAC_numerator[n] = hypo2_counts[below_cutoff].sum()
        NBC[n] = np.abs(hypo2_counts[below_cutoff]).sum()
    
    
    TPR = PAC_numerator/(PAC + PBC) #vectorized calculation
    FPR = NAC_numerator/(NAC + NBC)
    
    
    
    dy_dx = np.diff(FPR/TPR)
    dy_dx[~np.isfinite(dy_dx)] = 1
    
    signs = np.sign(dy_dx)
    # print("SIGN:", signs)
    turning_point = np.argmax(signs[1:]*signs[:1] < 0)
    
    
    negative_gradient = TPR[:turning_point], FPR[:turning_point]
    positive_gradient = TPR[turning_point:], FPR[turning_point:]
    
    apex = np.argmin(TPR)
    apex_x = TPR[apex]
    apex_y = FPR[apex]
    
    print("APEX:", apex_x, apex_y)
    
    negative_gradient = negative_gradient[0] - apex_x, negative_gradient[1] + apex_y
    
    positive_gradient = positive_gradient[0] - apex_x, positive_gradient[1] - apex_y
    
    
    plt.plot(np.linspace(-np.pi, np.pi, 50), hypo1_counts)
    plt.plot(np.linspace(-np.pi, np.pi, 50), hypo2_counts)
    plt.show()
    
    
    plt.plot(TPR[np.isfinite(TPR) & np.isfinite(FPR)], FPR[np.isfinite(TPR) & np.isfinite(FPR)])
    # print(TPR[np.isfinite(TPR) & np.isfinite(FPR)])
    # print(FPR[np.isfinite(TPR) & np.isfinite(FPR)])
    plt.show()
    
    
    new_TPR = np.concatenate( (positive_gradient[0], negative_gradient[0]) )
    new_FPR = np.concatenate( (positive_gradient[1], negative_gradient[1]) )
    plt.plot(new_TPR, new_FPR)
    BBB = np.max(-TPR), np.max(FPR)
    rect = matplotlib.patches.Rectangle((0,0),*BBB )
    rect.fill = False
    plt.gca().add_patch(rect)

    plt.show()
    

    # print(indices)


def positive_ROC(sample1, sample2):
    """This function produces a ROC curve from two histograms

    Parameters
    ----------
    sample1 : numpy.ndarray
        The first set of histogram counts. This is your "True" data
    sample2 : numpy.ndarray
        The second set of histogram counts. This if your "False" data
    

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray, float)
        returns the true rate, the false rate, and the area under the curve (assuming true rate is the x value)
    """
    
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)
    
    hypo1_counts = sample1.copy()/sample1.sum()
    hypo2_counts = sample2.copy()/sample2.sum()
    
    ratios = sorted(
        list(enumerate(hypo1_counts/hypo2_counts)), key=lambda x: x[1], reverse=True
    )
    
    ratios = np.array(ratios)[:,0].astype(int) #gets the bin indices only for the ordered ratio pairs
    ratios[~np.isfinite(ratios)] = 0
    length = len(ratios) + 1
    
    PAC = np.zeros(length) #"positive" above cutoff
    PBC = np.zeros(length) #"positive" below cutoff
    NAC = np.zeros(length) #"negative" above cutoff
    NBC = np.zeros(length) #"negative" below cutoff
    
    
    for n in range(length):
        above_cutoff = ratios[n:]
        below_cutoff = ratios[:n]
        
        PAC[n] = hypo1_counts[above_cutoff].sum() #gets the indices listed
        PBC[n] = hypo1_counts[below_cutoff].sum()
        
        NAC[n] = hypo2_counts[above_cutoff].sum()
        NBC[n] = hypo2_counts[below_cutoff].sum()
        
    TPR = PAC/(PAC + PBC) #vectorized calculation
    FPR = NAC/(NAC + NBC)
    
    TPR[~np.isfinite(TPR)] = 0
    FPR[~np.isfinite(FPR)] = 0
    
    print("AREA TEST:", np.trapz(FPR, TPR), np.trapz(TPR, FPR), np.trapz(1 - FPR, 1 - TPR), np.trapz(1 - TPR, 1 - FPR))
    
    return TPR, FPR, np.abs(np.trapz(FPR, TPR))

def negative_ROC(sample1, sample2):
    """This function produces a ROC curve from two histograms

    Parameters
    ----------
    sample1 : numpy.ndarray
        The first set of histogram counts. This is your "True" data
    sample2 : numpy.ndarray
        The second set of histogram counts. This if your "False" data
    

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray, float)
        returns the true rate, the false rate, and the area under the curve (assuming true rate is the x value)
    """
    
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)
    
    hypo1_counts = sample1.copy()/sample1.sum()
    hypo2_counts = sample2.copy()/sample2.sum()
    
    ratios = sorted(
        list(enumerate(hypo1_counts/hypo2_counts)), key=lambda x: x[1], reverse=True
    )
    
    ratios = np.array(ratios)[:,0].astype(int) #gets the bin indices only for the ordered ratio pairs
    ratios[~np.isfinite(ratios)] = 0
    length = len(ratios) + 1
    
    PAC = np.zeros(length) #"positive" above cutoff
    PBC = np.zeros(length) #"positive" below cutoff
    NAC = np.zeros(length) #"negative" above cutoff
    NBC = np.zeros(length) #"negative" below cutoff
    
    
    for n in range(length):
        above_cutoff = ratios[n:]
        below_cutoff = ratios[:n]
        
        PAC[n] = hypo1_counts[above_cutoff].sum() #gets the indices listed
        PBC[n] = hypo1_counts[below_cutoff].sum()
        
        NAC[n] = hypo2_counts[above_cutoff].sum()
        NBC[n] = hypo2_counts[below_cutoff].sum()
        
    TPR = PAC/(PAC + PBC) #vectorized calculation
    FPR = NAC/(NAC + NBC)
    
    TPR[~np.isfinite(TPR)] = 0
    FPR[~np.isfinite(FPR)] = 0
    
    return TPR, FPR, np.abs(np.trapz(FPR, TPR))