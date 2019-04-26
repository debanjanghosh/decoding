import numpy as np

import matplotlib.pyplot as plt

def create_histo(kwargs,pred_representations,\
               test_representations,\
               target_semantic_idxs):
    
    pred_representations /= np.linalg.norm(pred_representations, axis=1, keepdims=True)
    test_representations = test_representations / np.linalg.norm(test_representations, axis=1, keepdims=True)

    all_corrs = np.zeros(pred_representations.shape[1])

    for index in range(int(pred_representations.shape[1])):
            pred_reps_index = pred_representations[:,index]
            test_reps_index = test_representations[:,index]
    
            tmp = np.corrcoef(pred_reps_index,test_reps_index)
            all_corrs[index] = tmp[1][0]

  #  plt.hist(test_representations_small, normed=True, bins=1)
    plt.plot(all_corrs)
 #   plt.bar(5,test_representations_small)# normed=True, bins=1)
    plt.show()