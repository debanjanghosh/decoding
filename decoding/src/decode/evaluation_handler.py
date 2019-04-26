import numpy as np


def evaluation(kwargs,pred_representations,\
               test_representations,\
               target_semantic_idxs):
    
    pred_representations /= np.linalg.norm(pred_representations, axis=1, keepdims=True)
    test_representations = test_representations / np.linalg.norm(test_representations, axis=1, keepdims=True)
    
    eval_all = kwargs['eval_all']
    eval_context = kwargs['eval_context']
    eval_passage = kwargs['eval_passage']
    
    results = {}
    
    
    if eval_all == 'True':
        rank_accuracy_scores_avg, rankings,rank_of_correct = calcOneVsAllEval(kwargs,test_representations,\
                         pred_representations,\
                         target_semantic_idxs)
        
        results['all_ranking'] = rankings
        results['correct_ranking'] = rank_of_correct
        results['rank_accuracy_scores'] = rank_accuracy_scores_avg
        
    
    if eval_context == 'True':
        rank_accuracy_scores_avg, rankings,rank_of_correct = calcStrongWeakContextEval(kwargs,test_representations,\
                                  pred_representations,\
                                  target_semantic_idxs)
        
        results['all_ranking'] = rankings
        results['correct_ranking'] = rank_of_correct
        results['rank_accuracy_scores_strong'] = rank_accuracy_scores_avg_strong
        results['rank_accuracy_scores_weak'] = rank_accuracy_scores_avg_weak

    if eval_passage == 'True':
        rank_accuracy_scores_avg = calcPassageEval(kwargs,test_representations,\
                                  pred_representations,\
                                  target_semantic_idxs)
        
    #    results['all_passage_ranking'] = rankings
    #    results['correct_passage_ranking'] = rank_of_correct
        results['rank_accuracy_passage_scores'] = rank_accuracy_scores_avg
        
    return results

def calcOneVsAllEval(kwargs,test_representations,\
                     pred_representations,\
                     target_semantic_idxs):

    similarities = np.dot(pred_representations,\
                        test_representations.T)
    
    rankings = np.argsort(-similarities, axis=1)
    
    matches = np.equal(rankings, target_semantic_idxs[:, np.newaxis])


    rank_of_correct = np.argmax(matches, axis=1)
    rank_of_correct = [int(rank)+1 for rank in rank_of_correct]
    rank_of_correct_mean = np.average(rank_of_correct)

    rank_accuracy_scores = [ (1.0- ((rank-1.0)/\
                            (len(pred_representations)-1.0))) for rank in rank_of_correct]
    rank_accuracy_scores_avg = np.average(rank_accuracy_scores)
  #  print('rank_accuracy_scores: ' + str(rank_accuracy_scores_avg))
   #     plt.matshow(diff_mat)
    
    '''
    plt.colorbar()
    plt.show()
    '''
    return rank_accuracy_scores_avg, rankings, rank_of_correct_mean

def calcPassageEval(kwargs,test_representations,\
                     pred_representations,\
                     target_semantic_idxs):
    
    '''we need passage ids corres. to test utterances'''
    
    testPassageIds = kwargs['passage_ids']
    
    passageIdMap = {}
    for index,id in enumerate(testPassageIds):
        indexes = passageIdMap.get(id,[])
        indexes.append(index)
        passageIdMap[id] = indexes
        
    rank_accuracy_scores_all = []
    for id in passageIdMap.keys():
        indexes = passageIdMap.get(id)
        similarities = np.dot(pred_representations[indexes],\
                        test_representations[indexes].T)
    
        rankings = np.argsort(-similarities, axis=1)
        matches = np.equal(rankings, np.arange(len(indexes))[:, np.newaxis])

        rank_of_correct = np.argmax(matches, axis=1)
        rank_of_correct = [int(rank)+1 for rank in rank_of_correct]
        rank_accuracy_scores = [ (1.0- ((rank-1.0)/\
                            (len(indexes)-1.0))) for rank in rank_of_correct]
        rank_accuracy_scores_avg = np.average(rank_accuracy_scores)
        rank_accuracy_scores_all.append(rank_accuracy_scores_avg)
        
        rank_of_correct_mean = np.average(rank_of_correct)

   #     plt.matshow(diff_mat)
    
    '''
    plt.colorbar()
    plt.show()
    '''
    rank_accuracy_scores_all_avg = np.average(rank_accuracy_scores_all)
    print('rank_accuracy_scores: ' + str(rank_accuracy_scores_all_avg))

    return rank_accuracy_scores_all_avg



def calcStrongWeakContextEval(kwargs,test_representations,\
                                  pred_representations,\
                                  target_semantic_idxs):
    
    rank_of_correct = np.argmax(matches, axis=1)
    rank_of_correct_strong_context = rank_of_correct[0:32]
    rank_of_correct_weak_context = rank_of_correct[32:]
        
    rank_accuracy_scores_strong = [ (1.0- ((rank-1.0)/\
                            (len(pred_representations)-1.0))) for rank in rank_of_correct_strong_context]

    rank_accuracy_scores_avg_strong = np.average(rank_accuracy_scores_strong)
    print('rank_accuracy_scores_strong_sum: ' + subject +': ' + str(rank_accuracy_scores_avg_strong))

    rank_accuracy_scores_weak = [ (1.0- ((rank-1.0)/\
                            (len(pred_representations)-1.0))) for rank in rank_of_correct_weak_context]

    rank_accuracy_scores_avg_weak = np.average(rank_accuracy_scores_weak)
    print('rank_accuracy_scores_weak_sum: ' + subject +': ' + str(rank_accuracy_scores_avg_weak))

    return rank_accuracy_scores_avg_strong,rank_accuracy_scores_avg_weak,rankings, rank_of_correct


