#!/usr/bin/env python
"""
Learn a decoder mapping from functional imaging data to target model
representations.
"""
from argparse import ArgumentParser
from collections import defaultdict
import itertools
import logging
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
import sklearn

from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.metrics import (classification_report,roc_auc_score)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor

import scipy.io
from tqdm import tqdm

from voxel_selection_handler import *
from text_handler import *
from image_handler import *
from embed_histo import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
fh = logging.FileHandler('./data/config/results.log','a')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

import sys

from preprocessor import *
from evaluation_handler import *


def iter_folds(trainImages, 
               trainTextInputs, n_folds=20):
  """
  Yield CV folds of a dataset.
  """
  
  '''since we are in a CV mode we are only working on 
      training data'''
  
  N = len(trainImages)
  fold_size = N // n_folds #floor 
  extra = N-n_folds*fold_size
  
  for fold_i in range(n_folds):
    if fold_i == n_folds-1:
        idxs = np.arange(fold_i * fold_size, ((fold_i + 1) * fold_size)+extra)
    else:
        idxs = np.arange(fold_i * fold_size, (fold_i + 1) * fold_size)
    
    mask = np.zeros(N, dtype=np.bool)
    mask[idxs] = True
    
    train_imaging = trainImages[~mask]
    train_semantic = trainTextInputs[1][~mask]
    test_imaging = trainImages[mask]
    test_semantic = trainTextInputs[1][mask]
    target_semantic_idxs = idxs
    
    yield (train_imaging, train_semantic), (test_imaging, test_semantic, target_semantic_idxs)



def test_fold(kwargs,subject,clf,test_images,
         target_semantic_idxs,test_representations):

    pred_representations = clf.predict(test_images)
    pred_representations /= np.linalg.norm(pred_representations, axis=1, keepdims=True)
    
    test_representations = test_representations / np.linalg.norm(test_representations, axis=1, keepdims=True)
    
    similarities = np.dot(pred_representations,\
                        test_representations.T)

    rankings = np.argsort(-similarities, axis=1)
    matches = np.equal(rankings, target_semantic_idxs[:, np.newaxis])
    rank_of_correct = np.argmax(matches, axis=1)

    return target_semantic_idxs,rankings, rank_of_correct

 
def test(clf,test_images):
    
    pred_representations = clf.predict(test_images)
    return pred_representations
    
def train_rf(train_images,train_representations):
    
    '''
        Learn a decoder mapping from sentence encodings to subject brain images.
    '''
    N1=500
    N2=1000

    rf = Pipeline([#('remove', VarianceThreshold(threshold=0.01)),
                ('select1', SelectFromModel(ExtraTreesRegressor(N1, n_jobs=20
                                                                 ))),
                ('clf', ExtraTreesRegressor(N2, n_jobs=20)),
                ]) 


    rf.fit(train_images, train_representations)
    
    return rf


def train_ridge(train_images,train_representations):
    
    '''
        Learn a decoder mapping from sentence encodings to subject brain images.
    '''
    
    ridge = RidgeCV(
      alphas=[1, 10, .01, 100, .001, 1000, .0001, 10000, .00001, 100000, .000001, 1000000, 10000000,100000000,1000000000],
      fit_intercept=False
    )
    ridge.fit(train_images, train_representations)
    
    '''pls regression did not improve result'''
 #   pls2 = PLSRegression(n_components=2)
 #   pls2.fit(train_images, train_representations)
  #  logger.info("Best alpha: %f", ridge.alpha_)
    return ridge


def run_fold(fold, encodings, permute_targets=False):
   (train_imaging, train_semantic), (test_imaging, test_semantic, target_semantic_idxs) = fold
  
   clf = learn_decoder(train_imaging, train_semantic)

   if permute_targets:
        target_semantic_idxs = target_semantic_idxs.copy()
        np.random.shuffle(target_semantic_idxs)

   rankings, rank_of_correct = eval_ranks(clf, test_imaging, target_semantic_idxs, encodings)
   return target_semantic_idxs, rankings, rank_of_correct


def processTrainTest(kwargs,data_input):
    
    test_file_name = kwargs['test_file_name']
    train_file_name = kwargs['train_file_name']
    
    subject_name = kwargs['subject_name']
    trainTextInputs = data_input['train_text']
    train_rep = trainTextInputs[1]
    testTextInputs = data_input['test_text']
    testPassageIds = [line.strip().split('\t')[0].strip() \
                      for line in testTextInputs[0]]
    
    kwargs['passage_ids'] = testPassageIds
    
    test_rep = testTextInputs[1]

    trainImage = data_input['train_image']
    testImage =  data_input['test_image']

    predicted_ranks ={}
    perf_test = []
    target_semantic_idxs =[]
    context = kwargs['eval_context']
    context = 'False'
    if context == 'True':
        '''we have a general sense of ids to test
            on context. Otherwise, we will report that'''
        target_semantic_idxs = getContextIds()
    else:
        target_semantic_idxs = np.arange(len(testImage))
        
    '''
        classsifier = train_rf(trainImage,train_rep)
        all_predictions = test(classsifier,testImage)
    '''
        
    number_dim = train_rep.shape[1]
    all_predictions = np.zeros(test_rep.shape)
    for index in range(int(number_dim)):
          #  print(index)
        train_reps_index = train_rep[:,index]
        classsifier = train_ridge(trainImage,train_reps_index)
        predictions_index = test(classsifier,testImage)
        all_predictions[:,index]=predictions_index
        
     #   create_histo(kwargs,all_predictions,
     #                                   test_rep,target_semantic_idxs)
    results = evaluation(kwargs,all_predictions,
                                        test_rep,target_semantic_idxs)
    
        
    accuracy_score = results.get('rank_accuracy_scores',None)
    if accuracy_score is not None:
            
        msg = subject_name+ '\t'+ train_file_name + '\t' + test_file_name + '\t' + 'rank_accuracy_scores:' + str(accuracy_score)
        logger.info(msg)
        
    correct_ranking = results.get('correct_ranking',None)
    if correct_ranking is not None:
        msg = subject_name+ '\t'+train_file_name + '\t' + test_file_name + '\t' +'correct_ranking_avg:' + str(correct_ranking)
        logger.info(msg)
            
    rank_accuracy_passage_scores = results.get('rank_accuracy_passage_scores',None)
    if rank_accuracy_passage_scores is not None:
            
        msg = subject_name+ '\t'+train_file_name + '\t' + test_file_name + '\t' +'rank_accuracy_passage_scores:' + str(rank_accuracy_passage_scores)
        logger.info(msg)
            

def processCrossValidation(kwargs,data_input,n_folds):
    
    subject_name = kwargs['subject_name']
    trainTextInputs = data_input['train_text']
    testTextInputs = data_input['test_text']
    full_test_reps = testTextInputs[1]

    trainImage = data_input['train_image']
    testImage =  data_input['test_image']
        
    all_folds = iter_folds(trainImage, trainTextInputs, n_folds)
    
    index = 0
    target_semantic_idxs = np.arange(384)
    #target_semantic_idxs = np.arange(len(test_reps))


    '''for each fold we train and evaluate separately'''
    perf_test = np.zeros(n_folds)
    for i, fold in enumerate(all_folds):#, n_folds, desc="%s folds" % subject_name):
        # Calculate pre    dicted ranks and MAR for this subject on this fold
        train_images,train_reps = \
        fold[0][0], fold[0][1]
        
       # test_images,test_reps, target_semantic_idxs = \
       # fold[1][0],fold[1][1],fold[1][2]
        
        test_images,test_reps = \
        fold[1][0],fold[1][1]
        

        '''
        classsifier = train(train_images,train_reps)
        predictions = test(classsifier,test_images)
        '''
        
        predictions = np.zeros(test_reps.shape)
    #    selected_dim = kwargs['selected_dim']
        number_dim = train_reps.shape[1]

        for index in range(number_dim):
            #  print(index)
            train_reps_index = train_reps[:,index]
            classsifier = train_ridge(train_images,train_reps_index)
            predictions_index = test(classsifier,test_images)
            predictions[:,index]=predictions_index
        
        
        
        results = evaluation(kwargs,predictions,
                                        test_reps,target_semantic_idxs)
        
        perf_test[i]= results['rank_accuracy_scores']
        print(str(results['rank_accuracy_scores']))
 
    print(str(perf_test))
    print(str(perf_test.mean()))
        

def handleDecoding(kwargs):
    
    '''decide on the method of running experiments'''
    '''cross-validation vs. train/test'''
    n_folds = kwargs['n_folds']
    subjects = kwargs['subjects_name'].split(',')
    voxel_selections = kwargs['total_voxel_selections'].split(',')
    embed_methods = kwargs['embed_methods']
    
    train_text_stimuli,test_text_stimuli = getTextStimuli(kwargs)

    train_file_types = kwargs['train_file_types']
    test_file_types = kwargs['test_file_types']
    

    for subject in subjects: 
        print ('handling subject: ' + subject)
        kwargs['subject_name'] = subject
        
        for method in embed_methods:
            
            kwargs['embed_method'] =method
            
            for index,train_type in enumerate(train_file_types):
                test_type = test_file_types[index]
            
                train_text_encoding,test_text_encoding = \
                   getTextEncoding(kwargs,method,train_type,\
                                   test_type)    
                
                train_text = train_text_stimuli,train_text_encoding    
                test_text = test_text_stimuli,test_text_encoding    
   
                for voxel_selection in voxel_selections: 
                    
                    logger.info('# of voxel:' + '\t' + str(voxel_selection))
                    kwargs['total_voxel_selection'] = voxel_selection
                    data_input = {}
                    
                    voxel_selected = loadVoxelAndSelect(kwargs)
                    train_image,test_image = processImage(kwargs,voxel_selected)
                    
                    data_input['train_text'] = train_text
                    data_input['test_text'] = test_text
                    data_input['train_image'] = train_image
                    data_input['test_image'] = test_image
                    kwargs['train_file_name'] = train_type + '_' + method
                    kwargs['test_file_name'] = test_type + '_' + method
                    
                    
                    if n_folds == 0:
                        '''train/test'''
                        processTrainTest(kwargs,data_input)
                    
                    else:
                        processCrossValidation(kwargs,data_input,n_folds)
       

def main(args):
    
    processor = Processor()
    kwargs = processor.parseArguments(args[1])
    handleDecoding(kwargs)
    

if __name__ == '__main__':
    
    main(sys.argv[1:])
