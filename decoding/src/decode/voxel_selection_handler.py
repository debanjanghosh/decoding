
import scipy.io
import numpy as np
from collections import defaultdict
import os
from random import shuffle 


def collectNeighbors(kwargs,subject):
    
    voxel_neighbor_path = kwargs['voxel_neighbor_path']
    voxel_neighbor_file = kwargs['voxel_neighbor_file']
    
    main_path = kwargs['main_path']
    all_neighbors = []
    with open(main_path+voxel_neighbor_path+'/'+subject+'/'+voxel_neighbor_file + subject + '.txt') as f:
        for line in f:
            neighbors = line.split()
            all_neighbors.append(neighbors)

    return all_neighbors

def getNeighborTop(voxel_rankings,allNeighbors):
    
    voxel_top  = voxel_rankings[0:20000]

    all_neighbors = []
    for top in voxel_top:
        neighbors = allNeighbors[top]
        neighbors = [int(neighbor) for neighbor in neighbors]
        all_neighbors.extend(neighbors)
        
    all_neighbors_unique = list(set(all_neighbors))
    
    neighbors_max = []
    for top in voxel_rankings[20000:]:
        if top in all_neighbors_unique:
            neighbors_max.append(top)
            if len(neighbors_max) >=10000:
                break
    
    return neighbors_max
        

def loadVoxelAndSelect(kwargs):
    subject = kwargs['subject_name']
    voxelScorePath = kwargs['voxel_scores']
    mainPath = kwargs['main_path']
    embed_method = kwargs['embed_method']
    voxel_score_file = kwargs['voxel_score_file']
    voxel_score_file+=embed_method+'_'+subject+ '.mat'
    score_data = scipy.io.loadmat(mainPath+voxelScorePath+voxel_score_file)
    all_scores = score_data.get('scores')

    image_path = kwargs['image_path']
    train_image = kwargs['train_image']
    voxel_data = scipy.io.loadmat(mainPath+image_path+'/'+subject+'/'+train_image+subject)
    meta = voxel_data['meta']
    all_values_mat = voxel_data['examples']
    
    voxel_selection = kwargs['voxel_selection']
    if voxel_selection == 'False':
        return []
    
    '''we select voxels by two methods'''
    '''1) maximum predictability across all dimensions'''
    '''2) predictability separately/dimension'''
    '''3) PCA '''
    
    voxel_selection_process =  kwargs['voxel_selection_process']
    total_voxel_selection = int(kwargs['total_voxel_selection'])
    
    searchlight = kwargs['searchlight']
    if searchlight == 'True':
        allNeighbors = collectNeighbors(kwargs,subject)

    if voxel_selection_process == 'all_dimensions':
        voxel_top = selectVoxelsUsingAllDims(all_scores,total_voxel_selection)
    elif voxel_selection_process == 'sep_dimension' :
        cut_off = kwargs['cut_off_row']
        voxel_top = selectVoxelsUsingSeparateDims(all_scores,cut_off,\
                                                  total_voxel_selection)
    elif voxel_selection_process == 'PCA' :
        return []
      



    return voxel_top

def selectVoxelsUsingSeparateDims(all_scores,cut_off,total_voxel_selection):
    
    selected_voxels = []
    for row in range(len(all_scores)):
        columns = all_scores[row]
        rankings = np.argsort(-columns)
        rankings_top = rankings[:cut_off]
        
        selected_voxels.extend(rankings_top)
    
    selected_voxels = list(set(selected_voxels))
    shuffle(selected_voxels)
    voxel_top = selected_voxels[0:min(total_voxel_selection,len(selected_voxels))]

    return voxel_top


def selectVoxelsUsingAllDims(all_scores,total_voxel_selection):
    
    x = np.matrix(np.arange(12).reshape((3,4)))
    x_select = np.max(x)
    x_select = np.sum(x,axis=0)
    
    voxelsSelected = np.sum(all_scores,axis=0)
    voxel_rankings = np.argsort(-voxelsSelected)
    voxel_top  = voxel_rankings[0:total_voxel_selection]

    return voxel_top


 