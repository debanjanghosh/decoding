import os
import configparser

import numpy as np

#import gensim
import nltk 
from nltk.corpus import stopwords 

from sklearn.manifold import TSNE
#import plotly.offline as plt
import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
#import plotly.graph_objs as go

import sys


class Processor():
    
    def __init__(self):
        print ('inside pre processor class ')
    
    def parseArguments(self,configFile):
    
        config = configparser.ConfigParser()
        config.read(configFile)
    
        kwargs = {}
    
        try:
            header = 'DECODING'
            kwargs['main_path'] = config.get(header, 'main_path')
            kwargs['text_path'] = config.get(header, 'text_path')
            kwargs['image_path'] = config.get(header, 'image_path')
            kwargs['voxel_scores'] = config.get(header, 'voxel_scores')
            kwargs['searchlight'] = config.get(header, 'searchlight')
            kwargs['voxel_neighbor_path'] = config.get(header, 'voxel_neighbor_path')
            kwargs['voxel_neighbor_file'] = config.get(header, 'voxel_neighbor_file')
            kwargs['total_voxel_selections'] = config.get(header, 'total_voxel_selections')
            kwargs['cut_off_row'] = int(config.get(header,'cut_off_row'))
            kwargs['voxel_score_file'] = config.get(header,'voxel_score_file')
            kwargs['voxel_selection'] = config.get(header,'voxel_selection')
            kwargs['subjects_name'] = config.get(header, 'subjects_name')
            kwargs['train_sentence_file'] = config.get(header, 'train_sentence_file')
            kwargs['test_sentence_file'] = config.get(header, 'test_sentence_file')
            kwargs['reduce_dim'] =  config.get(header, 'reduce_dim')
            kwargs['n_folds'] =  int(config.get(header, 'n_folds'))
            kwargs['max_neighbors'] = int(config.get(header, 'max_neighbors'))
            kwargs['test_image'] =  config.get(header, 'test_image')
            kwargs['train_image'] =  config.get(header, 'train_image')
            kwargs['eval_context'] = config.get(header,'eval_context')
            kwargs['rep_path'] =  config.get(header, 'rep_path')
           # kwargs['test_representation_files'] =  config.get(header, 'test_representation_files').split(',')
           # kwargs['train_representation_files'] =  config.get(header, 'train_representation_files').split(',')
            kwargs['train_file_types'] =  config.get(header, 'train_file_types').split(',')
            kwargs['test_file_types'] =  config.get(header, 'test_file_types').split(',')

            kwargs['embed_methods'] =  config.get(header, 'embed_methods').split(',')

            kwargs['train_num'] =  config.get(header, 'train_num')
            kwargs['test_num'] =  config.get(header, 'test_num')
        
            kwargs['eval_passage'] =  config.get(header, 'eval_passage')
            kwargs['eval_all'] =  config.get(header, 'eval_all')

            kwargs['voxel_selection_process'] =  config.get(header, 'voxel_selection_process')
        
        except:
            print(kwargs)
            print("check the parameters that you entered in the config file")
            exit()
            
        print ('args = ' , kwargs)
        return kwargs
    
    

    def tsne(self,kwargs):
        
        repre_path = kwargs['rep_path']
        representation_file = kwargs['test_representation_file']

        textRepresentations = np.load(repre_path+representation_file)
        textRepresentations = textRepresentations / np.linalg.norm(textRepresentations, axis=1, keepdims=True)
        textRepresentations = textRepresentations[0:8]
        labels = ['sent1','sent2','sent3','sent4','sent5','sent6','sent7','sent8']
        
        similarities = np.dot(textRepresentations,\
                        textRepresentations.T)
        rankings = np.argsort(-similarities, axis=1)

        
        #tsne = TSNE(n_components=3, random_state=0)
        #np.set_printoptions(suppress=True)
        #Y = tsne.fit_transform(textRepresentations)
        
        tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
        new_values = tsne_model.fit_transform(textRepresentations)

        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])
        
        plt.figure(figsize=(16, 16)) 
        for i in range(len(x)):
            plt.scatter(x[i],y[i])
            plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
      #  plt.colorbar()
        plt.show()

def main(args):
    
    processor = Processor()
    kwargs = processor.parseArguments(args[1])
    
    processor.rename_neighbors()
    
  #  processor.saveNPFiles()

  #  processor.tsne(kwargs)
  #  processor.prepareEmbeddings()
    
if __name__ == '__main__':

    main(sys.argv[1:])