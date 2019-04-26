import numpy as np
from sklearn.decomposition import PCA

import sklearn  

def getTextStimuli(kwargs):
    
    trainTextInputs = getStimuli(kwargs,'train')
    testTextInputs = getStimuli(kwargs,'test')
    return trainTextInputs,testTextInputs

def getTextInputs(kwargs,type):
    
    '''actual text/text stimuli'''
    textStimuli = getTextStimuli(kwargs,type)
    
    return textStimuli

    
def getStimuli(kwargs,type):
    
    main_path = kwargs['main_path']
    text_path = kwargs['text_path']
    if type == 'train':
        sentence_file= kwargs['train_sentence_file']
    if type == 'test':
        sentence_file= kwargs['test_sentence_file']
        
    with open(main_path+text_path+sentence_file, "r") as sentences_f:
        textStimuli = [line.strip() for line in sentences_f]
    
    return textStimuli
 #   logger.info("Loaded text stimuli of size %s.", len(textStimuli))
 
def getTextEncoding(kwargs,embedding,train_type,\
                                   test_type):
    
   
    repre_path = kwargs['rep_path']
    if train_type == 'all' or train_type == 'last':
            train_type = 'contineous.' + train_type
            
    if test_type == 'all' or test_type == 'last':
            test_type = 'contineous.' + test_type

    train_file = repre_path+kwargs['train_num']+train_type+'.'+embedding+'.npy'
    print('train file '+train_file)
    train_rep = getTextContent(train_file,kwargs)
    
    test_file = repre_path+kwargs['test_num']+test_type+'.'+embedding+'.npy'
    print('test file '+test_file)
    test_rep = getTextContent(test_file,kwargs)

    return train_rep,test_rep
 
def getTextContent(fileName,kwargs):
    
    textRepresentation = np.load(fileName)
    textRepresentation = sklearn.preprocessing.scale(textRepresentation)

    reduce_dim = kwargs['reduce_dim']
    if reduce_dim == 'True':
        pca_dims = int(kwargs['selected_dim'])
        textRepresentation = reduceDimensions(pca_dims,textRepresentation)

    return textRepresentation
    
def getTextRepresentationFileNames(kwargs,type):

    '''textRepresentations = sentence vector representations'''
    repre_path = kwargs['rep_path']
    
    if type == 'train':
        file_types = kwargs['train_file_types']
    
    if type == 'test':
        file_types = kwargs['test_file_types']
    
    
    methods = kwargs['embed_methods']
    
    textRepresentationFiles = []

    for file_type in file_types:
        
        if file_type == 'all' or file_type == 'last':
            file_type = 'contineous.' + file_type
        for method in methods:
            if type == 'train':
                textRepresentationFiles.append(repre_path+'384'+\
                                             file_type+'.'+method+'.npy')
            if type == 'test':
                textRepresentationFiles.append(repre_path+'243'+\
                                             file_type+'.'+method+'.npy')
              
    #only load the file names....
    return textRepresentationFiles


def reduceDimensions(pca_dims,textRepresentations):
    
    pca = PCA(pca_dims).fit(textRepresentations)
    #logger.info("PCA explained variance: %f", sum(pca.explained_variance_ratio_) * 100)
    textRepresentations = pca.transform(textRepresentations)

    return textRepresentations