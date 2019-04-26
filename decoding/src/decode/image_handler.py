
import numpy as np
import scipy.io
from sklearn.decomposition import PCA

def processImage(kwargs,voxel_selected):
    
    subject = kwargs['subject_name']
    trainImage = getImage(kwargs,subject,'train')
    voxel_selection = kwargs['voxel_selection']
    voxel_selection_process =  kwargs['voxel_selection_process']
    total_voxel_selection = int(kwargs['total_voxel_selection'])
    
    if voxel_selection == 'True' :
        
        if voxel_selection_process == 'PCA':
            pca = PCA(total_voxel_selection).fit(trainImage)
            voxelSelected_trainImage = pca.transform(trainImage)
        else:
            voxelSelected_trainImage = voxelFiltered(trainImage,voxel_selected)
    else:
        voxelSelected_trainImage = trainImage
    
    testImage = getImage(kwargs,subject,'test')
    if voxel_selection == 'True':
        if voxel_selection_process == 'PCA':
            pca = PCA(total_voxel_selection).fit(testImage)
            voxelSelected_testImage = pca.transform(testImage)
        else:
            voxelSelected_testImage = voxelFiltered(testImage,voxel_selected)
    else:
        voxelSelected_testImage = testImage


    return voxelSelected_trainImage,voxelSelected_testImage


def voxelFiltered(trainImage,voxel_selected):

    voxelSelected_trainImage = np.zeros((len(trainImage),len(voxel_selected)),np.float32)
    for row in range(len(trainImage)):
        for index,column in enumerate(voxel_selected):
            voxelSelected_trainImage[row][index] = trainImage[row][column]

    return voxelSelected_trainImage

def getContextIds():
    
    target_semantic_idxs = []
    index = 3
    while True:
        target_semantic_idxs.append(index)
        index+=4
        if index >= 256:
            break
        
    return np.array(target_semantic_idxs)

def getImage(kwargs,subject,type):
    
    main_path = kwargs['main_path']
    image_path = kwargs['image_path']
    if type == 'train':
        imageFile = kwargs['train_image']
    elif type == 'test':
        imageFile = kwargs['test_image']
    else:
        raise Exception('wrong type set for experiments ' + \
         '(misspelled train/test?)')

    data = scipy.io.loadmat(main_path+image_path+'/'+subject+'/'+imageFile+subject)
    
    
  #  logger.info("Loaded subject train %s data.")
    images = data["examples"]
        
    return images

