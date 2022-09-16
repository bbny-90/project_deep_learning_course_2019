import numpy as np   
from PIL import Image

def get_name(dataset, num):
    '''
    Get the name of one image.
    argument:
        dataset: the dataset that the image is in
        num: the index of the image in the dataset
    return:
        imgName: the name of the image
    '''
    imgName = dataset[dataset['digitStruct']['name'][num][0]].value
    imgName = ''.join(chr(i) for i in imgName)
    
    return imgName


# The following 2 functions are basicly from
# https://github.com/bdiesel/tensorflow-svhn/blob/master/digit_struct.py
def bbox_helper(dataset, attr):
    '''
    bbox_helper abstracts the bbox or an array of bbox.
    used internally with get_bbox
    argument:
        dataset: the dataset that the image is in
        attr: the name of the attribute we want from the image
    return:
        attr: the attribute we want from the image
    '''
    if (len(attr) > 1):
        attr = [dataset[attr.value[j].item()].value[0][0] for j in range(len(attr))]
    else:
        attr = [attr.value[0][0]]
    return attr


def get_bbox(dataset, n):
    '''
    getBbox returns a dict of data for the n(th) bbox. 
    argument:
        dataset: the dataset that the image is in
        n: index for digit structure
    return: 
        bbox: a hash with the coordiantes
        e.g. {'width': [23.0, 26.0], 'top': [29.0, 25.0], 'label': [2.0, 3.0], 'left': [77.0, 98.0], 'height': [32.0, 32.0]}
    '''
    bbox = {}
    dataBbox = dataset['digitStruct']['bbox']
    bb = dataBbox[n].item()
    
    bbox['label'] = bbox_helper(dataset, dataset[bb]["label"])
    bbox['top'] = bbox_helper(dataset, dataset[bb]["top"])
    bbox['left'] = bbox_helper(dataset, dataset[bb]["left"])
    bbox['height'] = bbox_helper(dataset, dataset[bb]["height"])
    bbox['width'] = bbox_helper(dataset, dataset[bb]["width"])

    return bbox


def image_extractor(imgName, box, size, phase, dataDir):
    '''
    Extract the effective part of the image, i.e. the digit(s). 
    argument:
        imgName: the name of the image
        box: the set of attributes of the the image
        size: the output size we want
        phase: "train", "test" or "extra", used for finding the right folder
        dataDir: the folder containing all of the data
    return:
        img: the output image
    '''
    img = Image.open(dataDir + phase + '/' + imgName)
    
    box['right'] = list(np.array(box['left']) + np.array(box['width']))
    box['bottom'] = list(np.array(box['top']) + np.array(box['height']))
    
    # since the origin of an image is the top left, to extract the digit(s), 
    # we should use the minimum in "left" and "top" attributes and the maximum in "right" and "bottom" attributes
    img = img.crop((np.min(box['left']), np.min(box['top']), np.max(box['right']), np.max(box['bottom'])))
    img = img.resize((size, size))
    
    return img


def generate_trainable_data(dataset, size, phase, dataDir="./data/"):
    '''
    Generate data that can be used.  
    argument: 
        dataset: the dataset we use
        size: the size we want for the data (images)
        phase: "train", "test" or "extra", used for finding the right folder in image_extractor
        dataDir: the folder containing all of the data
    return: 
        trainableData: the output data that can be used
        label: the corresponding labels of the data
    '''
    num = dataset['digitStruct']['name'].shape[0]
    trainableData = np.zeros((num, size, size, 3))
    label = np.zeros((num, 6))
    
    for i in range(num):
        imgName = get_name(dataset, i)
        box = get_bbox(dataset, i)
        
        numDigit = len(box['label'])
        if numDigit >= 6:
            continue
        
        img = image_extractor(imgName, box, size, phase, dataDir)
        
        trainableData[i] = np.array(img)

        label[i, 0] = numDigit
        for j in range(numDigit):
            label[i, j+1] = box['label'][j]
        
    return trainableData, label


def preprocess_data(trainData, testData):
    '''
    Normalization of data. 
    argument: 
        trainData: the training data
        testData: the test data
    return: 
        trainData: the training data after preprocess
        testData: the test data after preprocess
    '''
    mean = np.mean(trainData, axis=0)
    trainData = trainData.astype(np.float32) - mean.astype(np.float32)
    testData = testData.astype(np.float32) - mean.astype(np.float32)

    trainData /= 255
    testData /= 255
    
    return trainData, testData
