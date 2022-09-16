import os
import urllib.request as url
import tarfile

def download_svhn(dataDir="./data", phase=None):
    '''
    The pipeline of downloading is learnt from cifar_utils.py in assignments.
    argument:
        dataDir: the name of the folder to store data
        phase: "all" or None, None means only to download training and test data
    return:
        None
    '''
    # create data folder #
    if not os.path.exists(dataDir):
        os.mkdir(dataDir)
    
    dataURL = "http://ufldl.stanford.edu/housenumbers/"
    if os.path.exists(dataDir + "/train.tar.gz"):
        print("SVHN training data already exists.")
    else:
        print("Start downloading training data.")
        url.urlretrieve(dataURL + "train.tar.gz", dataDir + "/train.tar.gz")
        print("Complete downloading training data.")
        
    if os.path.exists(dataDir + "/test.tar.gz"):
        print("SVHN test data already exists.")
    else:
        print("Start downloading test data.")
        url.urlretrieve(dataURL + "test.tar.gz", dataDir + "/test.tar.gz")
        print("Complete downloading test data.")
        
    if phase == "all":
        if os.path.exists(dataDir + "/extra.tar.gz"):
            print("SVHN extra data already exists.")
        else:
            print("Start downloading extra data.")
            url.urlretrieve(dataURL + "extra.tar.gz", dataDir + "/extra.tar.gz")
            print("Complete downloading extra data.")
        

def untar_svhn(dataDir="./data/", phase=None):
    '''
    Untar SVHN data.
    argument:
        dataDir: the name of the folder containing data
        phase: "all" or None, None means only to untar training data and test data
    return:
        None
    '''
    print("Start untaring training data.")
    tar = tarfile.open(dataDir + "train.tar.gz")
    names = tar.getnames()
    for name in names:
        tar.extract(name, dataDir)
    tar.close()
    print("Complete untaring training data.")

    print("Start untaring test data.")
    tar = tarfile.open(dataDir + "test.tar.gz")
    names = tar.getnames()
    for name in names:
        tar.extract(name, dataDir)
    tar.close()
    print("Complete untaring test data.")
    
    if phase == "all":
        print("Start untaring extra data.")
        tar = tarfile.open(dataDir + "extra.tar.gz")
        names = tar.getnames()
        for name in names:
            tar.extract(name, dataDir)
        tar.close()
        print("Complete untaring extra data.")