import pandas as pd
import numpy as np
import glob

def get_split_inds(N,labels,probabilities):

    catOrder = np.random.choice(len(labels),len(labels),replace=False)#,replace=False,p=probabilities)
    nTrain = int(len(labels) * 0.7)
    nVal = int(0.15 * len(labels))
    nTest = len(labels) - nTrain - nVal
    trainOrd = catOrder[:nTrain]
    valOrd = catOrder[nTrain:nTrain + nVal]
    testOrd = catOrder[nTrain + nVal:]

    trainLabels = [labels[i] for i in trainOrd]
    valLabels = [labels[i] for i in valOrd]
    testLabels = [labels[i] for i in testOrd]

    trainPs = probabilities[trainOrd]
    trainPs /= sum(trainPs)
    valPs = probabilities[valOrd] 
    valPs /= sum(valPs)
    testPs = probabilities[testOrd]
    testPs /= sum(testPs)
    #print(testLabels)
    #print(trainLabels)
    #print(valLabels)
    for l in trainLabels:
        #print("curr label: ",l)
        assert(l not in valLabels),print(valLabels) 
        assert(l not in testLabels),print(testLabels)

    for l in valLabels:
        assert(l not in testLabels)
    trainOrd = np.random.choice(len(trainLabels),N,replace=True,p=trainPs)
    valOrd = np.random.choice(len(valLabels),int(N * 0.15/0.7),replace=True,p=valPs)
    testOrd = np.random.choice(len(testLabels),int(N * 0.15/0.7),replace=True,p=testPs)
    
    trainLabels = [trainLabels[i] for i in trainOrd]
    valLabels = [valLabels[i] for i in valOrd]
    testLabels = [testLabels[i] for i in testOrd]
    return trainLabels,valLabels, testLabels

def get_fns(labels,fnDict):

    fns = []

    for l in labels:
        filenames = fnDict[l]
        fnInd = np.random.choice(len(filenames))
        fns.append(filenames[fnInd])
    
    return fns

def prepare_dataset_csv(data_path,data_file,dsName='cats'):

    ### here we will prepare our datasets for deepBDC - we need to create an
    ### image_names column, and then an associated "image labels" column. First, we drop 
    ### columns with fewer than 1000 ims for training (cats).
    ### for birds, randomly select 70%

    newTrain, newVal, newTest = 'train.csv','val.csv','test.csv'
    if dsName == 'cats':
        newTrainFull = data_path + '/data/' + newTrain 
        newValFull = data_path + '/data/' + newVal
        newTestFull = data_path + '/data/' + newTest
        data = data_path + '/data/' + data_file 
        ims = data_path + '/images/'
        data_df = pd.read_csv(data,sep=',',engine='python')
        #1. get value counts for all breeds
        labels = data_df.breed.unique()
        label_counts = {}
        label_fns = {}
        for l in labels:
            count = data_df.breed.value_counts()[l]
            if count > 100:

                label_counts[l] = count
                imFns = glob.glob(ims + l +'/*.jpg')
                label_fns[l] = imFns
                #label_counts[l]['count'] = count
                #label_counts[l]['p'] = 1/count

        N = sum(label_counts.values())
        labels = list(label_counts.keys())
        cat_ps = N/list(label_counts.values())
        cat_ps = np.array(cat_ps/sum(cat_ps))

        trainLabels,valLabels, testLabels = get_split_inds(N,labels,cat_ps)

        trainFns = get_fns(trainLabels,label_fns)
        valFns = get_fns(valLabels,label_fns)
        testFns = get_fns(testLabels,label_fns)
        
        trainDF = pd.DataFrame({'image_names':trainFns,'image_labels':trainLabels})
        valDF = pd.DataFrame({'image_names':valFns,'image_labels':valLabels})
        testDF = pd.DataFrame({'image_names':testFns,'image_labels':testLabels})

        trainDF.to_csv(newTrainFull)
        valDF.to_csv(newValFull)
        testDF.to_csv(newTestFull)

    else:
        data = data_path + '/' + data_file 
        data_df = pd.read_csv(data,sep=',',engine='python')

        newTrainFull = data_path + '/' + newTrain 
        newValFull = data_path + '/' + newVal
        newTestFull = data_path + '/' + newTest

        labels = data_df.labels.unique()
        
        label_fns = {}
        label_counts = {}
        for l in labels:
            
            count = data_df.labels.value_counts()[l]
            label_counts[l] = count

            imFns = data_df.loc[data_df.labels == l].filepaths
            label_fns[l] = list(imFns)
        #1. rename labels as "image_labels"
        #2. rename filepaths, give full filepath
        #3. save train df as new csv, test df as new csv
        N = sum(label_counts.values())
        labels = list(label_counts.keys())
        cat_ps = N/list(label_counts.values())
        cat_ps = np.array(cat_ps/sum(cat_ps))

        trainLabels,valLabels, testLabels = get_split_inds(N,labels,cat_ps)

        trainFns = get_fns(trainLabels,label_fns)
        valFns = get_fns(valLabels,label_fns)
        testFns = get_fns(testLabels,label_fns)
        
        trainDF = pd.DataFrame({'image_names':trainFns,'image_labels':trainLabels})
        valDF = pd.DataFrame({'image_names':valFns,'image_labels':valLabels})
        testDF = pd.DataFrame({'image_names':testFns,'image_labels':testLabels})

        trainDF.to_csv(newTrainFull)
        valDF.to_csv(newValFull)
        testDF.to_csv(newTestFull)

    return newTrain,newVal, newTest