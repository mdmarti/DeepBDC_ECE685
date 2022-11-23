import pandas as pd

def prepare_dataset_csv(data_path,data_file,dsName='cats'):

    ### here we will prepare our datasets for deepBDC - we need to create an
    ### image_names column, and then an associated "image labels" column. First, we drop 
    ### columns with fewer than 1000 ims for training (cats).
    ### for birds, randomly select 70%

    data = data_path + '/' + data_file 
    data_df = pd.read_csv(data,sep=',',engine='python')

    newTrain, newTest = 'train.csv','test.csv'
    if dsName == 'cats':
        #1. get value counts for all breeds
        #2. train: breeds with more data, test: breeds with less
        #3. for unique names in train: go through folders, get all images
        # with size >= 100x100
        # create new df with breeds in 'image_labels" column, 
        # im path in "image_names" column
        #4. save train df as new csv, test df as new csv
        pass
    else:
        #1. rename labels as "image_labels"
        #2. rename filepaths, give full filepath
        #3. save train df as new csv, test df as new csv
        pass
    return newTrain,newTest