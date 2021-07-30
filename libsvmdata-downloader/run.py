from LibsvmDataset import *
libclass = LibsvmDataset()
# download datasets for the regression task
file_path = './libsvm_regression.txt'
libclass.getAndCleanFromFile(file_path,
                             task='regression', 
                             normalization='feat-11',
                             force_download=False, 
                             force_clean=False, 
                             clean_verbose=False)
file_path = './libsvm_binary_small.txt'
libclass.getAndCleanFromFile(file_path,
                             task='binary', 
                             binary_lable='{-1,1}', 
                             normalization='feat-11',
                             force_download=False, 
                             force_clean=False, 
                             clean_verbose=False)