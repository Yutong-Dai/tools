import os
from urllib.request import urlretrieve
import progressbar
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import numpy as np
# import subprocess
# import shlex

class _bcolors:
    """
        Define colors for terminal output texts.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class LibsvmDataset:
    def __init__(self, download_dir="./raw", 
                 cleand_dir="./clean"):
        """
        Initialize the  LibsvmDataset class.
        
        Args:
            download_dir: A string specifies the place to store the downloaded raw dataset.
            cleand_dir: A string specifies the place to store the cleaned dataset.
        """
        self.download_dir = download_dir
        self.cleand_dir = cleand_dir
        for directory in [self.download_dir, self.cleand_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        self.url_regression = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression"
        self.url_binary = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"
        self.data_binary = [
            'a1a', 'a2a', 'a3a', 'a4a', 'a5a', 'a6a', 'a7a', 'a8a', 'a9a',
            'a1a.t', 'a2a.t', 'a3a.t', 'a4a.t', 'a5a.t', 'a6a.t', 'a7a.t', 'a8a.t', 'a9a.t', 
            'australian',
            'breast-cancer',
            'cod-rna', 'cod-rna.t', 'cod-rna.r', 
            'colon-cancer.bz2',
            'covtype.libsvm.binary.bz2',
            'diabetes',
            'duke.bz2',
            'fourclass',
            'german.numer',
            'gisette_scale.bz2', 'gisette_scale.t.bz2',
            'heart',
            'ijcnn1.bz2',
            'ionosphere_scale',
            'leu.bz2', 'leu.bz2.t',
            'liver-disorders', 'liver-disorders.t',
            'madelon', 'madelon.t',
            'mushrooms',
            'news20.binary.bz2',
            'phishing',
            'rcv1_train.binary.bz2','rcv1_test.binary.bz2',
            'real-sim.bz2', 'skin_nonskin',
            'splice', 'splice.t',
            'sonar_scale',
            'svmguide1', 'svmguide1.t', 'svmguide3', 'svmguide3.t', 
            'w1a', 'w2a', 'w3a', 'w4a', 'w5a', 'w6a', 'w7a', 'w8a',
            'w1a.t', 'w2a.t', 'w3a.t', 'w4a.t', 'w5a.t', 'w6a.t', 'w7a.t', 'w8a.t'
             #'epsilon_normalized.bz2', 'epsilon_normalized.t.bz2'
             #'HIGGS.bz2',
        ]        
        self.data_regression = [
            'abalone',
            'bodyfat',
            'cadata',
            'cpusmall',
            'log1p.E2006.train.bz2', 'log1p.E2006.test.bz2 '
            'E2006.train.bz2', 'E2006.test.bz2',
            'eunite2001', 'eunite2001.t', 'eunite2001.m',
            'housing',
            'mg',
            'mpg',
            'pyrim',
            'space_ga',
            'triazines',
            'YearPredictionMSD.bz2', 'YearPredictionMSD.t.bz2'
        ]       
        self.task_dict = {"binary":{"url":self.url_binary, 
                                    "dataset":self.data_binary}, 
                          "regression":{"url":self.url_regression, 
                                        "dataset":self.data_regression}}  
        # for printing
        self.pbar = None
        
    def _show_progress(self, block_num, block_size, total_size):
        """
            private function. Show the progress of urlretrieve for downloading data.
        """
        if self.pbar is None:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()
            self.pbar = None
    
    def _parseInputs(self, task=None, dataset=None, download_url=None):
        if task is not None and dataset is not None:
            print("You choose to use the task+dataset option.")
            try:
                work_dict = self.task_dict[task]
            except KeyError:
                print(f"{_bcolors.WARNING}Warning:Your input taks is [{task}], which currently is not supported.\n"\
                      f"However, you can provide an url pointing to the desired dataset to download it.{_bcolors.ENDC}")
                return
            is_available = dataset in work_dict["dataset"]
            if not is_available:
                print(f"{_bcolors.FAIL}Error occurs!\n"\
                     f"  1.Either the input dataset:[{dataset}] is not intended for the task:[{task}].\n"\
                     f"  2.Or the input dataset:[{dataset}] is not in the built-in database.\n"\
                     f"If you are sure the latter case happens, you can provide an url pointing to the desired dataset.{_bcolors.ENDC}"
                     )
                return
            self.download_url = work_dict["url"] + "/" + dataset
            self.task = task
            self.dataset = dataset
        elif download_url:
            print("You choose to use the url option.")
            try:
                task, dataset = download_url.split("/")[-2], download_url.split("/")[-1]
                self.download_url = download_url
                self.task = task
                self.dataset = dataset
            except IndexError:
                self.download_url = None
                print(f"{_bcolors.FAIL}The input url {download_url} is wrong.{_bcolors.ENDC}")
        else:
            raise ValueError(f"{_bcolors.FAIL}Code has bugs.{_bcolors.ENDC}")
        if self.download_url:
            print(f"Parsed task: [{self.task}] | Parsed dataset: [{self.dataset}]\nParsed download_url:[{self.download_url}]")
    def _getData(self, force_download):
        """
            Use urllib.request::urlretrieve to download the self.dataset and save to 
            self.download_dir/self.task based on the self.download_url.
            If the dataset already exists, then it will skip. However, you can force download
            by setting force_download=True.
        """
        if self.download_url is not None:
            # check whether the dataset is being downloaded or not
            directory = f"{self.download_dir}/{self.task}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            is_downloaded = os.path.exists(f"{directory}/{self.dataset}")
            if not is_downloaded or force_download:
                urlretrieve(self.download_url, f'{directory}/{self.dataset}', self._show_progress)
                #print("Start downloading... It may take a while")
                #subprocess.run(['wget', '-i', self.download_url, '-P', self.download_dir, 
                #                '-O', f'{self.download_dir}/{self.dataset}'])
                if os.path.exists(f"{directory}/{self.dataset}"):
                    print(f"{_bcolors.OKGREEN}dataset [{self.dataset}] is downloaded at [{directory}].{_bcolors.ENDC}")
            else:
                print(f"{_bcolors.WARNING}The dataset [{self.dataset}] already exists in [{directory}]!{_bcolors.ENDC}")
    def _cleanData(self, normalization, binary_label, force_clean):
        """
            Load the raw dataset from self.download_dir.
            If normalization is set, it will perform appropriate normalization.
            See docstring of the getAndClean to find valid values.
            
            If the cleaned dataset already exists, then it will skip. 
            However, you can force cleaning the dataset by setting force_clean=True.
        """
        directory = f"{self.cleand_dir}/{self.task}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        is_cleaned = os.path.exists(f"{directory}/{self.dataset}")
        if not is_cleaned or force_clean:
            data = load_svmlight_file(f"{self.download_dir}/{self.task}/{self.dataset}")
            X, y= data[0], data[1]
            n, p = X.shape
            # check label for the binary task
            if self.task == 'binary':
                y1old, y2old = np.unique(y)
                if binary_label is not None:
                    if  binary_label=='{-1,1}':
                        y1new, y2new = -1.0, 1.0
                    elif binary_label=='{0,1}':
                        y1new, y2new = 0.0, 1.0
                    else:
                        raise ValueError(f"Unrecognized binary_level: {binary_label}")
                    y[y==y1old] = y1new
                    y[y==y2old] = y2new
                    print(f"Original y-label range: {{ {y1old}, {y2old} }} -> New y-label range: {{ {np.unique(y)[0]}, {np.unique(y)[1]} }}")
                else:
                    raise ValueError(f"{_bcolors.FAIL}You should set the desired binary_level.\n For example, binary_label='[-1,1]'.{_bcolors.ENDC}")
            # check feature range
            if normalization is not None:
                print(f"Perform normalization:{normalization}")
                if normalization == 'feat-11':
                    for i in range(p):
                        temp = X[:,i]
                        if np.max(temp) > 1.0 or np.min(temp) < -1.0:
                            X[:,i] /= np.max(np.abs(temp))
                            if verbose:
                                print(f"  col:{i}: max:{np.max(temp):3.3e} | min:{np.min(temp):3.3e}\n"\
                                       "  Apply feature-wise [-1,1] scaling...")
                elif normalization == 'feat01':
                    for i in range(p):
                        temp = X[:,i]
                        xmax, xmin = np.max(temp), np.min(temp)
                        if xmax > 1.0 or  xmin < 0.0:
                            X[:,i] = (X[:,i] - xmin) / (xmax - xmin)
                            if verbose:
                                print(f"  col:{i}: max:{np.max(temp):3.3e} | min:{np.min(temp):3.3e}\n"\
                                       "  Apply feature-wise [0,1] scaling...")
                else:
                    raise ValueError(f"{_bcolors.FAIL}Unrecognized normalization: {normalization}{_bcolors.ENDC}")
            dump_svmlight_file(X, y, f"{directory}/{self.dataset}")
            if os.path.exists(f"{directory}/{self.dataset}"):
                print(f"{_bcolors.OKGREEN}Success: File saved at {directory}/{self.dataset}!{_bcolors.ENDC}")
                print("-*"*30)
        else:
             print(f"{_bcolors.WARNING}The cleaned dataset [{self.dataset}] already exists in [{directory}]!{_bcolors.ENDC}")
    def getAvailable(self):
            """Show supported tasks and for each supported task show avaiable datasets.
            Typical usage example:
            
                   libsvm = LibsvmDataset()
                   libsvm.getAvailable()
            """
            print("Current supported tasks are:")
            for k in self.task_dict.keys():
                print(f" ['{k}']", end="")
            print("\n=====================================")
            for k in self.task_dict.keys():
                print(f"For task:['{k}'], available datasets are:")
                print("----------------------------------------------------")
                dataset_lst = []
                for i,d in enumerate(self.task_dict[k]["dataset"]):
                    print(f" '{d}'", end=",")
                    if (i +1) % 5 == 0:
                        print("\n")
            print("\n")
    def getAndClean(self, task=None, dataset=None, download_url=None, 
                    binary_lable='{-1,1}', normalization='feat-11',
                    force_download=False, force_clean=False, clean_verbose=True
                   ):
        """Download and clean the dataset.
        Typical usage example:

              libsvm = LibsvmDataset()
              libsvm.getAndClean(task="binary", dataset="a1a", binary_lable='{-1,1}', normalization='feat-11')
              libsvm.getAndClean(task="regression", dataset="abalone", normalization='feat-11')
              libsvm.getAndClean(url='https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.bz2',
                                 binary_lable='{-1,1}', normalization='feat-11')              
        Args:

            task: A string specifies the task want to perform. Currently supported {'binary', 'regression'}.
            dataset: A string specifies the dataset you want to download. Use `getAvailable` method to show all
                     currently avaiable datasets for any given task.
            download_url: If the desired dataset is not provided for a given task, one can directly provide a url 
                          link to the desired data set. For example, one wants to download the avazu dataset.
                          One can visit https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#avazu.
                          If you want to download the "avazu-app.bz2" instance, you can right-click its name and 
                          select "copy link", then you should get a plain text as
                              <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.bz2>
                          Then just provides it as string.
            binary_label: If you want to perform binay classification, one can set labels to {-1,1} by providing
                          '{-1,1}'; or set labels to {0,1} by providing '{0,1}'. Default to '{-1,1}'.
            normalization: Perform feature-wise normalization. Currently supported options:
                             'feat-11': feature-wise scaling to range [-1,1]
                             'feat01': feature-wise scaling to range [0,1]
                           Default to '{-1,1}'.
            force_download: If set to True, then download the dataset even if it already exists. Default to False.
            force_clean:    If set to True, then clean the dataset even if it already exists in clean folder. Default to False.
            clean_verbose:  If set to True, will print out which feature being normalized. Default to False.
        """
        # reset
        self.download_url = None
        self.task = None
        self.dataset = None 
        self._parseInputs(task, dataset, download_url)
        if self.download_url is not None:
            self._getData(force_download)
            if self.task in ["binary", "regression"]:
                self._cleanData(normalization, binary_lable, force_clean)
            else:
                print(f"{_bcolors.WARNING}The clean rule for dataset:{self.dataset} with task:{self.task} is not defined. Hence, no cleaning is performed.{_bcolors.ENDC}")
        else:
            print(f"{_bcolors.WARNING}Fail to generate download url, please check your inputs.{_bcolors.ENDC}")