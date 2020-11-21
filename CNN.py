####################################
# This code is for pos rec analysis
# in PandaX-4T, using CNN as the architecture
# for training.
#####################################
import torch
import numpy as np
import json, time, sys
import pandas as pd
import glob
import pickle as pkl


#########################
# seeding
#########################
SEED    = int(time.time()/1000)  + int(time.time())%1000
_       = np.random.seed(SEED)
_       = torch.manual_seed(SEED)


#########################
# Hard-coded
#########################
WorkDir         = '/home/ustc/WorkSpace/PandaX-4T/PosRecNN'

LogDir          = WorkDir+'/logs'
WeightDir       = WorkDir+'/weights'
FigureDir       = WorkDir+'/figs'
OutputDir       = WorkDir+'/outputs'

TrainData       = [
    '/home/ustc/WorkSpace/PandaX-4T/PosRecNN/samples2/train_v2_smeared_job1_image.pkl',
    '/home/ustc/WorkSpace/PandaX-4T/PosRecNN/samples2/train_v2_smeared_job2_image.pkl', # for debug use, temporarily disabled
]
# TrainData       = WorkDir+'/samples/tmp_train_mixed.json' # just for debugging phase
TestData        = [
    '/home/ustc/WorkSpace/PandaX-4T/PosRecNN/samples2/test_v2_smeared_image.pkl',
]

TotalPMTNum     = 169 # +199

TotalPixelNum   = 128 #image pixel, padding the ones more than 169

# from Huangdi @ 2020-09-11
# Note it has be converted to MC ids
# commented is the database pmt ids started from 1
# InhibitedPMTs   = [5,6,176,252,302,348,353,365]
InhibitedPMTs   = [164, 163, 253, 290, 212, 286, 329, 343]


######################
## Input
######################

if len(sys.argv)<2:
    Message  = 'python3 CNN.py <mode (train or validate or test or inference)> <weight file> <number of iterations> <file head (test filename if inference mode)> < (only for validate mode) skip number of weight files> <(only for validate mode) Iter range lower> <(only for validate mode) Iter range upper> <(only for inference) inference filename> <(only for inference> inference output filename>\n'
    Message += 'Note <weight file> does not function if the mode is validate.'
    print(Message)
    exit()


Mode                = sys.argv[1]
WeightFilename      = None
if len(sys.argv)>2:
    WeightFilename  = sys.argv[2]
NumIter             = 1000
if len(sys.argv)>3:
    NumIter         = int(sys.argv[3])
FileHead            = 'baseline'
if len(sys.argv)>4:
    FileHead        = sys.argv[4]
SkipNum             = 1
if len(sys.argv)>5:
    SkipNum         = int(sys.argv[5])
IterRangeLower      = 0
IterRangeUpper      = 1000000000000
if len(sys.argv)>6:
    IterRangeLower  = int(sys.argv[6])
    IterRangeUpper  = int(sys.argv[7])
InferenceFilename   = None
if len(sys.argv)>8:
    InferenceFilename = sys.argv[8]
InferenceOutputFilename   = None
if len(sys.argv)>9:
    InferenceOutputFilename = sys.argv[9]


# check
if Mode not in ['train', 'test', 'validate', 'inference']:
    raise ValueError("Mode must be either train, validate or test or inference!")

if WeightFilename=='None' or WeightFilename=='none':
    WeightFilename  = None

if Mode=='inference':
    TestData    = [InferenceFilename]


########################
# Load json file
#######################
from pattern_plot import ChToMCPMTIDs, ToDataCoord, ToMCCoord, ChToDataPMTIDs
from Libs_cnn import FormDataset_Data, FormDataset_MC
dataset = []
DataSets    = TrainData
if Mode!='train':
    DataSets    = TestData

for Filename in DataSets:
    dataset_array   = pkl.load(open(Filename,'rb'))
    NumEvents       = len(dataset_array)
    print("===>>> File: "+Filename+" Loaded! Total event number = "+str(NumEvents))
    if Mode=='validate':
        # hardcode
        # if validate, just use the first 1000 events
        NumEvents = 1000
    dataset.extend(dataset_array[:NumEvents])



print("===>>> Number of events = "+str(dataset.__len__()))
print("===>>> Each image shape = "+str(dataset.__getitem__(1)[0].shape))


data_loader     = None
from SparseDataLoader import sparse_collate
if Mode=='train':
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        collate_fn=sparse_collate
    )
else:
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=sparse_collate
    )

##################
# blob
##################
from Libs_cnn import CNN

class BLOB:
    pass
blob            = BLOB()
blob.net        = CNN().float().cuda() # construct Lenet, use GPU
blob.criterion  = torch.nn.MSELoss() # L2 loss
blob.optimizer  = torch.optim.Adam(blob.net.parameters()) # use Adam optimizer algorithm
blob.softmax    = torch.nn.Softmax(dim=1) # not for training, but softmax score for each class
blob.iteration  = 0    # integer count for the number of train steps
blob.data       = None # data for training/analysis
blob.label      = None # label for training/analysis

#############################
## Main
#############################

from Libs_cnn import restore_state

# first restore weights
if WeightFilename is not None:
    restore_state(blob, WeightFilename)

StartNumIter    = blob.iteration
EndNumIter      = blob.iteration + NumIter


###############################
# train
# not tested
###############################
from Libs_cnn import train_loop

if Mode=='train':
    train_loss      = train_loop(blob, data_loader, EndNumIter, WeightDir, FileHead)
    # save to log
    LogFilename     = LogDir+'/'+FileHead+'-log-'+str(StartNumIter)+"-"+str(EndNumIter)+'.csv'
    Dict = {}
    Dict['iter']    = np.linspace(StartNumIter, EndNumIter-1, NumIter)
    Dict['loss']    = train_loss
    df              = pd.DataFrame(Dict)
    df.to_csv(LogFilename)
    exit()

###############################
# Validate
# not tested
###############################
import glob
from Libs_cnn import test_loop
if Mode=='validate':
    # Get the all the weight files with Filehead
    WeightFilenames = glob.glob(WeightDir+'/'+FileHead+'*')
    # Obtain the iteration number
    IterNum = []
    for weight_filename in WeightFilenames:
        IterNum.append(eval(
            weight_filename.split(WeightDir+'/'+FileHead+'-')[-1].split('.')[0]
        ))
    # convert to numpy array
    WeightFilenames = np.asarray(WeightFilenames)
    IterNum         = np.asarray(IterNum)
    # Sort
    inds            = np.argsort(IterNum)
    WeightFilenames = WeightFilenames[inds]
    IterNum         = IterNum[inds]
    # cut off
    inds1           = np.where(IterNum>IterRangeLower)[0]
    inds2           = np.where(IterNum<IterRangeUpper)[0]
    inds            = np.intersect1d(inds1, inds2)
    WeightFilenames = WeightFilenames[inds]
    IterNum         = IterNum[inds]
    # Skip weight files
    if SkipNum>1:
        WeightFilenames = WeightFilenames[::SkipNum]
        IterNum         = IterNum[::SkipNum]
    # Loop over
    Losses  = []
    for weight_filename in WeightFilenames:
        # restore the state
        restore_state(blob, weight_filename)
        # get the prediction
        res             = test_loop(blob, data_loader)
        loss            = np.average(res['loss'])
        Losses.append(loss)
        # print out
        print("===>>> File: "+weight_filename+' processed!\n')
    # save to pandas
    import os
    OutputValidationFilename    = LogDir+'/'+FileHead+'-'+'validate.csv'
    if not os.path.isfile(OutputValidationFilename):
        OutDict = {}
        OutDict['iter'] = IterNum
        OutDict['loss'] = Losses
        df      = pd.DataFrame(OutDict)
        df.to_csv(OutputValidationFilename)
    else:
        df  =  pd.read_csv(OutputValidationFilename)
        IterNum = IterNum.tolist()
        IterNum.extend(df.iter.tolist())
        Losses.extend(df.loss.tolist())
        # re-sort
        IterNum = np.asarray(IterNum)
        Losses  = np.asarray(Losses)
        inds    = np.argsort(IterNum)
        IterNum = IterNum[inds]
        Losses  = Losses[inds]
        # then save
        OutDict = {}
        OutDict['iter'] = IterNum
        OutDict['loss'] = Losses
        df = pd.DataFrame(OutDict)
        df.to_csv(OutputValidationFilename)
    exit()

###############################
# test & inference
# not tested
###############################
res         = test_loop(blob, data_loader)
prediction  = res['prediction']
label       = res['label']





from pattern_plot import ToDataPMTIDs
import pickle as pkl
# Output
OutputFilename  = OutputDir+'/'+FileHead+'-'+str('%d' % blob.iteration)+'-output.pkl'
if Mode=='inference':
    OutputFilename = InferenceOutputFilename
OutDict = {}
OutDict['prediction']   = prediction
OutDict['label']        = label


pkl.dump(
    OutDict,
    open(OutputFilename,'wb')
)




