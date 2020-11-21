#######################################
## Simple libs for ghost tagging
########################################
import torch
import numpy as np
import json, time, sys
import sparseconvnet as scn

GlobalTopNumPMTs        = 169

#################################
## CNN
#################################


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        ###############################
        # Hardcoded settings
        ###############################
        self._dimension     = 3
        reps                = 2
        kernel_size         = 2
        num_strides         = 7
        init_num_features   = 8
        nInputFeatures      = 1
        spatial_size        = 128 #padding the rest for 169 PMTs
        num_classes         = 2 # good versus ghost

        nPlanes             = [(2**i)*init_num_features for i in range(0, num_strides)] # every layer double the number of features
        downsample          = [kernel_size, 2]
        leakiness           = 0


        #################################
        # Input layer
        #################################
        self.input = scn.Sequential().add(
            scn.InputLayer(self._dimension, spatial_size, mode=3)).add(
            scn.SubmanifoldConvolution(self._dimension, nInputFeatures, init_num_features, 3, False))  # Kernel size 3, no bias
        self.concat = scn.JoinTable()
        #################################
        # Encode layers
        #################################\
        self.encoding_conv = scn.Sequential()
        for i in range(num_strides):
            if i < 4: #hardcoded
                self.encoding_conv.add(
                    scn.BatchNormLeakyReLU(nPlanes[i], leakiness=leakiness)).add(
                    scn.Convolution(self._dimension, nPlanes[i], nPlanes[i + 1],
                                    downsample[0], downsample[1], False))
            elif i < num_strides-1:
                self.encoding_conv.add(
                    scn.MaxPooling(self._dimension, 2, 2)
                )

        self.output     = scn.Sequential().add(
            scn.SparseToDense(self._dimension, nPlanes[-1])
        )
        ###################################
        # Final linear layer
        ###################################
        self.deepest_layer_num_features  = int(nPlanes[-1]*np.power(
            spatial_size/(2**(num_strides-1)),
            3.
        ))
        self.classifier                 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(self.deepest_layer_num_features, 2),
        )





    def forward(self, input):
        '''
        input shall be in (Batch, Dimension) shape
        as (batch_id, pmt_index,
        :param x:
        :return:
            coords,
            features
        '''
        # use the traditional inputs
        point_cloud = input
        coords = point_cloud[:, 0:self._dimension + 1].float()
        features = point_cloud[:, self._dimension + 1:].float()
        # for debug
        # print("coords type = "+str(type(coords)))
        # print("features type = "+str(type(features)))
        ########################
        # got through input layer
        ########################
        x           = self.input((coords, features))
        # for debug
        # print("===>>> After input x info: "+str(x.__repr__()))
        ########################
        # go through encoding layers
        ########################
        for i, layer in enumerate(self.encoding_conv):
            # for debug
            # print("==>> This is "+str(i)+" layer encoder!")
            # x           = self.encoding_block[i](x) # temporary not using res-blocks
            # print("==>> After residual block x info: "+str(x.__repr__()))
            x           = self.encoding_conv[i](x)
            # print("==>> After convolution x info: "+str(x.__repr__()))x size = "+str(x.size()))
        #########################
        # to dense
        #########################
        x           = self.output(x)
        x           = x.view((-1, self.deepest_layer_num_features))
        # print("===>>> After densing x size = "+str(x.size()))
        #########################
        # go through linear layer to give scores
        #########################
        rec_pos     = self.classifier(x)
        # print("===>>> After linear rec_pos size = "+str(rec_pos.size()))
        return coords, rec_pos




##################################
## Forward and Backward functions
##################################
def forward(blob, train=True):
    """
       Args: blob should have attributes, net, criterion, softmax, data, label

       Returns: a dictionary of predicted labels, softmax, loss, and accuracy
    """
    with torch.set_grad_enabled(train):
        # Prediction
        data        = blob.data.cuda()
        coords, score  = blob.net(data)
        # debug
        # print("score info: "+str(score.__repr__()))
        # Training
        loss, acc = -1, -1
        if blob.label is not None:
            label = blob.label.cuda().float()
            # # debug
            # print("score size = "+str(score.size()))
            # print("label size = "+str(label.size()))
            # print("score = "+str(score))
            # print("label = "+str(label))
            loss = blob.criterion(score, label.view((-1,2)))
            # print("loss = "+str(loss))
        blob.loss = loss
        # debug
        # print("loss info: "+str(loss.__repr__()))

        prediction  = score.detach().cpu().numpy()

        return {
            'prediction':   prediction,
            'loss':         loss.cpu().detach().item(),
        }

def backward(blob):
    blob.optimizer.zero_grad()  # Reset gradients accumulation
    blob.loss.backward()
    blob.optimizer.step()


#########################
# Save & Restore
#########################
def save_state(blob, prefix='./snapshot'):
    # Output file name
    filename = '%s-%d.ckpt' % (prefix, blob.iteration)
    # Save parameters
    # 0+1) iteration counter + optimizer state => in case we want to "continue training" later
    # 2) network weight
    torch.save({
        'global_step': blob.iteration,
        'optimizer': blob.optimizer.state_dict(),
        'state_dict': blob.net.state_dict()
        }, filename)
    return filename

def restore_state(blob, weight_file):
    # Open a file in read-binary mode
    with open(weight_file, 'rb') as f:
        # torch interprets the file, then we can access using string keys
        checkpoint = torch.load(f)
        # load network weights
        blob.net.load_state_dict(checkpoint['state_dict'], strict=False)
        # if optimizer is provided, load the state of the optimizer
        if blob.optimizer is not None:
            blob.optimizer.load_state_dict(checkpoint['optimizer'])
        # load iteration count
        blob.iteration = checkpoint['global_step']

#########################
# train loop
########################
def train_loop(blob, train_loader, num_iteration, WeightDir, FileHead):
    # Set the network to training mode
    blob.net.train()
    # Let's record the loss at each iteration and return
    train_loss=[]
    current_time        = time.time()
    # Loop over data samples and into the network forward function
    while blob.iteration < num_iteration:
        for i, data in enumerate(train_loader):
            if blob.iteration >= num_iteration:
                break
            if (i+1)%2000==0:
                # every 2000 iteration save the weight
                save_state(blob, WeightDir+'/'+FileHead)
            blob.iteration += 1
            # data and label
            blob.data, blob.label = data
            # call forward
            res = forward(blob,True)
            # Record loss
            train_loss.append(res['loss'])
            # once in a while, report
            if blob.iteration == 0 or (blob.iteration+1)%10 == 0:
                print('Iteration',blob.iteration,' ===>>> Loss',res['loss'])
                time_consumption        = time.time() - current_time
                current_time            = time.time()
                print('Cost '+str('%.3f' % time_consumption)+' seconds.')
            backward(blob)
    return np.array(train_loss)



##########################
## test loop
##########################

def test_loop(blob, test_loader):
    # give the prediction
    prediction, loss, accuracy, label    = [], [], [], []
    current_time        = time.time()
    for i, data in enumerate(test_loader):
        blob.data, blob.label = data
        res = forward(blob, False)
        # # debug
        # print("prediction = "+str(res['prediction']))
        # print("prediction type = "+str(type(res['prediction'])))
        # print("accuracy type = "+str(type(res['accuracy'])))
        # print("softmax type = "+str(type(res['softmax'])))
        # print("loss type = "+str(type(res['loss'])))
        # print("loss shape = "+str(res['loss'].size()))
        # print("label = "+str(blob.label))
        # print("label type = "+str(type(blob.label)))
        # prediction & label de-batch
        one_prediction          = res['prediction'].detach().cpu().numpy()
        one_batchid             = blob.data[:,3].detach().cpu().numpy()
        one_label               = blob.label.detach().cpu().numpy()
        for batch_id in np.unique(one_batchid):
            # batch_id is in ascending order by default from numpy unique
            # find indexes
            inds                = np.where(one_batchid==batch_id)[0]
            # append
            prediction.append(one_prediction[inds])
            label.append(one_label[inds])
        # append
        loss.append(res['loss'].detach().cpu().numpy())
        accuracy.append(res['accuracy'])
        if (i+1)%10==0:
            print("===>>> "+str(i+1)+" batches have been inferred!")
            time_consumption        = time.time() - current_time
            current_time            = time.time()
            print("===>>> Cost "+str('%.3f' % time_consumption)+" seconds!")
            print()
    return {
        'prediction':   prediction, # event-based
        'accuracy':     accuracy, # batch-based
        'loss':         loss, # batch-based
        'label':        label,# event-based
    }


##########################
## dataset format
##########################
from pattern_plot import ChToMCPMTIDs, ChToDataPMTIDs, ToDataChIDs, ToDataPMTIDs

# hardcoded binning
SpatialBinNumber      = 17
TimeBinNumber         = 100
TimeLower             = 0. # sample
TimeUpper             = 4000. # sample
TimeStep              = (TimeUpper - TimeLower) / float(TimeBinNumber)

def GetMap():
    '''
    return a dictionary as the mapping
    between pmtid and image index
    NOTE: it is hard coded
    :return:
    '''
    MapFilename        = '/home/ustc/WorkSpace/PandaX-4T/GhostTagging/maps/PMTIDtoImageIndex.json'
    dict_map = json.load(open(MapFilename))
    pmtid2voxel     = dict(zip(
        dict_map['data_pmt_id'],
        dict_map['image_index']
    ))
    return pmtid2voxel


def FormDataset_Data(pmthits, chids, hit_times, TotalNumberOfPMTs, NumEvents, InhibitedPMTs):
    '''
    Form a dataset from data
    :param pmthit:  (N) array
    :param chid:    (N) array
    :param TotalNumberOfPMTs: int
    :param NumEvents: int
    :return: [sparse_image, label]
        sparse_image: (N, 4) -> x, y, z, hits
        label:        (N)    -> mask (if this pixel has ghost hit > true hit, is 1)
    '''
    # get the map
    pmtid2voxel     = GetMap()
    # form image
    datasets        = []
    for ii in range(NumEvents):
        image       =   np.zeros((SpatialBinNumber, SpatialBinNumber, TimeBinNumber))
        # normalize hit_times
        hit_time_array      = np.asarray(hit_times[ii])
        hit_time_array      -= np.min(hit_time_array)
        for pmthit, pmtid, hittime in zip(
                pmthits[ii],
                ChToDataPMTIDs(chids[ii]),
                hit_time_array
        ):
            if (pmtid in InhibitedPMTs) or (pmtid > TotalNumberOfPMTs):
                continue
            x_index, y_index        = pmtid2voxel[pmtid]
            t_index                 = int((hittime-TimeLower)/TimeStep)
            if t_index>=TimeBinNumber:
                continue
            image[x_index, y_index, t_index]    += pmthit
        # dense to sparse
        inds            = np.where(image>0)
        sparse_image    = np.concatenate(
            (
              np.reshape(inds[0], (-1,1)),
              np.reshape(inds[1], (-1,1)),
              np.reshape(inds[2], (-1,1)),
              np.reshape(image[inds], (-1,1)),
            ),
            axis=1,
        )
        # create dummy label
        label           = np.zeros(sparse_image.shape[0])
        # append
        datasets.append(
            [
                sparse_image,
                label
            ]
        )
        # print
        if (ii+1)%100==0:
            print("==>> "+str(ii+1)+" events have been added!")
    return datasets


def FormDataset_MC(pmthits, mcids, hit_times, masks, TotalNumberOfPMTs, NumEvents, InhibitedPMTs):
    '''
    Form a dataset from MC sample
    :param pmthits:
    :param mcids:
    :param hit_times:
    :param masks:
    :param TotalNumberOfPMTs:
    :param NumEvents:
    :return: [sparse_image, label]
        sparse_image: (N, 4) -> x, y, z, hits
        label:        (N)    -> mask (if this pixel has ghost hit > true hit, is 1)
    '''
    # get the map
    pmtid2voxel = GetMap()
    # form image
    datasets = []
    for ii in range(NumEvents):
        image           = np.zeros((SpatialBinNumber, SpatialBinNumber, TimeBinNumber))
        image_masked    = np.zeros((SpatialBinNumber, SpatialBinNumber, TimeBinNumber))
        # normalize hit_times
        hit_time_array = np.asarray(hit_times[ii])
        hit_time_array -= np.min(hit_time_array)
        for pmthit, pmtid, hittime, mask in zip(
                pmthits[ii],
                ToDataPMTIDs(mcids[ii]),
                hit_time_array,
                masks[ii]
        ):
            if (pmtid in InhibitedPMTs) or (pmtid > TotalNumberOfPMTs):
                continue
            x_index, y_index = pmtid2voxel[pmtid]
            t_index = int((hittime - TimeLower) / TimeStep)
            if t_index>=TimeBinNumber:
                continue
            image[x_index, y_index, t_index] += pmthit
            if mask==0:
                image_masked[x_index, y_index, t_index] += pmthit
        # dense to sparse
        inds = np.where(image > 0)
        sparse_image = np.concatenate(
            (
                np.reshape(inds[0], (-1, 1)),
                np.reshape(inds[1], (-1, 1)),
                np.reshape(inds[2], (-1, 1)),
                np.reshape(image[inds], (-1, 1)),
            ),
            axis=1,
        )
        # form label
        truth_fractions     = image_masked[inds] / image[inds]
        inds2               = np.where(truth_fractions<0.5)[0]
        label               = np.zeros(sparse_image.shape[0])
        label[inds2]        = 1.
        label               = label.astype(np.int)
        # append
        datasets.append(
            [
                sparse_image,
                label
            ]
        )
        # print
        if (ii+1)%100==0:
            print("==>> "+str(ii+1)+" events have been added!")
    return datasets






