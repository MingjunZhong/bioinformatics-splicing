# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 14:38:33 2016

@author: mzhong
"""
import pandas as pd
import csv
import numpy as np
from pandas import HDFStore
from sklearn.cross_validation import train_test_split

import sys
import os
import time
import theano
import theano.tensor as T
import lasagne

### this is the gene expression data set
def read_gene_data():
    filename = '/afs/inf.ed.ac.uk/user/m/mzhong/BME_DLUT/2016/python-code-for-bme/K562_gene_expression_by_histon_mark.txt'
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        #i = 0
        for i, row in enumerate(reader):
            #print(row)
            splitreading = row[0].split("\t")
            if i==0:
                # the first row is the column names; define the dataframe here
                data_gene = pd.DataFrame(columns=splitreading[1:])
            else:
                data_gene.loc[splitreading[0]] = map(float,splitreading[1:])
            #print('test')
        genedata = HDFStore('genedata.h5')
        genedata['genedata'] = data_gene
        genedata.close()

def load_gene_dataset():
    genedata = HDFStore('/afs/inf.ed.ac.uk/user/m/mzhong/BME_DLUT/2016/python-code-for-bme/genedata.h5')
    data = genedata['genedata']
    genedata.close()
    return data
   
# ##################### Build the neural network model #######################
# This script defines a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.
   
def build_cnn(input_var=None, lengthOfInputVector=None):
    # We'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 1, lengthOfInputVector),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers
    print network.output_shape
    
    network = lasagne.layers.DropoutLayer(network, p=0.2)
    print network.output_shape
    
    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(1,3),
            nonlinearity=lasagne.nonlinearities.linear,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.
    print network.output_shape    
    
#    network = lasagne.layers.DropoutLayer(network, p=0.5)
#    print network.output_shape    
    # Max-pooling layer of factor 2 in both dimensions:
#    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1,2))
#    print network.output_shape
    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=256, filter_size=(1,3),
            nonlinearity=lasagne.nonlinearities.rectify)
    print network.output_shape
#    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1,2))
#    print network.output_shape#
#    network = lasagne.layers.DropoutLayer(network, p=0.5)
#    print network.output_shape
    # A fully-connected layer of 256 units with 50 dropout on its inputs:
#    network = lasagne.layers.DenseLayer(
#            lasagne.layers.dropout(network, p=.3),
#            num_units=lengthOfInputVector*10,
#            nonlinearity=lasagne.nonlinearities.rectify)
#    print network.output_shape        
#    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
#    network = lasagne.layers.DenseLayer(
#            lasagne.layers.dropout(network, p=.3),
#            num_units=lengthOfInputVector*8,
#            nonlinearity=lasagne.nonlinearities.softmax)
#    print network.output_shape       
    
#    network = lasagne.layers.DenseLayer(
#            lasagne.layers.dropout(network, p=.0),
#            num_units=lengthOfInputVector*10,
#            nonlinearity=lasagne.nonlinearities.rectify)
#    print network.output_shape    
#    
#    network = lasagne.layers.DropoutLayer(network, p=0.5)
#    print network.output_shape
    
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.0),
            num_units=lengthOfInputVector*5,
            nonlinearity=lasagne.nonlinearities.rectify)
    print network.output_shape    
    
    network = lasagne.layers.DropoutLayer(network, p=0.5)
    print network.output_shape
    
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.0),
            num_units=lengthOfInputVector*2,
            nonlinearity=lasagne.nonlinearities.rectify)
    print network.output_shape    
    
    network = lasagne.layers.DropoutLayer(network, p=0.5)
    print network.output_shape
    
    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.0),
            num_units=1,
            nonlinearity=lasagne.nonlinearities.linear)
    print network.output_shape       
    return network
               
# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        #print('start_idx={}'.format(start_idx))
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        #print('excerpt={}'.format(excerpt))
        X = pd.DataFrame(inputs[excerpt]).values.astype(np.float32)
        y = pd.DataFrame(targets[excerpt]).values.astype(np.float32)
        yield X, y
        #yield inputs[excerpt].astype(np.float32), targets[excerpt].astype(np.float32)
        
# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main_training(key, X_train, y_train, X_val, y_val, geneStore, model='cnn', num_epochs=500):
    # load the dataset
    print("loading data...")
    #X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()     
    
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('input')
    target_var = T.fmatrix('targets')
    lengthOfInputVector = np.shape(X_train)[1]
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'cnn':
        network = build_cnn(input_var, lengthOfInputVector= lengthOfInputVector)
    else:
        print("Unrecognized model type {}".format(model))
        
    # Create a loss expression for traing, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    # loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)    
    # loss = loss.mean()
    loss = T.mean((prediction - target_var)**2)
    # we could add some weight decay as well here, see lasagne.regularization.
    
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
#    updates = lasagne.updates.nesterov_momentum(
#           loss, params, learning_rate=0.0005, momentum=0.9)
    updates = lasagne.updates.adam(loss, params)        
    # Create a loss expression for validation/testing.  The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    #test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
    #                                                       target_var)
    #test_loss = test_loss.mean()
    test_loss = T.mean((test_prediction - target_var)**2)
    # As a bonus, also create an expression for the classification accuracy:
    #test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
    #                 dtype=theano.config.floatX)

    # Compile a function performing a training step on mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    
    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_prediction])
    
    prediction_for_gene_expres = theano.function([input_var],prediction) 
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    minibatch_size = 100
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, minibatch_size, shuffle=True):
            inputs, targets = batch
            inputs = np.reshape(inputs,(inputs.shape[0],1,1,lengthOfInputVector))
            train_err += train_fn(inputs, targets)
            train_batches += 1
            
        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, minibatch_size, shuffle=False):
            inputs, targets = batch
            inputs = np.reshape(inputs,(inputs.shape[0],1,1,lengthOfInputVector))
            err, val_prediction = val_fn(inputs, targets)
            val_err += err
            val_batches += 1
            
        #print predicted_gene_expres
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        #print("  test data length:\t\t{0},{1}".format(len(predicted_gene_expres),X_val.shape[0]))
        
    # store data in HDFStore
    inputs = pd.DataFrame(X_val).values.astype(np.float32)    
    inputs = np.reshape(inputs,(inputs.shape[0],1,1,lengthOfInputVector))
    predicted_gene_expres = prediction_for_gene_expres(inputs)
    
    
    geneStore[key+'/prediction'] = pd.DataFrame(np.array(predicted_gene_expres).flatten())
    geneStore[key+'/true_expres'] = pd.DataFrame(np.array(y_val).flatten())
    geneStore[key+'/X'] = pd.DataFrame(X_val)
#    # After training, we compute and print the test error:
#    test_err = 0
#    test_batches = 0
#    for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
#        inputs, targets = batch
#        inputs = np.reshape(inputs,(inputs.shape[0],1,1,lengthOfInputVector))
#        err, test_prediction = val_fn(inputs, targets)
#        test_err += err
#        test_batches += 1
#    print("Final results:")
#    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))

#    return predicted_gene_expres, y_val
    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
        
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

def get_input_output(data):
    X = data[data.columns[1:]].values
    y = data[data.columns[0]].values
    return X, y

def main():
    # the loaded data is a DataFrame
    genedata = load_gene_dataset()
    
    # randomly split the dataset to three folds
    # this code should be improved in the future
    kfold = 3.0
    data_kfold = {}
    train, fold1 = train_test_split(genedata, test_size=1/kfold)
    data_kfold['fold1'] = fold1
    fold3, fold2 = train_test_split(train, test_size=0.5)
    data_kfold['fold2'] = fold2
    data_kfold['fold3'] = fold3
    
    # now we want to train a network for each fold
    # store the results in h5 file
    geneStore = HDFStore('predGeneExp1.h5')
    for i, key in enumerate(data_kfold):
        print(key)
        test_data = data_kfold[key]
        X_val, y_val = get_input_output(test_data)
        keys = data_kfold.keys()
        keys.remove(key)
        training_data = pd.concat([data_kfold[keys[0]],data_kfold[keys[1]]])
        X_train, y_train = get_input_output(training_data)
        print(keys)
        # use the these data to train the network
        main_training(key, X_train, y_train, X_val, y_val, geneStore)
   
    # the h5 must be closed after using
    geneStore.close()

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on gene data using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)