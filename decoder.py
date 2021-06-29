import sys
import csv
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt
from scipy import io
from scipy import stats
from mpl_toolkits import mplot3d
from Neural_Decoding.decoders import LSTMDecoder
from Neural_Decoding.preprocessing_funcs import get_spikes_with_history
from Neural_Decoding.metrics import get_R2
from Neural_Decoding.metrics import get_rho

"""
timepoints are each point at which data is collected, not constant across examples
timebins is an arbitrary value for training
one example is one reach in this case

target format
input: tensor of shape (# examples, # timepoints, # timebins * # features (neurons))
output: tensor of shape (# examples, # timepoints * # features (x, y, z position))
"""

"""Bins data for RNN modeling

Args: 
    input_data (tensor): tensor of shape (# examples, # timepoints, # features)
    bins_before (int): number of timebins before to consider in RNN
    bins_current (int): consider current bin or not

Returns:
    out: data in shape (# examples, # timepoints-# timebins, # timebins, # features)
"""
def bin_data(input_data, bins_before=4, bins_current=1):
    out = []
    num_bins = bins_before+bins_current
    for example in input_data:
        new_example = []
        for i in range(num_bins,len(example)):
            new_example.append(example[i-num_bins:i])
        out.append(new_example)
    return out

"""Reshapes binned neural data from 4d ragged tensor to 3d tensor for RNN processing

Args:
    input_data (4d ragged tensor): binned neural data

Returns:
    out: data in shape (# examples, # timepoints - # timebins, # timebins * # features)
"""
def reshape_input(input_data):
    out = []
    for example in input_data:
        new_example = []
        for t in example:
            new_example.append([feature for timebin in t for feature in timebin])
        out.append(new_example)
    return out

"""Reshapes behavioral data from 3d ragged tensor to 2d tensor for RNN processing

Args:
    behavioral_data (4d ragged tensor): behavioral data

Returns:
    out: data in shape (# examples, # timepoints * # features)
"""
def reshape_output(behavioral_data):
    out = []
    for example in behavioral_data:
        out.append([feature for timepoint in example for feature in timepoint])
    return out

# in ms
input_dt = 10
output_dt = 2
timepoints = 250
reach_start = 30

bins_before = 10
bins_current = 0
bins_after = 0

M1_rates = np.array(io.loadmat('./data/EN10_d1_imec0_M1_rates.mat')['rates']) # n x t x k
M1_rates = np.swapaxes(M1_rates,0,2) # k x t x n
behavioral_data = np.load('./EN10_d1_behavioral.npy',allow_pickle=True) 

# trim the firing rates to match behavioral data
trimmed_rates = []
for i in range(len(M1_rates)):
    if behavioral_data[i]:
        trimmed_rate = M1_rates[i][int(reach_start-bins_before):int(reach_start+(timepoints//(input_dt/output_dt))),:].tolist()
        trimmed_rates.append(trimmed_rate)

# remove reaches with not enough timepoints
behavioral_data = np.array([reach for reach in behavioral_data if reach], dtype=float)

# bin firing rates
binned_data = np.array(bin_data(trimmed_rates, bins_before, bins_current), dtype=float)

# reshape firing rates and behavioral data to fit lstm 
#X = np.array(reshape_input(binned_data))
#y = np.array(reshape_output(behavioral_data))
X = np.reshape(binned_data, (binned_data.shape[0], binned_data.shape[1], binned_data.shape[2]*binned_data.shape[3]))
#y = np.reshape(behavioral_data, (behavioral_data.shape[0], behavioral_data.shape[1]*behavioral_data.shape[2]))
y = behavioral_data
"""y_dims = y.shape
y = np.reshape(y, (y_dims[0], y_dims[1]//3, y_dims[1]//timepoints))

for i in range(len(y)):
    plt.figure(i)
    ax = plt.axes(projection='3d')
    train_coords = np.hsplit(y[i], 3)
    ax.plot3D(train_coords[0].flatten(), train_coords[1].flatten(), train_coords[2].flatten(), 'red')
    plt.show()"""



# split into training/testing sets BY REACHES
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=1)
X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

#Z-score "X" inputs. 
X_train_mean=np.nanmean(X_train,axis=0)
X_train_std=np.nanstd(X_train,axis=0)
X_train=(X_train-X_train_mean)/X_train_std
X_test=(X_test-X_train_mean)/X_train_std

#Zero-center outputs
# TODO: figure out how to zero center outputs properly
y_train_mean=np.mean(y_train,axis=0)
y_train=y_train-y_train_mean
y_test=y_test-y_train_mean
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]*y_train.shape[2]))
y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]*y_test.shape[2]))

#Declare model
model_lstm=LSTMDecoder(units=600,dropout=0,num_epochs=20)

#Fit model
model_lstm.fit(X_train,y_train)

#Get predictions
y_train_predicted_lstm=model_lstm.predict(X_train)
y_test_predicted_lstm=model_lstm.predict(X_test)

# unwrap y
y_dims = y_train_predicted_lstm.shape
y_train_predicted_lstm = np.reshape(y_train_predicted_lstm, (y_dims[0], y_dims[1]//3, y_dims[1]//timepoints))
y_train = np.reshape(y_train, (y_dims[0], y_dims[1]//3, y_dims[1]//timepoints))

for i in range(len(y_train)):
    plt.figure(i)
    ax = plt.axes(projection='3d')
    train_coords = np.hsplit(y_train[i], 3)
    predicted_coords = np.hsplit(y_train_predicted_lstm[i], 3)
    ax.plot3D(train_coords[0].flatten(), train_coords[1].flatten(), train_coords[2].flatten(), 'red')
    ax.plot3D(predicted_coords[0].flatten(), predicted_coords[1].flatten(), predicted_coords[2].flatten(), 'blue')
    plt.show()

"""#Get metric of fit
R2s_lstm=get_R2(y_test,y_test_predicted_lstm)
print('R2s:', R2s_lstm)

coords1 = np.hsplit(y_train[:100], 3)
coords2 = np.hsplit(y_train_predicted_lstm[:100], 3)

fig = plt.figure()

ax = plt.axes(projection='3d')
ax.plot3D(coords1[0].flatten(), coords1[1].flatten(), coords1[2].flatten(), 'red')
ax.scatter3D(coords2[0].flatten(), coords2[1].flatten(), coords2[2].flatten(), 'blue')

plt.show()"""