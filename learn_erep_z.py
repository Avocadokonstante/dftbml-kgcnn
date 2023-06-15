import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
import pandas as pd
import numpy as np
from kgcnn.data.qm import QMDataset
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodes
from kgcnn.layers.modules import ExpandDims
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.message import MessagePassingBase
from kgcnn.layers.conv import GraphSageNodeLayer
from kgcnn.layers.modules import Dense, LazyConcatenate
from sklearn.model_selection import KFold
from kgcnn.utils.plots import plot_train_test_loss, plot_predict_true
from kgcnn.training.scheduler import LinearWarmupLinearLearningRateScheduler
from layers import LParameterAB, ErepForLab
from kgcnn.layers.pooling import PoolingLocalMessages, PoolingNodes
from kgcnn.layers.gather import GatherNodesSelection


import os
os.environ['BABEL_DATADIR'] = "/usr/local/run/openbabel-2.4.1/"

ks = tf.keras

from utils_kgcnn_plots import visualization_kgcnn
vis = visualization_kgcnn("z")
#vis.write_experiment_description()

from utils_read_write import utils_read_write
rw = utils_read_write()

#path_to_geo = "/home/keckhardt/DATA/02_ANI/Coords_8.xyz"
#path_to_energy = "/home/keckhardt/DATA/02_ANI/Atomization_DIFF_8.npy"

path_to_geo = "/home/keckhardt/DFTBML/hiwi-dftbml/kgcnn/all_geo.xyz"
path_to_energy = "/home/keckhardt/DFTBML/hiwi-dftbml/kgcnn/energies.npy"

dataset = QMDataset(data_directory="/home/keckhardt/DFTBML/hiwi-dftbml/kgcnn/", file_name=path_to_geo, dataset_name="CC")
dataset.prepare_data()
dataset.read_in_memory_xyz()
dataset.assign_property("graph_labels", (np.load(path_to_energy)).tolist())

dataset.map_list("set_range",
                 max_distance=100000, max_neighbours=100000, exclusive=False, self_loops=False)  # Simply all to all

# dataset = QMDataset(data_directory="/home/keckhardt/DFTBML/hiwi-dftbml/kgcnn/", dataset_name="CC")
# dataset.load()

labels = np.array(dataset.obtain_property("graph_labels"))
print("Energies shape:", labels.shape)
print("Range for first molecule: ", dataset.obtain_property("range_attributes")[0].shape)
dataset.save()





#print(dataset.obtain_property("node_symbol"))
print(dataset)

#I tried to create a simple message passing network with the library and am unsure if this is correct
def make_model():
    atom_number_input = ks.layers.Input(shape=(None, ), dtype="int64", ragged=True)
    pair_index_input = ks.layers.Input(shape=(None, 2), dtype="int64", ragged=True)
    distance_input = ks.layers.Input(shape=(None, 1), dtype="float32", ragged=True)
    atoms_vec = ExpandDims(axis=-1)(atom_number_input)
    #dist_vec = ExpandDims(axis=-1)(distance_input)

    #atom_number_per_bond = GatherNodes()([atoms_vec, pair_index_input])

    #erep_per_bond = GraphSageNodeLayer(units=1)([atom_number_input ,pair_index_input])
    
   
    #atom_number_input = ExpandDims(atom_number_input, axis=-1)
    #charge_q_input = ExpandDims(charge_q_input, axis=-1)
    atom_number_f = tf.cast(atoms_vec, dtype=tf.float32)
    
    #test = LazyConcatenate()([dist_vec, atom_number_f])
    n_in, n_out = GatherNodesSelection([0,1])([atoms_vec, pair_index_input])
    n_f = tf.cast(n_out, dtype=tf.float32)

    n = LazyConcatenate(axis=-1)([n_f, distance_input])
    dense_layer = Dense(10, activation="relu")
    node_message=dense_layer(n)
    node_updates = PoolingLocalMessages()([atom_number_input, node_message, pair_index_input])
    
    n_node_updates = LazyConcatenate()([atom_number_f, node_updates])
    erep_per_bond = Dense(1)(n_node_updates)
    e_total = PoolingNodes(pooling_method="sum")(erep_per_bond)

    model = ks.models.Model(inputs=[atom_number_input, pair_index_input, distance_input],
                            outputs=e_total)
    return model

inputs = [
    {"shape": [None], "name": "node_number", "dtype": "int64", "ragged": True},
    {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
    {"shape": [None, 1], "name": "range_attributes", "dtype": "float32", "ragged": True}
]

# Test Split
kf = KFold(n_splits=4, shuffle=True, random_state=0)

# Training on splits
# Randomize labels like that, so that with the random indices we can connect the datapoints in different files via the inidices
random_all = np.arange(len(labels)) 
np.random.seed(52)
np.random.shuffle(random_all)
# # Size of Training + Test
random = random_all[:30000]
#random = random_all
dataset = dataset[random]
print(len(dataset))
labels = labels[random]
#history_list, test_indices_list = [], []
training = int(0.8*dataset.length)
# print(len(labels))
test_index = random[training:]
train_index= random[:training]

dataset_train = dataset[training:]
dataset_test = dataset[:training]
labels_test = labels[training:]
labels_train = labels[:training]

# Training on splits
execute_folds = 4
history_list, test_indices_list = [], []
for i, (train_index, test_index) in enumerate(kf.split(X=np.arange(dataset_train.length)[:, None])):
    if i >= execute_folds:
        continue
    
    xtrain, ytrain = dataset_train[train_index].tensor(inputs), labels_train[train_index]
    xtest, ytest = dataset_train[test_index].tensor(inputs), labels_train[test_index]
    
    model = make_model()
    #exit()
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                  metrics=["mean_absolute_error"],
                  loss="mean_squared_error"
                  )
    hist = model.fit(xtrain, ytrain,
                     batch_size=128,
                     validation_data=(xtest, ytest),
                     epochs=20,
                     callbacks=[LinearWarmupLinearLearningRateScheduler(
                         learning_rate_start=1e-1,
                         learning_rate_stop=1e-3,
                         epo_warmup=1,
                         epo=10,
                         steps_per_epoch=2872
                     )],
                     validation_freq=1
                     )

    # Get loss from history
    history_list.append(hist)
    test_indices_list.append([train_index, test_index])

#--TEST ON NEW DATAPOINTS---------------------------------

prediction_test = dataset[test_index].tensor(inputs)
pred_energies = model.predict(prediction_test)
#pred_energies = scaler.inverse_transform(pred_energies)
#mean_energies= scaler.inverse_transform(labels[:training])
#mean = mean_energies.flatten().mean()
mean_energies= (labels[test_index])
mean = labels[:training].flatten().mean()
pred_mean = ((np.subtract(np.load(path_to_energy), mean))*(np.subtract(np.load(path_to_energy), mean))).mean()



#---VISUALISIERUNG----------------------------------------
c, e = rw.readXYZs(path_to_geo)

def get_list_lengths(lists):
  # Initialize an empty list to store the lengths
  lengths = []

  # Loop through each list in the input list
  for l in lists:
    # Calculate the length of the list
    length = len(l)
    # Add the length to the list of lengths
    lengths.append(int(length))

  # Return the list of lengths
  return np.array(lengths)

colors = get_list_lengths(e)
colors = colors[random]

vis.set_labels(labels)
vis.set_prediction_test(pred_energies.flatten())
vis.set_original_test(labels[test_index])
vis.set_history(history_list)
vis.set_train_mean(pred_mean)
vis.set_colors(colors[test_index])

energies_dftb = np.load("/home/keckhardt/DATA/02_ANI/Atomization_Energy_DFTB_El.npy")
energies_dftb = energies_dftb[test_index]
vis.set_energies_dftb_test(energies_dftb.flatten())

energies_cc = np.load("/home/keckhardt/DATA/02_ANI/Atomization_Energy_CC.npy")
energies_cc = energies_cc[test_index]
vis.set_energies_cc_test(energies_cc.flatten())




vis.create_complete_visualisation()





np.save("model_weights.npy", model.get_weights()[0])
np.save(os.path.join(vis.get_directory(), "{}_{}_model_weights.npy".format(vis.get_directory(), vis.get_set_id())), model.get_weights()[0])
