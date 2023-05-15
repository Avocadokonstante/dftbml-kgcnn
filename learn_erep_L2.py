import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
import pandas as pd
import numpy as np
from kgcnn.data.qm import QMDataset
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.gather import GatherNodes
from kgcnn.layers.modules import ExpandDims
from kgcnn.layers.pooling import PoolingNodes
from sklearn.model_selection import KFold
from kgcnn.utils.plots import plot_train_test_loss, plot_predict_true
from kgcnn.training.scheduler import LinearWarmupLinearLearningRateScheduler
from layers import LParameterAB, ErepForLab
import os
os.environ['BABEL_DATADIR'] = "/usr/local/run/openbabel-2.4.1/"

ks = tf.keras

from utils_kgcnn_plots import visualization_kgcnn
vis = visualization_kgcnn("L2")
#vis.write_experiment_description()

from utils_read_write import utils_read_write
rw = utils_read_write()

path_to_geo = "/home/keckhardt/DATA/02_ANI/Coords.xyz"
path_to_energy = "/home/keckhardt/DATA/02_ANI/Atomization_DIFF.npy"

dataset = QMDataset(data_directory="", file_name=path_to_geo, dataset_name="CC")
dataset.prepare_data()
dataset.read_in_memory()
dataset.assign_property("graph_labels", (np.load(path_to_energy)).tolist())

dataset.map_list("set_range",
                 max_distance=100000, max_neighbours=100000, exclusive=False, self_loops=False)  # Simply all to all
labels = np.array(dataset.obtain_property("graph_labels"))
print("Energies shape:", labels.shape)
print("Range for first molecule: ", dataset.obtain_property("range_attributes")[0].shape)


def make_model_L_ab():
    """Using Klopmann Ohno formular for L_ab"""
    atom_number_input = ks.layers.Input(shape=(None, ), dtype="int64", ragged=True)
    charge_q_input = ks.layers.Input(shape=(None, ), dtype="int64", ragged=True)
    pair_index_input = ks.layers.Input(shape=(None, 2), dtype="int64", ragged=True)
    distance_input = ks.layers.Input(shape=(None, 1), dtype="float32", ragged=True)

    charge_vec = ExpandDims(axis=-1)(charge_q_input)
    charges_per_bond = GatherNodes()([charge_vec, pair_index_input])
    atoms_vec = ExpandDims(axis=-1)(atom_number_input)
    atom_number_per_bond = GatherNodes()([atoms_vec, pair_index_input])

    gamma_per_bond = LParameterAB()(atom_number_per_bond)
    erep_per_bond = ErepForLab(add_eps=False)([charges_per_bond, distance_input, gamma_per_bond])
    e_total = PoolingNodes(pooling_method="sum")(erep_per_bond)

    model = ks.models.Model(inputs=[atom_number_input, charge_q_input, pair_index_input, distance_input],
                            outputs=e_total)
    return model

inputs = [
    {"shape": [None, ], "name": "node_number", "dtype": "int64", "ragged": True},
    {"shape": [None, ], "name": "node_number", "dtype": "int64", "ragged": True},
    {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
    {"shape": [None, 1], "name": "range_attributes", "dtype": "int64", "ragged": True}
]

# Test Split
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Training on splits
# Randomize labels like that, so that with the random indices we can connect the datapoints in different files via the inidices
random_all = np.arange(len(labels)) 
np.random.shuffle(random_all)
# Size of Training + Test
#random = random_all[:15000]
random = random_all
dataset = dataset[random]
labels = labels[random]
history_list, test_indices_list = [], []
training = int(0.8*dataset.length)
print(len(labels))
print(dataset[:training].length)
print(dataset[training:].length)

# Training on splits
execute_folds = 5
history_list, test_indices_list = [], []
for i, (train_index, test_index) in enumerate(kf.split(X=np.arange(dataset.length)[:, None])):
    if i >= execute_folds:
        continue
    
    xtrain, ytrain = dataset[train_index].tensor(inputs), labels[train_index]
    xtest, ytest = dataset[test_index].tensor(inputs), labels[test_index]

    model = make_model_L_ab()
    print(model.summary())
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-1),
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

prediction_test =  dataset[training:].tensor(inputs)
pred_energies = model.predict(prediction_test)
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
vis.set_original_test(labels[training:])
vis.set_history(history_list)
vis.set_train_mean(pred_mean)
vis.set_colors(colors[training:])

energies_dftb = np.load("/home/keckhardt/DATA/02_ANI/Atomization_Energy_DFTB_El.npy")
energies_dftb = energies_dftb[random]
energies_dftb = energies_dftb[training:]
vis.set_energies_dftb_test(energies_dftb.flatten())

energies_cc = np.load("/home/keckhardt/DATA/02_ANI/Atomization_Energy_CC.npy")
energies_cc = energies_cc[random]
energies_cc = energies_cc[training:]
vis.set_energies_cc_test(energies_cc.flatten())




vis.create_complete_visualisation()





np.save("model_weights.npy", model.get_weights()[0])
np.save(os.path.join(vis.get_directory(), "{}_{}_model_weights.npy".format(vis.get_directory(), vis.get_set_id())), model.get_weights()[0])
