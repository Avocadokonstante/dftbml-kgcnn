import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as cm
import os
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap
from datetime import datetime
import time

class visualization_kgcnn:
    def __init__(self, network_name):
        directory_name, set_id = self.get_time()
        self.directory_name = directory_name
        self.set_id = set_id
        self.network_name = network_name
        
        #I'm using domain specific naming
        self.predicted_energies_test = [] #prediction
        self.original_energies_test = [] #labels
        self.labels = []
        self.history = []
        self.mean_train = 0
        
        self.colors = []

        
        #Only needed because of delta learning approach
        self.energies_atomization_cc_test = [] 
        self.energies_atomization_dftb_test = []
        self.corrected_atomization_dftb = []
        
    
                
    #-----------------------ALLE GRAFEN AUTOMATISIERT------------------------
    def create_complete_visualisation(self):
        print("Plots are created")
        self.make_directory(self.directory_name)
        #Prediction vs Labels on Test set
        self.plot_predict_true(self.predicted_energies_test, self.original_energies_test, "H", 
                               self.network_name, self.directory_name, self.set_id, "Predicted repulsive energy" )
        
        self.plot_predict_true_colors(self.predicted_energies_test, self.original_energies_test, self.colors, "H", 
                               self.network_name, self.directory_name, self.set_id, "Predicted repulsive energy" )

        #Prediction vs Labels on corrected DFTB vs DFT
        self.corrected_atomization_dftb = self.energies_atomization_dftb_test + self.predicted_energies_test
        self.plot_predict_true(self.corrected_atomization_dftb, self.energies_atomization_cc_test, "H", 
                               self.network_name, self.directory_name, self.set_id, "Predicted modified energy DFTB" )

        self.plot_predict_true_colors(self.corrected_atomization_dftb, self.energies_atomization_cc_test, self.colors, "H", 
                               self.network_name, self.directory_name, self.set_id, "Predicted modified energy DFTB" )

        #Compare to Lilienfeld
        self.compare_to_lilienfeld()
        
        #Compare distributions of atomization energies (3Hist in one Diagram)
        self.plot_histogram_w_3(self.directory_name, self.set_id, self.network_name, "Distribution of modified energies", "H", 
                                self.energies_atomization_dftb_test, self.corrected_atomization_dftb, self.energies_atomization_cc_test,["dftb", "dftb + prediction" ,"cc"], 100)

        #Distribution of Model Result
        self.plot_histogram_w_1(self.directory_name, self.set_id, self.network_name, "Distribution of predicted repulsive energy", "H", 
                                self.predicted_energies_test,["predicted delta"], 100)

        self.plot_histogram_w_1(self.directory_name, self.set_id, self.network_name, "Distribution of predicted modified energy DFTB", "H", 
                                self.corrected_atomization_dftb,["dftb_corr"], 100)
        
        self.plot_train_test_loss(self.history, loss_name="loss", val_loss_name="val_loss", model_name=self.network_name, data_unit="H", 
                                  title="Train loss", filepath=self.directory_name, file_name=self.set_id, mean=self.mean_train)
    #-----------------------NICHT PLOT FUNKTIONEN----------------------------
    def get_time(self):
        now = datetime.now() # current date and time
        return now.strftime("%Y%m%d"), now.strftime("%Y%m%d_%H%M%S")

    def make_directory(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)

    def write_experiment_description(self):
        print("Please write an experiment description:")
        description = str(input())
        if (not description):
            return
        
        self.make_directory(self.directory_name)
        with open('%s/%s_%s_Experiment_Description.txt'%(self.directory_name, self.set_id, self.network_name), 'w') as f:
            f.write(description)
            f.close()
    
    def calculate_mean_absolute_error(self, prediction, labels):
        mean_error = (np.abs((np.subtract(prediction, labels)))).mean()
        return mean_error
    
    def calculate_mean(list):
        return list.mean()
    
    def H_to_kj(self, mean_error):
        H_to_kj = 4.3597482*(10**-21)
        frac_mol = 1 / 6.0221415*(10**23)
        return mean_error * H_to_kj, mean_error * (H_to_kj * frac_mol)
    
    def compare_to_lilienfeld(self):
        mean_error = self.calculate_mean_absolute_error(self.corrected_atomization_dftb, self.energies_atomization_cc_test)
        e_kj, e_kj_mol = self.H_to_kj(mean_error)
        print("Mean_Abs in [H] = " + str(mean_error))
        print("Mean_Abs in [kj] = " + str(e_kj), "\n Mean_Abs in [kj/mol] = " + str(e_kj_mol) +"\n"+ "Compare to Anatole error of 9.9kj/mol")

    #-----------------------PLOT FUNKTIONEN-----------------------------------
    def plot_train_test_loss(self, histories: list, loss_name: str = None,
                            val_loss_name: str = None, data_unit: str = "", model_name: str = "",
                            filepath: str = None, file_name: str = "", title: str = "",
                            mean: float = None):
        r"""Plot training curves for a list of fit results in form of keras history objects. This means, training-
        and test-loss is plotted vs. epochs for all splits.
        Args:
            histories (list): List of :obj:`tf.keras.callbacks.History()` objects.
            loss_name (str): Which loss or metric to pick from history for plotting. Default is "loss".
            val_loss_name (str): Which validation loss or metric to pick from history for plotting. Default is "val_loss".
            data_unit (str): Unit of the loss. Default is "".
            model_name (str): Name of the model. Default is "".
            filepath (str): Full path where to save plot to, without the name of the file. Default is "".
            file_name (str): File name base. Model name and dataset will be added to the name. Default is "".
            dataset_name (str): Name of the dataset which was fitted to. Default is "".
        Returns:
            matplotlib.pyplot.figure: Figure of the training curves.
        """
        # We assume multiple fits as in KFold.
        if data_unit is None:
            data_unit = ""
        if loss_name is None:
            loss_name = [x for x in list(histories[0].history.keys()) if "val_" not in x]
        if val_loss_name is None:
            val_loss_name = [x for x in list(histories[0].history.keys()) if "val_" in x]

        if not isinstance(loss_name, list):
            loss_name = [loss_name]
        if not isinstance(val_loss_name, list):
            val_loss_name = [val_loss_name]

        train_loss = []
        for x in loss_name:
            loss = np.array([np.array(hist.history[x][5:]) for hist in histories])
            train_loss.append(loss)
        val_loss = []
        for x in val_loss_name:
            loss = np.array([hist.history[x][5:] for hist in histories])
            val_loss.append(loss)

        fig = plt.figure()
        for i, x in enumerate(train_loss):
            vp = plt.plot(np.arange(len(np.mean(x, axis=0))), np.mean(x, axis=0), alpha=0.85, label=loss_name[i])
            plt.fill_between(np.arange(len(np.mean(x, axis=0))),
                            np.mean(x, axis=0) - np.std(x, axis=0),
                            np.mean(x, axis=0) + np.std(x, axis=0), color=vp[0].get_color(), alpha=0.2
                            )
        for i, y in enumerate(val_loss):
            val_step = len(train_loss[i][0]) / len(val_loss[i][0])
            vp = plt.plot(np.arange(len(np.mean(y, axis=0))) * val_step + val_step, np.mean(y, axis=0), alpha=0.85,
                        label=val_loss_name[i])
            plt.fill_between(np.arange(len(np.mean(y, axis=0))) * val_step + val_step,
                            np.mean(y, axis=0) - np.std(y, axis=0),
                            np.mean(y, axis=0) + np.std(y, axis=0), color=vp[0].get_color(), alpha=0.2
                            )
            plt.xlim(xmin=5)
            #plt.plot(train_loss[i][:], r"{0}: {1:0.4f} $\pm$ {2:0.4f} ".format(
            #                val_loss_name[i], np.mean(y, axis=0)[-1], np.std(y, axis=0)[-1]) + data_unit, color=vp[0].get_color())
            plt.scatter([len(train_loss[i][0])], [np.mean(y, axis=0)[-1]],
                        label=r"{0}: {1:0.4f} $\pm$ {2:0.4f} ".format(
                            val_loss_name[i], np.mean(y, axis=0)[-1], np.std(y, axis=0)[-1]) + data_unit, color=vp[0].get_color()
                        )
            
        plt.axhline(mean, color='r')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(title + " for model:" + model_name)
        plt.legend(loc='upper right', fontsize='small')
        print("Saving loss function graph")
        if filepath is not None:
            self.make_directory(filepath)
            plt.savefig(os.path.join(filepath, file_name + "_" + model_name + "_" + title.replace(" ", "_")))
        plt.close()

    def plot_histogram_w_1(self, directory, filename, network_name, description, unit, prediction, lables, gap):
        paper_rc = {'lines.linewidth': 2.5}
        sns.set_theme(font_scale=1.25, palette=sns.color_palette("colorblind").as_hex(), rc=paper_rc)
        sns.set_style('white')
        bottom, top = self.calculate_linespace(prediction)
        bins = np.linspace(bottom, top, gap)
        plt.hist(prediction, bins=bins, alpha=0.3, label=lables[0])  # the blue one
        plt.title(description)
        plt.xlabel(unit)
        plt.ylabel('Count')

        png = 'png'
        plt.tight_layout()
        print("Saving Histogram")
        plt.savefig(os.path.join(directory, filename + "_" + network_name + "_" + description.replace(" ", "_") ))
        plt.close()

    def plot_histogram_w_2(self, directory, filename, network_name, description, unit, dftb, dft, lables, gap):
        bottom, top = self.calculate_linespace(dftb)
        bins = np.linspace(bottom, top, gap)
        plt.hist(dftb, bins, alpha=0.5, label='dftb')
        plt.hist(dft, bins, alpha=0.5, label='dft')
        plt.title(description)
        plt.xlabel(unit)
        plt.ylabel('Count')
        png = 'png'
        plt.legend(loc='upper right')
        print("Saving Histogram")
        plt.savefig(os.path.join(directory, filename + "_" + network_name + "_" + description.replace(" ", "_") ))
        plt.close()
        
    def plot_histogram_w_3(self, directory, filename, network_name, description, unit, dftb, dftb_corr, dft, lables, gap):
        bottom, top = self.calculate_linespace(dftb)
        bins = np.linspace(bottom, top, gap)
        plt.hist(dftb, bins=bins, alpha=0.3, label=lables[0])
        plt.hist(dft, bins=bins, alpha=0.3, label=lables[2])
        plt.hist(dftb_corr, bins=bins, alpha=0.3, label=lables[1])
        #plt.title(description)
        plt.title("Comparison of Modified Energies")
        plt.xlabel(unit)
        plt.ylabel('Count')
        png = 'png'
        plt.legend(loc='upper right')
        print("Saving Histogram")
        plt.savefig(os.path.join(directory, filename + "_" + network_name + "_" + description.replace(" ", "_") ))
        plt.close()

    def calculate_linespace(self, axis):
        smallest_value = np.amin(axis, axis=None, out=None)
        largest_value = np.amax(axis, axis=None, out=None)
        return smallest_value, largest_value



    def plot_predict_true(self, y_predict, y_true, data_unit: list = None, model_name: str = "",
                        filepath: str = None, file_name: str = "", dataset_name: str = "", target_names: list = None
                        ):
        r"""Make a scatter plot of predicted versus actual targets. Not for k-splits.
        Args:
            y_predict (np.ndarray): Numpy array of shape `(N_samples, n_targets)` or `(N_samples, )`.
            y_true (np.ndarray): Numpy array of shape `(N_samples, n_targets)` or `(N_samples, )`.
            data_unit (list): String or list of string that matches `n_targets`. Name of the data's unit.
            model_name (str): Name of the model. Default is "".
            filepath (str): Full path where to save plot to, without the name of the file. Default is "".
            file_name (str): File name base. Model name and dataset will be added to the name. Default is "".
            dataset_name (str): Name of the dataset which was fitted to. Default is "".
            target_names (list): String or list of string that matches `n_targets`. Name of the targets.
        Returns:
            matplotlib.pyplot.figure: Figure of the scatter plot.
        """
        if len(y_predict.shape) == 1:
            y_predict = np.expand_dims(y_predict, axis=-1)
        if len(y_true.shape) == 1:
            y_true = np.expand_dims(y_true, axis=-1)
        num_targets = y_true.shape[1]

        if data_unit is None:
            data_unit = ""
        if isinstance(data_unit, str):
            data_unit = [data_unit]*num_targets
        if len(data_unit) != num_targets:
            print("WARNING:kgcnn: Targets do not match units for plot.")
        if target_names is None:
            target_names = ""
        if isinstance(target_names, str):
            target_names = [target_names]*num_targets
        if len(target_names) != num_targets:
            print("WARNING:kgcnn: Targets do not match names for plot.")

        fig = plt.figure()
        for i in range(num_targets):
            mae_valid = np.mean(np.abs(y_true[:, i] - y_predict[:, i]))
            plt.scatter(y_predict[:, i], y_true[:, i], alpha=0.3,
                        label=target_names[i] + " MAE: {0:0.4f} ".format(mae_valid) + "[" + data_unit[i] + "]")
   
        
        plt.plot(np.arange(np.amin(y_true), np.amax(y_true), 0.05),
                np.arange(np.amin(y_true), np.amax(y_true), 0.05), color='red')
        plt.xlabel('Prediction in H')
        plt.ylabel('Reference in H')
        #plt.title(dataset_name)
        plt.title("Prediciton vs. Reference")
        plt.legend(loc='upper left', fontsize='x-large')
        print("Saving predict vs true plot")
        if filepath is not None:
            plt.savefig(os.path.join(filepath, file_name + "_" + model_name + "_" + dataset_name.replace(" ", "_") ))
        #plt.show()
        #return fig
        plt.close()

    def plot_predict_true_colors(self, y_predict, y_true, colors, data_unit: list = None, model_name: str = "",
                        filepath: str = None, file_name: str = "", dataset_name: str = "", target_names: list = None
                        ):
        r"""Make a scatter plot of predicted versus actual targets. Not for k-splits.
        Args:
            y_predict (np.ndarray): Numpy array of shape `(N_samples, n_targets)` or `(N_samples, )`.
            y_true (np.ndarray): Numpy array of shape `(N_samples, n_targets)` or `(N_samples, )`.
            data_unit (list): String or list of string that matches `n_targets`. Name of the data's unit.
            model_name (str): Name of the model. Default is "".
            filepath (str): Full path where to save plot to, without the name of the file. Default is "".
            file_name (str): File name base. Model name and dataset will be added to the name. Default is "".
            dataset_name (str): Name of the dataset which was fitted to. Default is "".
            target_names (list): String or list of string that matches `n_targets`. Name of the targets.
        Returns:
            matplotlib.pyplot.figure: Figure of the scatter plot.
        """
        if len(y_predict.shape) == 1:
            y_predict = np.expand_dims(y_predict, axis=-1)
        if len(y_true.shape) == 1:
            y_true = np.expand_dims(y_true, axis=-1)
        if len(colors.shape) == 1:
            colors = np.expand_dims(colors, axis=-1)
        num_targets = y_true.shape[1]

        if data_unit is None:
            data_unit = ""
        if isinstance(data_unit, str):
            data_unit = [data_unit]*num_targets
        if len(data_unit) != num_targets:
            print("WARNING:kgcnn: Targets do not match units for plot.")
        if target_names is None:
            target_names = ""
        if isinstance(target_names, str):
            target_names = [target_names]*num_targets
        if len(target_names) != num_targets:
            print("WARNING:kgcnn: Targets do not match names for plot.")

        cmap = sns.color_palette("coolwarm", as_cmap=True)
        fig = plt.figure()
        for i in range(num_targets):
            mae_valid = np.mean(np.abs(y_true[:, i] - y_predict[:, i]))
            plt.scatter(y_predict[:, i], y_true[:, i], c=colors[:, i],cmap=cmap, alpha=0.3,
                        label=target_names[i] + " MAE: {0:0.4f} ".format(mae_valid) + "[" + data_unit[i] + "]")
        plt.plot(np.arange(np.amin(y_true), np.amax(y_true), 0.05),
                np.arange(np.amin(y_true), np.amax(y_true), 0.05), color='red')
        plt.xlabel('Predicted')
        plt.ylabel('Original')
        plt.title("Prediction of " + model_name + " for " + dataset_name)
        plt.legend(loc='upper left', fontsize='x-large')
        print("Saving predict vs true plot")
        if filepath is not None:
            plt.savefig(os.path.join(filepath, file_name + "_" + model_name + "_c" + "_" + dataset_name.replace(" ", "_") ))
        #plt.show()
        #return fig
        plt.close()

            
            
#-----------------------SETTER FOR VISUALIZATION-------------------------------------------------------------
    def set_prediction_test(self, prediction):
        self.predicted_energies_test = prediction
        
    def set_original_test(self, labels):
        self.original_energies_test = labels
        
    def set_labels(self, labels):
        self.labels = labels
    
    def set_energies_dftb_test(self, dftb_energies):
        self.energies_atomization_dftb_test = dftb_energies
        
    def set_energies_cc_test(self, cc_energies):
        self.energies_atomization_cc_test = cc_energies
        
    def set_history(self, history):
        self.history = history
        
    def set_train_mean(self, mean):
        self.mean_train = mean
        
    def set_colors(self, colors):
        self.colors = colors
        
    def get_directory(self):
        return self.directory_name
    
    def get_set_id(self):
        return self.set_id
