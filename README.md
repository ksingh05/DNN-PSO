# DNN-PSO

This repository has details of the code used for "Accelerated Optimization of Curvilinearly Stiffened Panels using Deep Learning" (currently under review for Thin-walled Structures)

`model.h5` has the weights of the trained DNN Keras model.

`data_set` folder has the training and testing data. 

`model_accuracy.py` can be executed using python. It will print the accuracy of the DNN over the training and testing dataset.

`shape_opt` folder has the python code `run_shape_opt_using_DNN.py` which can do shape optimization of curvilinearly stiffened panel using trained DNN. 

`size_opt` folder has an example MSC.NASTRAN `.bdf` file for conducting the size optimization of the curvilinearly stiffened panel.
