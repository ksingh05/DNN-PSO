# Accelerated Optimization of Curvilinearly Stiffened Panels using Deep Learning

This repository has details of the code used for "Accelerated Optimization of Curvilinearly Stiffened Panels using Deep Learning" (currently under review for Thin-walled Structures)

`model.h5` has the weights of the trained DNN Keras model.

`data_set` folder has the training and testing data. 

`model_accuracy.py` can be executed using python. It will print the accuracy of the DNN over the training and testing dataset.

`shape_opt` folder has the python code `run_shape_opt_using_DNN.py` which can do shape optimization of curvilinearly stiffened panel using trained DNN. 

`size_opt` folder has an example MSC.NASTRAN `.bdf` file for conducting the size optimization of the curvilinearly stiffened panel.

## Example Output of `model_accuracy.py`

```
Training Accuracy: 97.34% to predict withiin 10% of actual value
Testing Accuracy: 95.84% to predict withiin 10% of actual value
```

## Example Output of `run_shape_opt_using_DNN.py`

```
Loaded model from disk !!

Running Optimization ....

Shape Optimal Configuration:
[0.168  0.581  0.006  0.081  0.67  -0.026]
```
The research paper would be made public once its accepted.
