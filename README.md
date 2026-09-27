# Cory AIML339 Project

**This README files contains information on how to read the project.**

All of the official work for this project can be seen in the respective CNN or ViT Experiments .ipynb files.
the .py files are either the wrapper files for the CNN and ViT experiments or were for testing purposes.

The .csv files are for the bounding box information for the images.
.png files are created from the gradcam and attention rollout to visualise how the model makes decisions. 

the results folder contains the results of the experiments, they have the tuned weights for the models .pt, each model has a specific seed and an identifier to say if it has been trained on the augmented data or not.

The runs folder contains the results of each experiment of the model. 
 In the subfolder of each runs folder, it contains information on the following:
- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss


