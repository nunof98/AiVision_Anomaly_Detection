# AiVision_Anomaly_Detection
This project proposes a solution for anomaly detection in metalic parts using deep learning and autoencoders.

## Method
This method uses SSIM residual maps to calculate the loss between the original image and the autoencoder reconstruction, and then selecting the highest intensity pixels to decide, based on the area of the blobs, if the part is defective or not.

## Datasets
This project has been tested on [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/) and on a dataset I created with metalic parts from [ETMA metal parts](https://etmametalparts.com/en/home/)

## Models
There is a total of 7 models based on the Convolutional Auto-Encoder (CAE) architecture implemented in this project:

* mvtecCAE is the model implemented in the MVTec Paper
* baselineCAE implemented in: https://github.com/natasasdj/anomalyDetection
* inceptionCAE implemented in: https://github.com/natasasdj/anomalyDetection
* resnetCAE implemented in: https://arxiv.org/pdf/1606.08921.pdf
* skipCAE implemented in: https://arxiv.org/pdf/1606.08921.pdf
* myCAE is a model I created which is similar to the mvtecCAE
* myCAE_optuna is the same as myCAE, but with hyperparameters sugested by [Optuna](https://optuna.org/)

## Usage
### Training (`train.py`)
The autoencoders train exclusively on defect-free images and learns to reconstruct (predict) defect-free training samples.
To use this script, you must enter the path to where the training dataset is in your machine and it's ready to run.
#### Note:
This code trains the models for 100 epochs by default.
It also saves the training history, the images of loss and metric functions throughout training and a plot of the model

### Testing (`test.py`)
This script gets the first model on the saved_models folder and uses it to test on an image of your chosing.
To use this script, you must enter the path to where the testing dataset is in your machine and it's ready to run.
#### Note:
This script saves the images it uses during the testing to the model folder.

### Implementation on site (`main.py`)
This script gets the first model on the saved_models folder and uses it to test on a list or array of images.
It then sends the results of the test, using MQTT, to a dashboard built with Node-RED so that the data can be visualized.
To use this script, you must enter the path to where the testing dataset is in your machine and it's ready to run.

### IoU Testing (`IoU_test.py`)
This script tests the models' ability to accurately detect the location of the anomaly, using IoU.
It gets the first model on the saved_models folder and uses it to test on a list or array of images, by detecting the defect and creating bounding boxes for both the model prediction and the grounth truth, comparing both and scoring the accuracy of the model.
To use this script, you must enter the path to where the testing dataset is in your machine and it's ready to run.
#### Note:
This script requires the ground truth of the anomalies for it to be fully used. You can also use it the measure the models' accuracy without IoU.

## Authors
* Nuno Fernandes - https://github.com/nunof98
