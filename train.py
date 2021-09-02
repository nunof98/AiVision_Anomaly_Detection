# %% Imports
import os
import datetime
import timeit
import glob
import json
import random
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from autoencoder.models import myCAE
from autoencoder.models import myCAE_optuna
from autoencoder.models import myCAE_optuna_ETMA
from autoencoder.models import mvtecCAE
from autoencoder.models import baselineCAE
from autoencoder.models import inceptionCAE
from autoencoder.models import resnetCAE
from autoencoder.models import skipCAE
from autoencoder import metrics
from autoencoder import losses

from tensorflow_addons.optimizers import CyclicalLearningRate
from keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model

# %% Function: create np array with train images


def img_to_np(path, resize=True, noise=False):
    img_array = []
    size = 256
    color = (255, 255, 255)
    fpaths = glob.glob(path, recursive=True)
    for fname in fpaths:
        img = Image.open(fname).convert('L')
        if(resize):
            img = img.resize((size, size))
        if(noise):
            for i in range(0, 5):
                x1 = random.randrange(size)
                y1 = random.randrange(size)
                x2 = random.randrange(size)
                y2 = random.randrange(size)
                img = cv.line(np.uint8(img), (x1, y1), (x2, y2), color, 1)
        img_array.append(np.asarray(img))
    images = np.array(img_array)
    return images


# %% Paths and parameters
dataset = "ETMA_dataset"
path_train = r'C:\Disciplinas\Licenciatura\3º ano\Projeto\AiVision Defects\ETMA_dataset\train\**\*.*'
path_test = r'C:\Disciplinas\Licenciatura\3º ano\Projeto\AiVision Defects\ETMA_dataset\test\**\*.*'

#dataset = "screw"
#path_train = r'C:\Disciplinas\Licenciatura\3º ano\Projeto\AiVision Defects\mvtec_dataset\{}\train\**\*.*'.format(dataset)
#path_test = r'C:\Disciplinas\Licenciatura\3º ano\Projeto\AiVision Defects\mvtec_dataset\{}\test\**\*.*'.format(dataset)

# parameters
architecture = 'myCAE_optuna_ETMA'
color_mode = 'grayscale'
rescale = 1.0 / 255
shape = (256, 256)
vmin = 0.0
vmax = 1.0
dynamic_range = vmax - vmin
batch_size = 8

# %% Load dataset
# get train data as np array
x_train = img_to_np(path_train)
x_train = x_train.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), shape[0], shape[1], 1))

# get train data as np array and aply noise
noise_factor = 0.10
x_train_noisy = img_to_np(path_train, noise=True)
x_train_noisy = x_train_noisy.astype('float32') / 255.
x_train_noisy = np.reshape(
    x_train_noisy, (len(x_train_noisy), shape[0], shape[1], 1))
x_train_noisy = x_train_noisy + noise_factor * \
    np.random.normal(loc=0.0, scale=1.0, size=x_train_noisy.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)

# %% Create model

if architecture == 'myCAE':
    autoencoder, encoder = myCAE.build_model(shape, color_mode)
elif architecture == 'myCAE_optuna':
    autoencoder, encoder = myCAE_optuna.build_model(shape, color_mode)
elif architecture == 'myCAE_optuna_ETMA':
    autoencoder, encoder = myCAE_optuna_ETMA.build_model(shape, color_mode)
elif architecture == 'mvtecCAE':
    autoencoder, encoder = mvtecCAE.build_model(shape, color_mode)
elif architecture == 'baselineCAE':
    autoencoder, encoder = baselineCAE.build_model(shape, color_mode)
elif architecture == 'inceptionCAE':
    autoencoder, encoder = inceptionCAE.build_model(shape, color_mode)
elif architecture == 'resnetCAE':
    autoencoder, encoder = resnetCAE.build_model(shape, color_mode)
elif architecture == 'skipCAE':
    autoencoder, encoder = skipCAE.build_model(shape, color_mode)

# %% Autoencoder

# cyclic learning rate
# cyclical_learning_rate = CyclicalLearningRate(
#    initial_learning_rate=1e-5,
#    maximal_learning_rate=1e-2,
#    step_size=4 * x_train.shape[0] / BATCH_SIZE,
#    scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
#    scale_mode='cycle')
#adam = tf.keras.optimizers.Adam(learning_rate=cyclical_learning_rate)

# set metrics to monitor training
if color_mode == 'grayscale':
    loss_function = losses.ssim_loss(dynamic_range)
    metric_function = [metrics.ssim_metric(dynamic_range)]
    hist_keys = ('loss', 'val_loss', 'ssim', 'val_ssim')
elif color_mode == 'rgb':
    loss_function = losses.mssim_loss(dynamic_range)
    metric_function = [metrics.mssim_metric(dynamic_range)]
    hist_keys = ('loss', 'val_loss', 'mssim', 'val_mssim')

#loss_function = losses.l2_loss
#adam = tf.keras.optimizers.Adam(learning_rate=0.00032130145895989355)
# compile model
autoencoder.compile(optimizer='adam', loss=loss_function,
                    metrics=metric_function)

# get model summary
autoencoder.summary()

# %% Train
# start timer
startTime = timeit.default_timer()

history = autoencoder.fit(x_train_noisy, x_train,
                          epochs=100,
                          batch_size=batch_size,
                          verbose=1,
                          validation_data=(x_train_noisy, x_train),
                          callbacks=[TensorBoard(log_dir='/tmp/AiVision_Defects')])

# calculate elapsed time in seconds
elapsedTime = timeit.default_timer() - startTime
# print elapse time in hours
print("Time taken for the Network to train: {:.3}h".format(
    elapsedTime / 60 / 60))

# evaluate model
loss_eval, ssim_eval = autoencoder.evaluate(
    x_train_noisy, x_train, batch_size=batch_size)
loss_eval = '%.2E' % loss_eval
ssim_eval = round(ssim_eval, 2)
# print values
print("Evaluated Loss: {}".format(loss_eval))
print("Evaluated SSIM: {}".format(ssim_eval))

# %% Save model
# create a directory to save model
now = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
path_save = r'C:\Disciplinas\Licenciatura\3º ano\Projeto\AiVision Defects\saved_models\{}'.format(
    dataset)
path_save += r'\loss_{}-ssim_{}-{}'.format(loss_eval, ssim_eval, now)

path_save_autoencoder = path_save + r'\autoencoder'
path_save_encoder = path_save + r'\encoder'

# create save directories
if not os.path.isdir(path_save_autoencoder):
    os.makedirs(path_save_autoencoder)
if not os.path.isdir(path_save_encoder):
    os.makedirs(path_save_encoder)

# save autoencoder model
autoencoder.save(path_save_autoencoder)
# save encoder model
encoder.save(path_save_encoder)

# %% Plot results
# plot loss
fig_loss = plt.figure('Plot Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
fig_loss.savefig(path_save + r'\Plot Loss.png')
plt.show()

# print final loss value
val_loss = '%.2E' % history.history['val_loss'][-1]
print('Loss: {}'.format(val_loss))

# plot ssim
fig_ssim = plt.figure('Plot SSIM')
plt.plot(history.history['ssim'], label='train')
plt.plot(history.history['val_ssim'], label='test')
plt.title('Model Structural Similarity')
plt.ylabel('SSIM')
plt.xlabel('Epoch')
plt.legend()
fig_ssim.savefig(path_save + r'\Plot SSIM.png')
plt.show()

# print final ssim value
val_ssim = round(history.history['val_ssim'][-1], 2)
print('SSIM: {}'.format(val_ssim))

# plot the models architecture
plot_model(autoencoder, path_save + r'\Autoencoder.png', show_shapes=True)

# %% Save training history
path_save_history = path_save + r'\train_history'
if not os.path.isdir(path_save_history):
    os.makedirs(path_save_history)

with open(os.path.join(path_save_history, "train_history.json"), "w") as json_file:
    json.dump(history.history, json_file, indent=4, sort_keys=False)

hist_dict = hist_dict = dict((key, history.history[key]) for key in hist_keys)
hist_df = pd.DataFrame(hist_dict)
hist_csv_file = os.path.join(path_save_history, "history.csv")
with open(hist_csv_file, mode="w") as csv_file:
    hist_df.to_csv(csv_file)

info = {
    "model": {"architecture": architecture, "loss": val_loss, },
    "preprocessing": {
        "color_mode": color_mode,
        "rescale": rescale,
        "shape": shape,
        "vmin": vmin,
        "vmax": vmax,
        "dynamic_range": dynamic_range,
    },
    # "lr_finder": {"lr_base": self.lr_base, "lr_opt": self.lr_opt,},
    "training": {
        "batch_size": batch_size,
        "epochs_trained": int(np.argmin(np.array(hist_dict["val_loss"]))),
    },
}

with open(os.path.join(path_save_history, "info.json"), "w") as json_file:
    json.dump(info, json_file, indent=4, sort_keys=False)
