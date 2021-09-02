# %%Imports
import os
import sys
import glob
import random
import base64
import time
import timeit
import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage import exposure
from skimage.metrics import structural_similarity as ssim
from datetime import datetime
import paho.mqtt.client as mqtt
from keras.models import load_model

# %% Function: raise flag when connect to client


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.connected_flag = True # set flag
        print("connected OK")
    else:
        print("Bad connection Returned code=",rc)
        
# %% Function: lower flag when disconnect to client


def on_disconnect(client, userdata, rc):
    print("disconnecting reason  " + str(rc))
    client.connected_flag = False

# %% Function: create np array with train images


def img_to_np(path, resize=True, noise=False):
    img_array = []
    shape = (256, 256)
    color = (255, 255, 255)
    fpaths = glob.glob(path, recursive=True)
    for fname in fpaths:
        img = Image.open(fname).convert('L')
        if(resize):
            img = img.resize(shape)
        if(noise):
            for i in range(0, 5):
                x1 = random.randrange(shape[0])
                y1 = random.randrange(shape[0])
                x2 = random.randrange(shape[0])
                y2 = random.randrange(shape[0])
                img = cv.line(np.uint8(img), (x1, y1), (x2, y2), color, 1)
        img_array.append(np.asarray(img))
    images = np.array(img_array)
    return images


# %%


def resmaps_ssim(imgs_input, imgs_pred):
    resmaps = np.zeros(shape=imgs_input.shape, dtype="float64")
    scores = []
    for index in range(len(imgs_input)):
        img_input = imgs_input[index]
        img_pred = imgs_pred[index]
        score, resmap = ssim(
            img_input,
            img_pred,
            win_size=11,
            gaussian_weights=True,
            multichannel=False,
            sigma=1.5,
            full=True,
        )
        resmaps[index] = 1 - resmap
        scores.append(score)
    resmaps = np.clip(resmaps, a_min=-1, a_max=1)
    return scores, resmaps


# %%Load model
#dataset = 'screw'
dataset = 'ETMA_dataset'
path_load = r'C:\Disciplinas\Licenciatura\3º ano\Projeto\AiVision Defects\saved_models\{}'.format(dataset)
path_load = os.path.join(path_load, os.listdir(path_load)[0])

path_load_autoencoder = path_load + r'\autoencoder'
autoencoder = load_model(path_load_autoencoder, compile=False)

# %%Load dataset
dataset = 'ETMA_dataset'
path_test = r'C:\Disciplinas\Licenciatura\3º ano\Projeto\AiVision Defects\ETMA_dataset\test\**\*.*'

#dataset = "screw"
#path_test = r'C:\Disciplinas\Licenciatura\3º ano\Projeto\AiVision Defects\mvtec_dataset\{}\test\**\*.*'.format(dataset)
shape = (256, 256)

# get test data as np array
x_test = img_to_np(path_test)
x_test = x_test.astype('float32') / 255.
x_test = np.reshape(x_test, (len(x_test), shape[0], shape[1], 1))

# variables
time_interval = 60 # time in seconds
count_ok = 0
count_nok = 0
count_total = 0
count_parts_minute = 0
last_time = ""

# %%MQTT
# select broker
broker = '127.0.0.1'
port = 1883

# create new instance
client = mqtt.Client('AiVision_Defects')
client.connected_flag = False

# bind call back function
client.on_connect = on_connect
client.on_disconnect = on_disconnect
print("Connecting to broker ", broker)

client.loop_start()

# connect to broker
try:
    client.connect(broker, port) #connect to broker
    # wait in loop
    while not client.connected_flag:
        print("In wait loop")
        time.sleep(1)
except:
    print('connection failed')
    sys.exit('quitting') #Should quit or raise flag to quit or retry

# %% Loop
# start timer
for i in range(0, len(x_test)):
    startTime = timeit.default_timer()
    # get input image
    print(i)
    img_in = x_test[i]
    plt.imsave(path_load + r'\input_image.png', img_in.squeeze(2), cmap='gray')

    # get autoencoder prediction
    img_decoded = autoencoder.predict(np.expand_dims(img_in, 0))
    img_decoded = img_decoded[0]
    plt.imsave(path_load + r'\decoded_image.png', img_decoded.squeeze(2), cmap='gray')

    # get loss image
    #d = (img_in - img_decoded).astype(float)
    #img_loss = np.sqrt(np.einsum('...i,...i->...', d, d))
    _, img_loss = resmaps_ssim(img_in.squeeze(2), img_decoded.squeeze(2))
    plt.imsave(path_load + r'\loss_image.png', img_loss, cmap='gray')

    # segment anomaly -------------------------------------------------------------
    # apply gaussian blur filter to loss image
    img_blur = cv.GaussianBlur(img_loss, (5, 5), 0)
    # unnormalize image (put pixe value between 0 and 255)
    img_blur = img_blur * 255

    # get threshold value for binarization
    _, bins_center = exposure.histogram(img_blur)
    thresh_value = np.max(bins_center) * (2/3)

    # apply otsu method to binarize
    #img_defect = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    _, img_defect = cv.threshold(img_blur, thresh_value, 255, cv.THRESH_BINARY)
    plt.imsave(path_load + r'\defect_image.png', img_defect, cmap='gray')

    # blob detection---------------------------------------------------------------
    # Get contours
    contours = cv.findContours(
        np.uint8(img_defect), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv.contourArea)

    # test blob size
    blob_area_thresh = 85
    blob_area = cv.contourArea(big_contour)
    if blob_area < blob_area_thresh:
        print("OK")
        client.publish('AiVision_Defects/result', 'OK')
        count_ok += 1
    else:
        print("NOK")
        client.publish('AiVision_Defects/result', 'NOK')
        count_nok += 1
    
    # get total part count
    count_total = count_ok + count_nok
    
    # calculate elapsed time in seconds
    elapsedTime = timeit.default_timer() - startTime
    # calculate estimated production per chosen time interval
    count_parts_minute = round(1 * time_interval / elapsedTime, 0)
    
    # send images to node red
    list_imgs = ['input_image', 'decoded_image',
                 'loss_image', 'defect_image']
    for img in list_imgs:
        path_img = path_load
        path_img += r'\{}.png'.format(img)
        encoded = base64.b64encode(open(path_img, "rb").read())
        client.publish('AiVision_Defects/{}'.format(img), encoded)
    
    # send statistical info to node red
    client.publish('AiVision_Defects/defect_percentage', round((count_nok / count_total) * 100, 0))
    client.publish('AiVision_Defects/total', count_total)
    client.publish('AiVision_Defects/ok_parts', count_ok)
    client.publish('AiVision_Defects/defect_parts', count_nok)
    client.publish('AiVision_Defects/parts_minute', count_parts_minute)
    # TODO
    # make list of 10 last defects
    
    # reset all counter with shift change (every 8h)
    now = datetime.now()
    current_time = now.strftime("%H")
    if current_time != last_time and (current_time == "6" or current_time == "14" or current_time == "22"):
        # raise flag so that it only executes once per shift
        last_time = current_time
        
        # select correct shift (data belongs to previous shift)
        if current_time == "6":
            current_time = "22"
        else:
            current_time = str(int(current_time) - 8)
        
        # send statistical info to node red
        client.publish('AiVision_Defects/defect_percentage_shift_{}'.format(current_time), round((count_nok / count_total) * 100, 0))
        client.publish('AiVision_Defects/total_shift_{}'.format(current_time), count_total)
        client.publish('AiVision_Defects/ok_parts_shift_{}'.format(current_time), count_ok)
        client.publish('AiVision_Defects/defect_parts_shift_{}'.format(current_time), count_nok)
        count_ok = 0
        count_nok = 0
        count_total = 0
        count_parts_minute = 0
    
    # wait 1 second
    #time.sleep(1)
    
# disconnect
client.loop_stop()
client.disconnect()
