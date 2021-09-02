# %% Imports
import os
import glob
import random
import statistics
import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage import exposure
from skimage.metrics import structural_similarity as ssim
from tensorflow import keras
#from keras.models import load_model
import tensorflow_addons as tfa
import time

# %% Function: calculate mean squared error
def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    error /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return error

# %% Function: compare images with SSIM
def compare_images(imageA, imageB, titleA, titleB):
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    # setup the figure
    fig = plt.figure(titleA + " vs. " + titleB)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    # show first image
    ax = fig.add_subplot(1, 2, 1)
    ax.title.set_text(titleA)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")
    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    ax.title.set_text(titleB)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")
    # show the images
    plt.show()
    
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

#%% Function: get bounding boxes
def get_boundingbox(image):
    contours, hierarchy = cv.findContours(np.uint8(image), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    big_contour = max(contours, key=cv.contourArea)
    x,y,w,h = cv.boundingRect(big_contour)
    
    color = (255, 0, 0)
    drawing = image.copy()
    drawing = cv.rectangle(drawing, (x, y), (x+w, y+h), color, 1)
    
    return x, y, x+w, y+h, drawing

#%% Function: Intersection over Union (IoU)
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return round(iou, 2)

#%%
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

# %% Load model
path_load = r'' # path to saved models
path_load = os.path.join(path_load, os.listdir(path_load)[0])

path_load_autoencoder = path_load + r'\autoencoder'
path_load_encoder = path_load + r'\encoder'

autoencoder = keras.models.load_model(path_load_autoencoder, compile=False, )
encoder = keras.models.load_model(path_load_encoder, compile=False, )


path_test = r'' # path to test dataset
path_save_results = r'' # path to save results

# paramaters
shape = (256, 256)

# get test data as np array
x_test = img_to_np(path_test)
x_test = x_test.astype('float32') / 255.
x_test = np.reshape(x_test, (len(x_test), shape[0], shape[1], 1))

# get test data as np array
x_ground_truth = img_to_np(path_ground_truth)
x_ground_truth = x_ground_truth.astype('float32') / 255.
x_ground_truth = np.reshape(x_ground_truth, (len(x_ground_truth), shape[0], shape[1], 1))

# %% Predictions
#cmap = 'viridis'
cmap = 'gray'
iou_array = []
lines = []
count_ok = 0
count_nok = 0

for i in range(0, len(x_test)):
    lines.append("{}".format(i))
    #input image
    #img_in = x_test[41 + i]
    img_in = x_test[i]
    # decoded/output image
    img_decoded = autoencoder.predict(np.expand_dims(img_in, 0))
    img_decoded = img_decoded[0]
    # loss image
    #d = (img_in - img_decoded).astype(float)
    #img_loss = np.sqrt(np.einsum('...i,...i->...', d, d))
    _, img_loss = resmaps_ssim(img_in.squeeze(2), img_decoded.squeeze(2))
    
    # apply gaussian filter to loss image
    img_blur = cv.GaussianBlur(img_loss, (5, 5), 0)
    img_blur = img_blur * 255

    # histogram
    _, bins_center = exposure.histogram(img_blur)
    thresh_value = np.max(bins_center) * (2/3)

    # binarize image
    _, img_defect = cv.threshold(img_blur, thresh_value, 255, cv.THRESH_BINARY)

    # Get contours
    contours = cv.findContours(np.uint8(img_defect), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv.contourArea)
        
    # test blob size
    blob_area_thresh = 85
    blob_area = cv.contourArea(big_contour)
    if blob_area < blob_area_thresh:
        print("OK")
        lines.append('OK')
        count_ok += 1
    else:
        print("NOK")
        lines.append('NOK')
        count_nok += 1
    
    # get bounding box
    boxA = get_boundingbox(img_defect)
    boxB = get_boundingbox(x_ground_truth[i])
    img_boxes = cv.addWeighted(np.uint8(boxA[4]), 1, np.uint8(boxB[4]), 1, 0)
    cv.imwrite(r'C:\Disciplinas\Licenciatura\results\bb_mvtec\{}.png'.format(i), boxA[4])
    cv.imwrite(r'C:\Disciplinas\Licenciatura\results\bb_model\{}.png'.format(i), boxB[4])
    cv.imwrite(r'C:\Disciplinas\Licenciatura\results\bb_both\{}.png'.format(i), img_boxes)
    
    # IoU evaluation
    iou = bb_intersection_over_union(boxA, boxB)
    print('IoU: {}'.format(iou))
    lines.append('IoU: {}'.format(iou))
    iou_array.append(iou)

# total part counter
count_total = count_ok + count_nok
print('Total: {}'.format(count_total))
lines.append('Total: {}'.format(count_total))
# calculate model accuracy
accuracy = round(count_nok / count_total, 2)
print('Accuracy: {}'.format(accuracy * 100))
lines.append('Accuracy: {}%'.format(accuracy * 100))
# calculate median IoU rate
mean_iou = statistics.mean(iou_array)
print('Median IoU: {}'.format(mean_iou))
lines.append('Median IoU: {}'.format(mean_iou))

with open(path_save_results + r'\results_{}.txt'.format(anomaly), 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')

