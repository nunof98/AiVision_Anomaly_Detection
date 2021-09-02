# %% Imports
import os
import glob
import random
import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage import exposure
from skimage.metrics import structural_similarity as ssim
from keras.models import load_model
#import tensorflow_addons as tfa

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

# %% Function: get bounding boxes


def get_boundingbox(image):
    contours, hierarchy = cv.findContours(
        np.uint8(image), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    big_contour = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(big_contour)

    return x, y, x+w, y+h

# %% Function: Intersection over Union (IoU)


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


# %% Load model
#dataset = 'screw'
dataset = 'ETMA_dataset'
path_load = r'C:\Disciplinas\Licenciatura\3º ano\Projeto\AiVision Defects\saved_models\{}\\'.format(dataset)
path_load += os.listdir(path_load)[0]

path_load_autoencoder = path_load + r'\autoencoder'
path_load_encoder = path_load + r'\encoder'

# custom_objects={"F1Score": tfa.metrics.F1Score})
autoencoder = load_model(path_load_autoencoder, compile=False)
# custom_objects={"F1Score": tfa.metrics.F1Score})
encoder = load_model(path_load_encoder, compile=False)

# %% Load dataset
#path_test = r'C:\Disciplinas\Licenciatura\3º ano\Projeto\AiVision Defects\mvtec_dataset\{}\test\**\*.*'.format(dataset)
path_test = r'C:\Disciplinas\Licenciatura\3º ano\Projeto\AiVision Defects\ETMA_dataset\test\**\*.*'

# paramaters
shape = (256, 256)

# get test data as np array
x_test = img_to_np(path_test)
x_test = x_test.astype('float32') / 255.
x_test = np.reshape(x_test, (len(x_test), shape[0], shape[1], 1))

# get test data as np array and aply noise
noise_factor = 0.10
x_test_noisy = img_to_np(path_test, noise=True)
x_test_noisy = x_test_noisy.astype('float32') / 255.
x_test_noisy = np.reshape(
    x_test_noisy, (len(x_test_noisy), shape[0], shape[1], 1))
x_test_noisy = x_test_noisy + noise_factor * \
    np.random.normal(loc=0.0, scale=1.0, size=x_test_noisy.shape)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# get test data as np array
#x_ground_truth = img_to_np(path_ground_truth)
#x_ground_truth = x_ground_truth.astype('float32') / 255.
#x_ground_truth = np.reshape(x_ground_truth, (len(x_ground_truth), shape[0], shape[1], 1))

# %% Predictions
#cmap = 'viridis'
cmap = 'gray'

#[5, 22, 42, 47, 62, 88, 130, 131, 141, 146, 154, 201]
# input image
i = 62
#i = 110
img_in = x_test[i]
plt.imsave(path_load + r'\input_image.png', img_in.squeeze(2), cmap=cmap)

# noisy image
img_noisy = x_test_noisy[i]
plt.imsave(path_load + r'\noisy_image.png', img_noisy.squeeze(2), cmap=cmap)

# encoded image
img_encoded = encoder.predict(np.expand_dims(img_in, 0))
img_encoded = img_encoded[0, :, :, 0]
plt.imsave(path_load + r'\encoded_image.png', img_encoded, cmap=cmap)

# decoded/output image
img_decoded = autoencoder.predict(np.expand_dims(img_in, 0))
img_decoded = img_decoded[0]
plt.imsave(path_load + r'\decoded_image.png', img_decoded.squeeze(2), cmap=cmap)

# loss image
#d = (img_in - img_decoded).astype(float)
#img_loss = np.sqrt(np.einsum('...i,...i->...', d, d))
#plt.imsave(path_load + r'\loss_image.png', img_loss, cmap=cmap)
_, img_loss = resmaps_ssim(img_in.squeeze(2), img_decoded.squeeze(2))
plt.imsave(path_load + r'\loss_image.png', img_loss, cmap=cmap)

# %% Results
#cmap = 'viridis'
cmap = 'gray'

# create figure
fig_results = plt.figure('Results')
# input image
ax1 = fig_results.add_subplot(1, 5, 1)
plt.axis('off')
ax1.imshow(img_in, cmap=cmap)
ax1.title.set_text('Original')
# noisy image
ax2 = fig_results.add_subplot(1, 5, 2)
plt.axis('off')
ax2.imshow(img_noisy, cmap=cmap)
ax2.title.set_text('Noisy')
# code image
ax3 = fig_results.add_subplot(1, 5, 3)
plt.axis('off')
ax3.imshow(img_encoded, cmap=cmap)
ax3.title.set_text('Code')
# output image
ax4 = fig_results.add_subplot(1, 5, 4)
plt.axis('off')
ax4.imshow(img_decoded, cmap=cmap)
ax4.title.set_text('Reconstructed')
# loss image
ax5 = fig_results.add_subplot(1, 5, 5)
plt.axis('off')
ax5.imshow(img_loss, cmap=cmap)
ax5.title.set_text('Loss')

# save figure
fig_results.savefig(path_load + r'\Results_{}.png'.format(cmap))
# show plot
plt.tight_layout()
plt.show()

# %% Segmentation anomaly
#cmap = 'viridis'
cmap = 'gray'

# apply gaussian filter to loss image
img_blur = cv.GaussianBlur(img_loss, (5, 5), 0)
img_blur = img_blur * 255

# plot histogram
fig_histogram = plt.figure('Histogram')
hist, bins_center = exposure.histogram(img_blur)
thresh_value = np.max(bins_center) * (2/3)
plt.plot(bins_center, hist, lw=2)
plt.axvline(thresh_value, color='k', ls='--')
plt.xlabel('pixel values')
plt.ylabel('pixel count')
fig_histogram.savefig(path_load + r'\Histogram.png')
plt.show()

# apply otsu method to binarize
#img_defect = cv2.adaptiveThreshold(np.uint8(img_blur*255), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
_, img_defect = cv.threshold(img_blur, thresh_value, 255, cv.THRESH_BINARY)
# save defect image
plt.imsave(path_load + r'\defect_image.png', img_defect, cmap=cmap)

# create figure
fig_segmentation = plt.figure('Segmentation')
# input image
ax1 = fig_segmentation.add_subplot(1, 3, 1)
plt.axis('off')
ax1.imshow(img_loss, cmap=cmap)
ax1.title.set_text('Original')
# noisy image
ax2 = fig_segmentation.add_subplot(1, 3, 2)
plt.axis('off')
ax2.imshow(img_blur, cmap=cmap)
ax2.title.set_text('Filtered')
# noisy image
ax3 = fig_segmentation.add_subplot(1, 3, 3)
plt.axis('off')
ax3.imshow(img_defect, cmap=cmap)
ax3.title.set_text('Defect')
# save figure
fig_segmentation.savefig(path_load + r'\Segmentation.png')
# show figure
plt.tight_layout()
plt.show()

# %% Blob detection
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
else:
    print("NOK")

# %% IoU evaluation
#boxA = get_boundingbox(x_test[41 + i])
#boxB = get_boundingbox(x_ground_truth[i])
#iou = bb_intersection_over_union(boxA, boxB)
#print('IoU: {}'.format(iou))

# %% Compare images with SSIM
compare_images(img_in.squeeze(2), img_in.squeeze(2), "Original", "Original")
compare_images(img_in.squeeze(2), img_decoded.squeeze(2), "Original", "Reconstructed")

# %% Plot several examples
n = 5
fig_examples = plt.figure('Examples')
for i in range(1, n + 1):
    img_in = x_test[random.randrange(x_test.shape[0])]
    img_out = autoencoder.predict(np.expand_dims(img_in, 0))

    # Display original
    ax = plt.subplot(2, n, i)
    ax.title.set_text(i)
    plt.imshow(img_in, cmap='gray')
    plt.axis('off')

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(img_out[0], cmap='gray')
    plt.axis('off')

fig_examples.savefig(path_load + r'\Examples.png')
plt.tight_layout()
plt.show()
