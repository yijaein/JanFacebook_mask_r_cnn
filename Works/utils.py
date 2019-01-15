import math
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def save_auroc(auc, roc, filename):
    plt.plot(*zip(*roc), label='ROC curve')
    plt.plot([0, 1], label='Random guess', linestyle='--', color='red')
    plt.legend(loc=4)
    plt.ylabel('TPR (True Positive Rate)')
    plt.xlabel('FPR (False Positive Rate)')
    plt.title('ROC Curve (AUROC : %7.3f)' % (auc))
    plt.axis([0, 1, 0, 1])
    plt.grid()
    plt.savefig(filename)
    plt.clf()
    plt.close()


def compute_auroc(predict, target):
    n = len(predict)

    # Cutoffs are of prediction values
    cutoff = predict

    TPR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    for k in range(n):
        predict_bin = 0

        TP = 0  # True	Positive
        FP = 0  # False Positive
        FN = 0  # False Negative
        TN = 0  # True	Negative

        for j in range(n):
            if (predict[j] >= cutoff[k]):
                predict_bin = 1
            else:
                predict_bin = 0

            TP = TP + (predict_bin & target[j])
            FP = FP + (predict_bin & (not target[j]))
            FN = FN + ((not predict_bin) & target[j])
            TN = TN + ((not predict_bin) & (not target[j]))

        # value to prevent divison by zero
        # Implement It is divided into smaller values instead, If you divide by 0 when you compute TPR, FPR
        very_small_value = 1e-10

        # True	Positive Rate
        # TPR[k] = float(TP) / float(TP + FN)
        TPR[k] = float(TP) / (float(TP + FN) if float(TP + FN) != 0 else very_small_value)
        # False Positive Rate
        # FPR[k] = float(FP) / float(FP + TN)
        FPR[k] = float(FP) / (float(FP + TN) if float(FP + TN) != 0 else very_small_value)

    TPR[n] = 0.0
    FPR[n] = 0.0
    TPR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, TPR), reverse=True)

    AUROC = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        AUROC = AUROC + h * w

    return AUROC, ROC


def compute_psnr(mse, max_signal=1, min_mse=1e-10):
    # Make mse not too low so psnr is not infinite
    # When max_signal == 1 and min_use == 1e-10, max of psnr has a value of 100.0
    if min_mse is not None:
        mse = mse if mse > min_mse else min_mse
    return 10 * math.log10(math.pow(max_signal, 2) / mse)


def save_log_graph(log, save=None, sep='\t', x_column_idx=0, y_column_idx=1):
    if not os.path.exists(log):
        print('not found')
        return

    if save is None:
        path, file = os.path.split(log)
        name, _ = os.path.splitext(file)
        save = os.path.join(path, name + '.png')

    # read log
    xValues = list()
    yValues = list()
    with open(log, 'rt') as f:
        while True:
            line = f.readline()
            if not line:
                break

            columns = line.strip().split(sep)
            if len(columns) < 2:
                continue

            xValues.append(float(columns[x_column_idx]))
            yValues.append(float(columns[y_column_idx]))

    # save graph
    plt.plot(xValues, yValues)
    plt.title(name)
    plt.savefig(save)
    plt.close()


def contour_filling_lesion(image):
    image = image.astype(np.uint8)

    # 64 is reasonable(?) threshold
    image[image > 64] = 255

    kernel = np.ones((5, 5), np.float32) / 25.0
    image = cv2.filter2D(image, -1, kernel)
    image = cv2.filter2D(image, -1, kernel)

    return np.array(image)


def contour_filling(image):
    image = image.astype(np.uint8)

    _, contours, hier = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    areaArray = []

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)

    # first sort the array by area
    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

    for i in reversed(range(len(areaArray))):
        contour = sorteddata[i][1]
        if i == 0:
            cv2.drawContours(image, [contour], 0, 255, -1)
        else:
            cv2.drawContours(image, [contour], 0, 0, -1)

    if len(sorteddata) > 0:
        hull = cv2.convexHull(sorteddata[0][1])
        cv2.drawContours(image, [hull], 0, 255, -1)
    else:
        image.fill(0)

    return np.array(image)


def fill_small_area(img, valid_value=1, small_area_ratio=0.02, fill_value=0):
    flat_img = img.reshape((-1))

    all_pixels = len(flat_img)
    brightness_pixels = len(flat_img[flat_img >= valid_value])
    area_ratio = brightness_pixels / all_pixels

    if small_area_ratio >= area_ratio:
        # It has got small area
        # fill image
        img[:] = fill_value

    return img


def post_processing(img, channels=1, threshold=100):
    img = np.transpose(img, (1, 2, 0)) if channels == 3 else img
    img = contour_filling_lesion(img)

    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    img = contour_filling(img)
    img = fill_small_area(img)

    return img
