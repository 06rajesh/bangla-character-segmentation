#!/usr/bin/python3

import cv2
import numpy as np


def resize(img, height=30):
    ratio = height/img.shape[0]
    width = int(img.shape[1]*ratio)
    return cv2.resize(img, (width, height))


def get_region_of_interest(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # (2) threshold
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # (3) minAreaRect on the no-zeros
    pts = cv2.findNonZero(threshed)
    ret = cv2.minAreaRect(pts)
    box = cv2.boxPoints(ret)

    x1 = int(min(box[0][0], box[1][0], box[2][0], box[3][0]))  # top-left pt. is the leftmost of the 4 points
    x2 = int(max(box[0][0], box[1][0], box[2][0], box[3][0]))  # bottom-right pt. is the rightmost of the 4 points
    y1 = int(min(box[0][1], box[1][1], box[2][1], box[3][1]))  # top-left pt. is the uppermost of the 4 points
    y2 = int(max(box[0][1], box[1][1], box[2][1], box[3][1]))  # bottom-right pt. is the lowermost of the 4 points

    return threshed[y1: y2, x1: x2]


def get_lines(roi):
    # get areas where we can split image on whitespace to make OCR more accurate
    color_level = np.array([np.sum(line) for line in roi])
    cuts = []
    i = 0
    while i < len(color_level):
        if color_level[i] > 250:
            begin = i
            while i < len(color_level) and color_level[i] > 250:
                i += 1
            cuts.append([begin, i])
        else:
            i += 1

    lines = []
    for line in cuts:
        lines.append([(0, line[0]), (roi.shape[1], line[1])])
    return lines


def get_words(line, debug=False):
    # Word Segmentation from single line
    transposed = np.transpose(line)

    color_level = np.array([np.sum(line) for line in transposed])
    cuts = []
    i = 0
    while i < len(color_level):
        if color_level[i] > 250:
            begin = i
            while i < len(color_level) and color_level[i] > 250:
                if i + 2 < len(color_level) and color_level[i+1] < 250:
                    if color_level[i+2] > 250:
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1
            if debug:
                print(i)
            cuts.append([begin, i])
        else:
            i += 1

    words = []
    for item in cuts:
        word_horizon = [(item[0], 0), (item[1], line.shape[0])]
        words.append(word_horizon)

    return words


def get_intersect(row):
    cuts = []
    i = 0
    while i < len(row):
        if row[i] > 180:
            begin = i
            while i < len(row) and row[i] > 180:
                i += 1
            cuts.append([begin, i])
        else:
            i += 1
    points = np.array([int(np.sum(line)/len(line)) for line in cuts])
    return points
