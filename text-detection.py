#!/usr/bin/python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scanner

img = cv2.imread('samples/img_4.png')
roi = scanner.get_region_of_interest(img)

cuts = scanner.get_lines(roi)
line = cuts[5]

single = roi[line[0][1]: line[1][1], line[0][0]: line[1][0]]
resized = scanner.resize(single)
words = scanner.get_words(resized)

word = words[0]

# custom matra detection
word_roi = resized[word[0][1]: word[1][1], word[0][0]: word[1][0]]
color_level = np.array([np.sum(line) for line in word_roi])
max_row = color_level.argmax()
# cv2.line(word_roi, (0, max_row), (word[1][0], max_row), (255, 255, 255), 1)
matra_lines = []

for i in range(max_row-1, max_row+2):
    if color_level[i] > (color_level[max_row]/2):
        matra_lines.append(i)


size = word_roi.shape
ub, lb = word[0][1], word[1][1]

top_segment = word_roi[0:min(matra_lines)-1, 0: size[1]]
top_segment_height = top_segment.shape[0]
bottom_segment = word_roi[max(matra_lines)+1: size[1], 0: size[1]]
# kernel = np.ones((3, 3), np.uint8)/10
# top_segment = cv2.dilate(top_segment, kernel, iterations=1)

top_matras = scanner.get_words(top_segment)
top_intersection = scanner.get_intersect(top_segment[top_segment_height-2])
# print(top_matras)

characters = scanner.get_words(bottom_segment)

blank_image = np.zeros((lb-ub, 200), np.uint8)

char_images = []

# print(blank_image.shape)
for idx, char in enumerate(characters):
    sb = char[0][0]
    eb = char[1][0]
    this_char = word_roi[ub: lb, sb: eb]
    temp_img = np.zeros((lb-ub, 30), np.uint8)
    temp_img[ub: lb, 0: eb-sb] = this_char
    char_images.append(temp_img)
    # blank_image[ub: lb, idx*30: (idx+1)*30] = temp_img
    # cv2.line(word_roi, (sb, ub), (sb, lb), (255, 255, 255), 1)
    # cv2.line(word_roi, (eb, ub), (eb, lb), (255, 255, 255), 1)

# reversed_level = color_level[::-1]  # reverse the array to plot on graph

# all characteres loop plot
plt.figure(1)
for idx in range(len(char_images)):
    print(3, int(idx/3)+1, idx % 3 + 1)
    plt.subplot(3, int(idx/3)+1, idx % 3 + 1)
    plt.imshow(char_images[idx])

# plt.figure(1)
# plt.subplot(211)
# plt.imshow(char_images[1])
# plt.subplot(212)
# plt.imshow(resized)
# plt.barh(np.arange(len(reversed_level)), reversed_level)
# plt.ylabel('Pixel Density')
plt.show()

# cv2.imshow('Single World', resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
