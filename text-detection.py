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
word = words[10]

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
bottom_segment = word_roi[max(matra_lines)+1: size[1], 0: size[1]]

# increasing/decreasing the thickness of lines if needed
kernel = np.ones((2, 2), np.uint8)/10
# top_segment = cv2.dilate(top_segment, kernel, iterations=1)
# bottom_segment = cv2.erode(bottom_segment, kernel, iterations=1)

top_matras = scanner.extract_upper_matra(top_segment)

characters = scanner.get_words(bottom_segment, one_pxl_exempt=False)

# checking characters with complexity
# checker = characters[8]
# full_char = bottom_segment[0: bottom_segment.shape[0], checker[0][0]: checker[1][0]]
# char_1 = bottom_segment[int(bottom_segment.shape[0]*0.70): bottom_segment.shape[0], checker[0][0]: checker[1][0]]
# contours, hierarchy = cv2.findContours(char_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# if len(contours) > 1:
#     full_char = cv2.erode(full_char, kernel, iterations=1)
#     spcl_chars = scanner.get_words(full_char, one_pxl_exempt=False)
#     print(len(spcl_chars))


char_images = []

# create all the character image
for idx, char in enumerate(characters):
    char_ub = min(matra_lines)
    sb = char[0][0]
    eb = char[1][0]
    temp_img = np.zeros((lb - ub, 40), np.uint8)

    this_char = word_roi[char_ub: lb, sb: eb]
    if this_char.shape[1] > this_char.shape[0]:
        print(idx)
        chars = scanner.divide_complex_word(this_char)
        print(len(chars))

    temp_img[min(matra_lines): lb, 0: eb-sb] = this_char
    for matra in top_matras:
        if sb < matra.intersection < eb:
            points = matra.points
            this_matra = word_roi[points[0][1]: points[1][1], points[0][0]: points[1][0]]
            temp_img[points[0][1]: points[1][1], 0: points[1][0] - points[0][0]] = this_matra
    char_images.append(temp_img)

# reversed_level = color_level[::-1]  # reverse the array to plot on graph

# all characters loop plot
plt.figure(1)
for idx in range(9):
    if idx < len(char_images):
        plt.subplot(4, 3, idx + 1)
        plt.imshow(char_images[idx])


plt.subplot(4, 3, 10)
plt.imshow(word_roi)
# plt.subplot(212)
# plt.imshow(resized)
# plt.barh(np.arange(len(reversed_level)), reversed_level)
# plt.ylabel('Pixel Density')
plt.show()

# cv2.imshow('Single World', resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
