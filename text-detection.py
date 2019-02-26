#!/usr/bin/python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scanner

img = cv2.imread('samples/img_3.jpg')
roi = scanner.get_region_of_interest(img)

cuts = scanner.get_lines(roi)
line = cuts[5]

single = roi[line[0][1]: line[1][1], line[0][0]: line[1][0]]
resized = scanner.resize(single)
words = scanner.get_words(resized)

word = words[2]

# custom matra detection
word_roi = resized[word[0][1]: word[1][1], word[0][0]: word[1][0]]
color_level = np.array([np.sum(line) for line in word_roi])
max_row = color_level.argmax()
cv2.line(word_roi, (0, max_row), (word[1][0], max_row), (255, 255, 255), 1)

size = word_roi.shape

top_segment = word_roi[0:max_row+1, 0: size[1]]
bottom_segment = word_roi[max_row+2: size[1], 0: size[1]]

characters = scanner.get_words(bottom_segment)
print(characters)

reversed_level = color_level[::-1]  # reverse the array to plot on graph

plt.figure(1)
plt.subplot(121)
plt.imshow(bottom_segment)
plt.subplot(122)
plt.barh(np.arange(len(reversed_level)), reversed_level)
plt.ylabel('Pixel Density')
plt.show()

# cv2.imshow('Single World', resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
