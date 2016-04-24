#!/usr/bin/python

import sys
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


class Detector(object):
	def __init__(self, image_path):
		self.image_path = image_path
		self.img = cv2.imread(self.image_path)
		self.vis = self.img.copy()
		self.rects = {'ch': [], 'text': []}

	def detect_char(self, visual=True):
		self.vis = cv2.imread(self.image_path)
		gray = cv2.imread(self.image_path, 0)

		erc1 = cv2.text.loadClassifierNM1('text/trained_classifierNM1.xml')
		er1 = cv2.text.createERFilterNM1(erc1)

		erc2 = cv2.text.loadClassifierNM2('text/trained_classifierNM2.xml')
		er2 = cv2.text.createERFilterNM2(erc2)

		regions = cv2.text.detectRegions(gray, er1, er2)
		rects = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]

		if visual:
			for rect in rects:
				self.rects['ch'].append(rect)
				cv2.rectangle(self.vis, rect[0:2], (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)
		else:
			for rect in rects: self.rects['ch'].append(rect)

	def detect_text(self, visual=True):
		img = self.img
		vis = self.vis

		# Extract channels to be processed individually
		channels = cv2.text.computeNMChannels(img)

		# Append negative channels to detect ER- (bright regions over dark background)
		cn = len(channels) - 1

		for c in range(0, cn):
			channels.append((255 - channels[c]))

		# Apply the default cascade classifier to each independent channel (could be done in parallel)
		print("Extracting Class Specific Extremal Regions from " + str(len(channels)) + " channels ...")
		print("    (...) this may take a while (...)")
		for channel in channels:
			erc1 = cv2.text.loadClassifierNM1('text/trained_classifierNM1.xml')
			er1 = cv2.text.createERFilterNM1(erc1, 16, 0.00015, 0.13, 0.2, True, 0.1)

			erc2 = cv2.text.loadClassifierNM2('text/trained_classifierNM2.xml')
			er2 = cv2.text.createERFilterNM2(erc2, 0.5)

			regions = cv2.text.detectRegions(channel, er1, er2)
			# print regions
			rects = cv2.text.erGrouping(img, channel, [r.tolist() for r in regions])
			# rects = cv2.text.erGrouping(img,gray,[x.tolist() for x in regions], cv2.text.ERGROUPING_ORIENTATION_ANY,'../../GSoC2014/opencv_contrib/modules/text/samples/trained_classifier_erGrouping.xml',0.5)

			if visual:
				for r in range(0, np.shape(rects)[0]):
					rect = rects[r]
					self.rects['text'].append(rect)
					cv2.rectangle(vis, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 255), 2)

			else:
				for r in range(0, np.shape(rects)[0]): self.rects['text'].append(rects[r])

	def visualize(self):
		vis = self.vis
		vis = vis[:, :, ::-1]  # flip the colors dimension from BGR to RGB
		plt.imshow(vis)
		plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
		plt.show()

	def getRects(self):
		return self.rects

	# TODO max, min filter is not good enough,need clustering
	def getTextBlocks(self):
		x1, y1, x2, y2 = float('inf'), float('inf'), 0, 0
		for k in self.rects:
			for r in self.rects[k]:
				x1_, y1_, w, h = r
				x1, y1, x2, y2 = min(x1, x1_), min(y1, y1_), max(x2, x1_ + w), max(y2, y1_ + h)
		# cv2.rectangle(self.vis, (x1,y1), (x2,y2), (0, 255, 255), 2)
		# self.visualize()
		cropped_img = self.vis[y1:y2, x1:x2]
		# cv2.imshow('cropped',cropped_img)
		cv2.imwrite(self.image_path.replace('.jpg', '_cropped.jpg'), cropped_img)
		return x1, y1, x2, y2


if __name__ == '__main__':
	detector = Detector('data/scenetext02.jpg')
	detector.detect_text(visual=False)
	# detector.visualize()
	print detector.getTextBlocks()
