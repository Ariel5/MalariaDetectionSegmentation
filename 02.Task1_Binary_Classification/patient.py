import os
import numpy
import PIL
import torch
from PIL import Image, ImageFilter
from torchvision import transforms
import re
import torchvision
import math
import random

class Patient(object):
	
	def __init__(self, ID, path, truth, truth_columns, config):
		self.ID = str(ID)
		self.path = str(path)
		self.truth = truth.copy()
		self.truth_columns = truth_columns.copy()
		self.config = config
		self.downsample_ratio = 1
		self.rotate_angle = 0

	def get_h_w_resize_keep_ratio(self, h, w):
		patch_size = self.config.data.patch_size
		if h>w:
			return patch_size, int(patch_size*w/h)
		return int(patch_size*h/w), patch_size

	def get_padding_parameters(self, h, w):
		patch_size = self.config.data.patch_size
		if w==patch_size:
			return (patch_size-h)//2, 0
		return 0, (patch_size-w)//2

	def get_resized_image(self, image):
		patch_size = self.config.data.patch_size//self.downsample_ratio
		w, h = image.size
		h, w = self.get_h_w_resize_keep_ratio(h, w)
		ph, pw = self.get_padding_parameters(h, w)
		image = torchvision.transforms.functional.resize(image, (h, w))
		image = torchvision.transforms.functional.pad(image, (pw, ph))
		image = torchvision.transforms.functional.resize(image, (patch_size, patch_size))
		return image

	def get_image(self, scanner, view, hbv_only=False):
		if hbv_only:
			if 'hb' not in self.path.lower():
				return None
		for r,d,f in os.walk(self.path):
			for file in f:
				if scanner.lower() not in r.lower():
					continue
				if 'View-'+str(view) not in file:
					continue
				if int(re.findall(r'View-(\d+)', file)[-1])!=view:
					continue
				if '.tif' not in file.lower():
					continue

				image_original = Image.open(r+'/'+file).convert('RGB')
				self.downsample_ratio = 1
				image = self.get_resized_image(image_original)
				return image
		return None

	def get_mask(self, scanner, view, hbv_only=False, downsample_ratio=1):
		if hbv_only:
			if 'hb' not in self.path.lower():
				return None

		for r,d,f in os.walk(self.path):
			for file in f:
				if scanner.lower() not in r.lower():
					continue
				if 'View-'+str(view) not in file:
					continue
				if int(re.findall(r'View-(\d+)', file)[-1])!=view:
					continue
				if '-contour.bmp' not in file.lower():
					continue

				image_original = Image.open(r+'/'+file).convert('RGB')
				# image = self.mask_to_roi(image_original)
				self.downsample_ratio = downsample_ratio
				image = self.get_resized_image(image_original)
				return image
		return None


	# def mask_to_roi(self, mask):
	# 	mask = numpy.asarray(mask)
	# 	from skimage.measure import label
	# 	connected_area_amount = label(mask).max()
	# 	# if connected_area_amount>1:
	# 	# 	return None

	# 	y_0 = mask[:,:,0].sum(axis=1).nonzero()[0][0]
	# 	y_1 = mask[:,:,0].sum(axis=1).nonzero()[0][-1]
		
	# 	box_y0 = max(0, y_0-10)
	# 	box_y1 = y_0+(y_1-y_0)//2

	# 	mask1 = numpy.zeros_like(mask)
	# 	mask1[box_y0:box_y1, :, :]=1
	# 	mask2 = mask*mask1

	# 	box_x0 = mask2.sum(axis=0).nonzero()[0][0]+10
	# 	box_x1 = mask2.sum(axis=0).nonzero()[0][-1]-10

	# 	if box_x0>box_x1:
	# 		return None

	# 	result = numpy.zeros_like(mask)
	# 	result[box_y0:box_y1, box_x0:box_x1, :]=1

	# 	result = Image.fromarray(result)

	# 	return result











	def get_binary_truth(self, threshold):
		truth = int(self.truth[0])
		if truth <=threshold:
			return torch.tensor([0]).long()
		else:
			return torch.tensor([1]).long()

	# def get_truth(self):
	# 	truth = int(self.truth[0])
	# 	if truth <=10:
	# 		return torch.tensor([1, 0, 0, 0]).float(), torch.tensor([0]).long()
	# 	elif truth <=33:
	# 		return torch.tensor([0, 1, 0, 0]).float(), torch.tensor([1]).long()
	# 	elif truth <=33:
	# 		return torch.tensor([0, 0, 1, 0]).float(), torch.tensor([2]).long()
	# 	else:
	# 		return torch.tensor([0, 0, 0, 1]).float(), torch.tensor([3]).long()

	def get_truth_normalized(self):
		try:
			if numpy.isnan(self.truth[0]):
				return None
			truth = math.log(float(self.truth[0]))/self.config.data.regression_normalize_max
			return torch.tensor([truth]).float()
		except:
			return None

	def get_truth_us_3_category(self):
		# if int(self.truth[0]) in [5,6]:
		if int(self.truth[0]) in [0,1,2]:
			category = 0
		# elif int(self.truth[0]) in [7]:
		elif int(self.truth[0]) in [3]:
			category = 1
		else:
			category = 2
		return torch.tensor([category]).long()

	def get_truth_us_3_category_float(self):
		if int(self.truth[0]) in [5,6]:
			category = 0
		elif int(self.truth[0]) in [7]:
			category = 1
		else:
			category = 2
		truth = float(category)/2.0
		return torch.tensor([truth]).float()

	def get_truth_category(self):
		category = int(self.truth[0])-self.config.data.category_offset
		return torch.tensor([category]).long()

	def get_ID(self):
		return self.ID

	def __str__(self):
		return self.ID+', '+str(self.truth)+', '+self.path