import torch
import torchvision
import PIL
import random
import datetime
import numpy

class ImageDataset(torch.utils.data.Dataset):

	def __init__(self, patients, config, mode):
		
		self.x = []
		self.mask = []
		self.y = []
		self.pid = []
		self.viewgroup = []

		self.mode = mode

		self.batch_size = config.solver.batch_size

		self.patch_size = config.data.patch_size

		self.transform0 = torchvision.transforms.ColorJitter(brightness=.1, contrast=.1)
		self.transform1 = torchvision.transforms.RandomAffine(degrees=10, scale=(.9,1.1), shear=2, resample=PIL.Image.BILINEAR)
		self.transformNormalize = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

	
		for patient in patients:
			for scanner in ['toshi', 'siem', 'phi']:
				case_x = []
				case_mask = []
				case_viewgroup = []
				case_y = None
				case_pid = None

				for view in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
					image = patient.get_image(scanner=scanner, view=view, hbv_only=True)
					if image is None:
						continue
					
					mask = patient.get_mask(scanner=scanner, view=view, hbv_only=True, downsample_ratio=1)
					roi = self.mask_to_roi(mask)
					if roi is None:
						continue
					mask = self.fuseROI(mask, roi)

					patient.downsample_ratio = 16
					# roi = patient.get_resized_image(roi)
					mask = patient.get_resized_image(mask)

					
					case_x.append(image)
					case_mask.append(mask)
					case_y = patient.get_binary_truth(threshold=config.data.binary_threshold)
					case_pid = patient.ID + ':' + scanner
					
					if view in [1,2]:
						case_viewgroup.append(1)
					elif view in [3,4]:
						case_viewgroup.append(2)
					elif view in [5,6,7,8]:
						case_viewgroup.append(3)
					elif view in [9,10]:
						case_viewgroup.append(4)
					elif view in [11,12]:
						case_viewgroup.append(5)
					else:
						case_viewgroup.append(6)

				if len(case_x)>0:
					self.x.append(case_x)
					self.mask.append(case_mask)
					self.y.append(case_y)
					self.pid.append(case_pid)
					self.viewgroup.append(case_viewgroup)

					
		
	def __len__(self):
		return (len(self.x)//self.batch_size)


	def __getitem__(self, index):

		index_start = index*self.batch_size
		index_end = min(len(self.x), (index+1)*self.batch_size)

		result_length_list = []
		result_x = []
		result_mask = []
		result_y = []
		result_pid = []
		result_viewgroup = []

		while len(result_x)==0:
			for ii in range(index_start, index_end):
				
				case_x = self.x[ii]
				case_mask = self.mask[ii]
				case_viewgroup = self.viewgroup[ii]

				case_length = 0
				
				temp_x = []
				temp_mask = []
				temp_viewgroup = []
				
				for i in range(len(case_x)):
					if self.mode=='Training':
						random.seed(datetime.datetime.now())
						include_in_training_p = random.randint(1, 100)
						if include_in_training_p>50:
							continue

					case_length += 1
					image = case_x[i]
					mask = case_mask[i]

					if self.mode=='Training':
						random.seed(datetime.datetime.now())
						seed = random.randint(0,2**32)

						random.seed(seed)
						image = self.transform0(image)
						random.seed(seed)
						image = self.transform1(image)
						
						random.seed(seed)
						mask = self.transform1(mask)

					image = torchvision.transforms.functional.to_tensor(image)
					mask = torchvision.transforms.functional.to_tensor(mask)

					# image = self.transformNormalize(image)
					# mask = self.transformNormalize(mask)

					mask = mask[0,:,:].unsqueeze(0)

					# result_x.append(image)
					# result_mask.append(mask)
					# result_viewgroup.append(case_viewgroup[i])
					temp_x.append(image)
					temp_mask.append(mask)
					temp_viewgroup.append(case_viewgroup[i])

				if case_length>1:
					result_x.extend(temp_x)
					result_mask.extend(temp_mask)
					result_viewgroup.extend(temp_viewgroup)
					result_length_list.append(case_length)
					result_y.append(self.y[ii])
					result_pid.append(self.pid[ii])

			# for ii in range(index_start, index_end):
				
			# 	case_x = self.x[ii]
			# 	case_mask = self.mask[ii]
			# 	case_viewgroup = self.viewgroup[ii]


			# 	random.seed(datetime.datetime.now())
			# 	i = random.randint(0, len(case_x)-1)

			# 	# for i in range(len(case_x)):

			# 	case_length = 0
			
			# 	temp_x = []
			# 	temp_mask = []
			# 	temp_viewgroup = []
				
			# 	case_length = 1
			# 	image = case_x[i]
			# 	mask = case_mask[i]

			# 	if self.mode=='Training':
			# 		random.seed(datetime.datetime.now())
			# 		seed = random.randint(0,2**32)

			# 		random.seed(seed)
			# 		image = self.transform0(image)
			# 		random.seed(seed)
			# 		image = self.transform1(image)
					
			# 		random.seed(seed)
			# 		mask = self.transform1(mask)

			# 	image = torchvision.transforms.functional.to_tensor(image)
			# 	mask = torchvision.transforms.functional.to_tensor(mask)

			# 	# image = self.transformNormalize(image)
			# 	# mask = self.transformNormalize(mask)

			# 	mask = mask[0,:,:].unsqueeze(0)

			# 	# result_x.append(image)
			# 	# result_mask.append(mask)
			# 	# result_viewgroup.append(case_viewgroup[i])
			# 	temp_x.append(image)
			# 	temp_mask.append(mask)
			# 	temp_viewgroup.append(case_viewgroup[i])

			# 	result_x.extend(temp_x)
			# 	result_mask.extend(temp_mask)
			# 	result_viewgroup.extend(temp_viewgroup)
			# 	result_length_list.append(case_length)
			# 	result_y.append(self.y[ii])
			# 	result_pid.append(self.pid[ii])


		result_x = torch.stack(result_x, dim=0)
		result_mask = torch.stack(result_mask, dim=0)
		result_y = torch.stack(result_y, dim=0)

		return result_x, result_mask, result_y, result_pid, result_length_list, result_viewgroup



	def mask_to_roi(self, mask):
		mask = numpy.asarray(mask)

		y_0 = mask[:,:,0].sum(axis=1).nonzero()[0][0]
		y_1 = mask[:,:,0].sum(axis=1).nonzero()[0][-1]
		
		box_y0 = max(0, y_0-10)
		box_y1 = y_0+(y_1-y_0)//2

		# import pdb; pdb.set_trace()
		mask1 = numpy.zeros_like(mask)
		mask1[box_y0:box_y1, :, :]=1
		mask2 = mask*mask1

		box_x0 = mask2.sum(axis=0).nonzero()[0][0]+10
		box_x1 = mask2.sum(axis=0).nonzero()[0][-1]-10

		if box_x0>box_x1:
			return None

		result = numpy.zeros_like(mask)
		result[box_y0:box_y1, box_x0:box_x1, :]=255

		result = PIL.Image.fromarray(result)

		return result



	def fuseROI(self, mask, roi):
		mask = numpy.asarray(mask).copy()
		roi = numpy.asarray(roi).copy()
		result = mask+roi
		result[result>0]=255
		result = PIL.Image.fromarray(result)
		return result

	def binaryImage(self, image):
		image = numpy.asarray(image).copy()
		image[image>0]=255
		image = PIL.Image.fromarray(image)
		return image
		













































































