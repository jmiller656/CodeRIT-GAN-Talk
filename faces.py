import random
import cv2
import numpy as np
import os
import scipy.misc
imdir = 'faces'
faces = os.listdir(imdir)
num_faces = len(os.listdir(imdir)) 
def get_batch(im_size,batch_size):
	items = []
	for i in range(batch_size):
		image_index = random.randint(0,num_faces-1)
		"""temp = cv2.imread("FACE/"+faces[image_index])
		temp = cv2.resize(temp,(im_size,im_size))
		temp = temp.astype(np.float32)
		#temp -= 128.0
		#temp /= 255.print temp
		temp /= 127.5
		temp -= 1"""
		items.append(get_image(imdir+"/"+faces[image_index],im_size,im_size))
	return items

def get_image(image_path, input_height, input_width,
		resize_height=64, resize_width=64,
		is_crop=False, is_grayscale=False):
	image = imread(image_path, is_grayscale)
	return transform(image, input_height, input_width,
	resize_height, resize_width, is_crop)

def save_images(images, size, image_path):
	return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
	img = scipy.misc.imread(path)
	img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB).astype(np.float)
	return img 

def merge_images(images, size):
	return inverse_transform(images)

def merge(images, size):
	h, w = images.shape[1], images.shape[2]
	img = np.zeros((h * size[0], w * size[1], 3))
	for idx, image in enumerate(images):
		i = idx % size[1]
		j = idx // size[1]
		img[j*h:j*h+h, i*w:i*w+w, :] = image
	return img

def imsave(images, path):
	return scipy.misc.imsave(path, inverse_transform(images))

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
	if crop_w is None:
		crop_w = crop_h
	h, w = x.shape[:2]
	j = int(round((h - crop_h)/2.))
	i = int(round((w - crop_w)/2.))
	return scipy.misc.imresize(
      	x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, 
              resize_height=64, resize_width=64, is_crop=False):
	if is_crop:
		cropped_image = center_crop(
		image, input_height, input_width, 
		resize_height, resize_width)
	else:
		cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
	return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
  	return (images+1.)/2.
