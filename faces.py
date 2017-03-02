import random
import cv2
import numpy as np
import os
faces = os.listdir('FACE')
num_faces = len(os.listdir('FACE')) 
print num_faces
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
		items.append(get_image("FACE/"+faces[image_index],im_size,im_size))
	return items

def get_image(image_path, input_height, input_width,
		resize_height=64, resize_width=64,
		is_crop=True, is_grayscale=False):
	image = imread(image_path, is_grayscale)
	return transform(image, input_height, input_width,
	resize_height, resize_width, is_crop)

def save_images(images, size, image_path):
	return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
	if (is_grayscale):
		return cv2.imread(path, flatten = True).astype(np.float)
	else:
		return cv2.imread(path).astype(np.float)

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

def imsave(images, size, path):
	return cv2.imwrite(path, merge(images, size))

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
	if crop_w is None:
		crop_w = crop_h
	h, w = x.shape[:2]
	j = int(round((h - crop_h)/2.))
	i = int(round((w - crop_w)/2.))
	return cv2.resize(
      	x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, 
              resize_height=64, resize_width=64, is_crop=True):
	if is_crop:
		cropped_image = center_crop(
		image, input_height, input_width, 
		resize_height, resize_width)
	else:
		cropped_image = cv2.resize(image, [resize_height, resize_width])
	return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
  	return (images+1.)/2.
