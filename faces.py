import random
import cv2
import numpy as np
import os
faces = os.listdir('faces')
num_faces = len(os.listdir('faces')) 

def get_batch(im_size,batch_size):
	items = []
	for i in range(batch_size):
		image_index = random.randint(0,num_faces)
		temp = cv2.imread("faces/"+faces[image_index])
		temp = cv2.resize(temp,(im_size,im_size))
		#temp = np.asarray([temp,temp,temp])
		temp = temp.astype(np.float32)
		temp -= 128.0
		temp /= 255.0
		items.append(temp)
	return items
