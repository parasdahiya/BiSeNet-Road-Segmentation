import os
import numpy as np
from PIL import Image
import re
import random
from glob import glob
import shutil
import time


def one_hot(gt_image):
	background_color = [255, 0, 0]
	gt_bg = np.all(gt_image == background_color, axis=2)
	gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
	gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
  
	return gt_image


def preprocess_image(image, gt_image, width, height):
  
  image = image.resize((width, height), Image.ANTIALIAS)    # best down-sizing filter
  gt_image = gt_image.resize((width, height), Image.ANTIALIAS)    # best down-sizing filter
  
  image = np.float32(image)
  image = image/255.0
  
  gt_image = np.array(gt_image)
  gt_image = np.float32(one_hot(gt_image))
  
  return image, gt_image
  

def gen_batch_fn(data_dir, width = 1200, height = 350):
  
  image_paths = glob(os.path.join(data_dir, 'image_2', '*.png'))        
  gt_paths = {
      re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path 
      for path in glob(os.path.join(data_dir, 'gt_image_2', '*_road_*.png'))
  }
  
  def get_batch(batch_size):
    
    random.shuffle(image_paths)
    for batch in range(0, len(image_paths), batch_size):
      images = []
      gt_images = []
      
      for image_file in image_paths[batch: batch + batch_size]:
        gt_file = gt_paths[os.path.basename(image_file)]
        
        image = Image.open(image_file)
        gt_image = Image.open(gt_file)
        
        image, gt_image = preprocess_image(image, gt_image, width, height)
        
        #print(image.shape, gt_image.shape)
        
        images.append(image)
        gt_images.append(gt_image)
        
      images = np.array(images)
      gt_images = np.array(gt_images) 
      
      yield images, gt_images
        
  return get_batch


def gen_batch_fn_idd(data_dir, width = 1200, height = 350, datatype='train'):
  
	image_paths = glob(os.path.join(data_dir, 'leftImg8bit', datatype, '*', '*.png'))
	paths = []
	for img_path in image_paths:
		m = os.path.basename(img_path).split('_')[0]
		st = m + '_gtFine_polygons.png'
		n = os.path.dirname(img_path).split('/')
		n[2] = 'gtFine'
		n = '/'.join(n)
		gt_path = os.path.join(n, st)
		paths.append([img_path, gt_path])

	def get_batch(batch_size):
		
		random.shuffle(paths)
		for batch in range(0, len(paths), batch_size):
			images = []
			gt_images = []
			
			for path in paths[batch: batch + batch_size]:
				
				image = Image.open(path[0])
				gt_image = Image.open(path[1])
				
				image, gt_image = preprocess_image(image, gt_image, width, height)
								
				images.append(image)
				gt_images.append(gt_image)
				
			images = np.array(images)
			gt_images = np.array(gt_images) 
			
			yield images, gt_images
  
	return get_batch
    

