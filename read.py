import struct
import argparse
import os
from struct import unpack
import numpy as np
from matplotlib import pylab as plt
from scipy import ndimage
import glob


"""return a dictionary that contains country code, image, timestamp,
recognized and key-id as keys nad each has their own value"""
def unpack_drawing(file_handle):
	key_id, = unpack('Q', file_handle.read(8))
	countrycode, = unpack('2s', file_handle.read(2))
	recognized, = unpack('b', file_handle.read(1))
	timestamp, = unpack('I', file_handle.read(4))
	n_strokes, = unpack('H', file_handle.read(2))
	image = []
	for i in range(n_strokes):
		n_points, = unpack('H', file_handle.read(2))
		fmt = str(n_points) + 'B'
		x = unpack(fmt, file_handle.read(n_points))
		y = unpack(fmt, file_handle.read(n_points))
		image.append((x, y))
	
	return {
		'key_id': key_id,
		'countrycode': countrycode,
		'recognized': recognized,
		'timestamp': timestamp,
		'image': image
}

"""generate list of drawings"""
def unpack_drawings(filename):
	with open(filename, 'rb') as f:
		while True:
			try:
				yield unpack_drawing(f)
			except struct.error:
				break

def main():
	parser = argparse.ArgumentParser(description='generating the images for sketch data of a given class.')
	parser.add_argument('--class_name',type=str, help='The name of the class')
	parser.add_argument('--dataset_path',type=str, help='directory for the bin files')
	args = parser.parse_args()
	print('class_name {}'.format(args.class_name))
	print('dataset_path {}'.format(args.dataset_path))

    
	if os.path.isdir(args.class_name):
		print('directory {} already exists'.format(args.class_name))
		return
    
	os.makedirs(args.class_name)

	i = 0
	for drawing in unpack_drawings(args.dataset_path):

		# save the list of drawing
		if drawing['recognized'] == 0:
			continue
		plt.clf()
		plt.axis('off')
		plt.gca().invert_yaxis()
		for (x_path, y_path) in drawing['image']:
			plt.plot(list(x_path), list(y_path), color="black")  

		i+=1
		plt.savefig('{}/drawing{}.png'.format(args.class_name,i), bbox_inches = 'tight')


if __name__=='__main__':
	main()






