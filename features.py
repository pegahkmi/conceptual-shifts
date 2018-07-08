"""extract convolutional neural network features"""
import argparse
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
import numpy as np

#parameters 
batch_size = 32


def main():
	parser = argparse.ArgumentParser(description='Extract the features for each class.')
	parser.add_argument('--max_images',type=int, help='The maximum number of sketches for training')
	parser.add_argument('--image_path',type=str, help='path to the images')
	args = parser.parse_args()
	print('max_images {}'.format(args.max_images))
	print('image_path {}'.format(args.image_path))


	base_model = VGG16(weights='imagenet')
	model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
	images = ['{}/drawing{}.png'.format(args.image_path,i) for i in range(1,args.max_images+1)]
	features = []
	for indx in range(0,len(images),batch_size):
		batch=[]
		for img_path in images[indx:indx+batch_size]:
			img = image.load_img(img_path, target_size=(224, 224))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			batch.append(preprocess_input(x))
		print(images[indx])

		batch = np.concatenate(batch,axis=0)	
		features.append(model.predict(batch))

	feat = np.concatenate(features,axis=0)
	np.save("features_nose",feat)


if __name__=='__main__':
	main()
