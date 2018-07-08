
import argparse
import numpy as np
from scipy.spatial import distance 



def load_files(file_name):
	res = np.load(file_name)
	labels = res['arr_0'][0]
	centriods = res['arr_0'][1]
	return labels, centriods


def distance_matrix(centriods1, centriods2):	
	dst = distance.cdist(centriods1,centriods2, 'euclidean')
	return dst
	


def main():
	parser = argparse.ArgumentParser(description='Generate the list of conceptuasl shifts categories.')
	parser.add_argument('--max_num_cs',type=int, help='The maximum number of conceptual shifts for each category')
	args = parser.parse_args()
	print('max_num_cs {}'.format(args.max_num_cs))

	categories = ['aircraft-carrier', 'airplane', 'alarm-clock', 'ambulance',
	              'angel', 'ant', 'apple', 'arm', 'axe','backpack', 'bananas',
	              'baseball', 'basketball', 'bathtub', 'beach', 'blackberry', 
	              'brain', 'bridge' 'calculator', 'carrot', 'cat' ,'ceiling-fan', 
	              'cell phone', 'church', 'computer', 'cookie', 'cooler', 'crocodile',
	              'crown', 'dolphin', 'donut', 'dumbbell', 'eye', 'eraser', 
	              'finger', 'fish', 'flower', 'grapes', 'hamburger', 'hand','hotdog', 
	              'ice cream', 'knee', 'leg', 'mermaid', 'moustache', 'mouth'
	              'mushroom', 'nose', 'ocean', 'penguin', 'pineapple', 'potato',
	              'rainbow', 'roller-coaster', 'shark', 'sheep', 'smiley face', 
	              'snail', 'strawberry', 'tooth', 'toothbrush', 'tree', 'turtle', 'zebra'] 

	
	for cat1 in categories:
		lb1, cn1 = load_files('kmeans-results/kmeans_{}.npz'.format(cat1))
		vc = []
		ls = []
		for cat2 in categories:
			 lb2, cn2 = load_files('kmeans-results/kmeans_{}.npz'.format(cat2))
			 if cat1 == cat2:
				continue

			 dm = distance_matrix(cn1,cn2)
			 d = np.argmin(dm)
			 ds = np.min(dm)

			 d = np.unravel_index(d, (20,20))

			 vc.append(ds)

			 ls.append((d,cat1,cat2))
		
		position = np.argsort(vc)[:args.max_num_cs]
		print([("{:.3f}".format(vc[x]) ,ls[x]) for x in position])


if __name__=='__main__':
	main()







