# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:13:17 2016

@author: Pascal
"""



import sys, argparse,glob
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse



"""
Execute main script

"""


def main(argv):
	parser = argparse.ArgumentParser(description='Main script')
	parser.add_argument('--source', required=True, help='source for the pictures')
	args = parser.parse_args()
	source = args.source
	


	if(source != "webcam"):
		file_list = glob.glob("../data/" + source + "/*")
		print(file_list)
		
		#Get shape of picture to initialize weights
		particle_number = 500
		weights = np.array([(1.0/particle_number) for x in range(particle_number)])
		print(weights)


		for file_picture in sorted(file_list):
			print(file_picture)
			array_picture = io.imread(file_picture, as_grey=False)


			'''
			Generate particles

			'''

			'''
			Predict

			'''

			# particles = generate_particle(shape, weights, particle_number) 

			

			'''
			Update weights

			'''

			# weights = update_weights(particles, array_picture)




			'''
			Evaluate

			'''

			#ellipse = evaluate(particles, weight ...)


			'''
			Plot picture

			Plot particle proportionally to weights

			Plot ellipse

			'''
			

			#Random data example to show you the structure
			particles = {}
			particles["x"] = np.random.rand(particle_number) * array_picture.shape[1]
			particles["y"] = np.random.rand(particle_number) * array_picture.shape[0]

			weights = np.random.rand(particle_number)
			

			ellipse = {}
			ellipse["xy"] = [np.random.rand(1) * array_picture.shape[1],np.random.rand(1) * array_picture.shape[0]]
			ellipse["angle"] = np.random.randint(181)
			ellipse["height"] = np.random.randint(50,150)

			#Display picture/particles/ellipse
			display_picture(picture=array_picture, ellipse=ellipse, particles=particles, weights=weights)




	else:
		#TODO WEBCAM CODE
		pass


	
		

def display_picture(picture ,ellipse, particles, weights):
	
	
	fig,ax = plt.subplots(1)
	

	#Display picture
	ax.imshow(picture)

	#Deactivate autoscale to have clean window
	ax.autoscale(False)

	#Plot particles with weights
	ax.scatter(particles["x"], particles["y"], marker="x", s=weights*100 )

	#Add ellipse
	ells = Ellipse(xy=ellipse["xy"], width=20, height=ellipse["height"], angle=ellipse["angle"])
	ax.add_patch(ells)
	
	
	plt.show()
	




if __name__ == "__main__":
    main(sys.argv)