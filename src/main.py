# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:13:17 2016

@author: Pascal
"""



import sys, argparse,glob
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Circle

from observation import evaluate

"""
Execute main script

Exemple of execution :
python3 main.py --source "sequence1"

"""


def main(argv):
	parser = argparse.ArgumentParser(description='Main script')
	parser.add_argument('--source', required=True, help='source for the pictures')
	args = parser.parse_args()
	source = args.source
	
	#TODO => passer cette valeur en argument du script ?
	particle_number = 400


	if(source != "webcam"):
		file_list = glob.glob("../data/" + source + "/*")
		print(file_list)
		
		
		


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

			


			#Random data example to show you the structure
			particles = []
			for _ in range(particle_number):
				particle = {}
				#X ET Y INVERSE POUR POUVOIR PLOT CORRECTEMENT
				particle["x"] = int(np.random.rand(1) * array_picture.shape[0])
				particle["y"] = int(np.random.rand(1) * array_picture.shape[1])
				particle["weight"] = np.random.rand(1)

				particles.append(particle)


			'''
			Compute the likelihood of each particle

			Find the best cercle of each particle

			'''
			particles = evaluate(array_picture, particles)



			'''
			Update and normalize weights with their likelihood

			'''

			particles = update_weights(particles)
			


			'''
			Plot picture

			Plot particle proportionally to weights

			Plot ellipse/circle

			'''
			

			
			
			'''
			ellipse = {}
			ellipse["x"] = np.random.rand(1) * array_picture.shape[0]
			ellipse["y"] = np.random.rand(1) * array_picture.shape[0]
            ellipse["angle"] = np.random.randint(181)
			ellipse["height"] = np.random.randint(50,150)
			'''

			#Display picture/particles/ellipse
			display_picture(picture=array_picture, particles=particles)




	else:
		#TODO WEBCAM CODE
		pass


	
		

def display_picture(picture ,particles):
	
	
	fig,ax = plt.subplots(1)
	

	#Display picture
	ax.imshow(picture)

	#Deactivate autoscale to have clean window
	ax.autoscale(False)

	#Plot particles with weights
	for particle in particles:

		#X ET Y inverse pour pouvoir plot correctement
		ax.scatter(particle["y"], particle["x"], marker="x", s=particle["weight"]*100 )



		'''
		#Add ellipse
		ells = Ellipse(xy=ellipse["xy"], width=20, height=ellipse["height"], angle=ellipse["angle"])
		ax.add_patch(ells)
		'''

		#Add circle if particle has key best_cercle
		if "best_cercle" in particle:
			#Add circle
			#ENCORE INVERSE X ET Y 
			circle = particle["best_cercle"]
			circle_patch = Circle(xy=(circle["c"], circle["r"]), radius=circle["radius"])
			circle_patch.set_facecolor("None")
			circle_patch.set_edgecolor("red")
			ax.add_patch(circle_patch)
		
	
	plt.show()
	


##TODO : optimiser cette fonction
'''
Fixe les poids des particules en fonction de leur likelihood, et normalise pour avoir une somme égale a 1.

Args:
    dictionnaire de particle (clé utilisée : "likelihood" et "weight")
    	
Returns:
	dictionnaire de particle avec clé "weight" mise à jour
'''
def update_weights(particles):
	normalizing_constant = sum([particle["likelihood"] for particle in particles])

	for particle in particles :
		particle["weight"] = particle["likelihood"] / normalizing_constant 
		print(particle["weight"])
		print("\n")
	
	return particles


if __name__ == "__main__":
    main(sys.argv)