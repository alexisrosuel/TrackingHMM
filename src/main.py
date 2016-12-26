# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:13:17 2016

@author: Pascal
"""



import sys, argparse,glob
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Ellipse, Circle

from observation import evaluate
from particle_filtering import multinomial_resample, update_particles

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
	particle_number = 100

	std = 20
	if(source != "webcam"):
		file_list = glob.glob("../data/" + source + "/*")
		print(file_list)
		
		
		

		


		
		#Initialization
		particles = []
		sorted_file_list = sorted(file_list)
		array_first_picture = io.imread(sorted_file_list[0], as_grey=False)


		for _ in range(particle_number):
			particle = {}
			#X ET Y INVERSE POUR POUVOIR PLOT CORRECTEMENT
			particle["x"] = int(np.random.rand(1) * array_first_picture.shape[0])
			particle["y"] = int(np.random.rand(1) * array_first_picture.shape[1])
			particle["weight"] = 1/particle_number
			particles.append(particle)


		for file_picture in sorted_file_list:
			print(file_picture)
			array_picture = io.imread(file_picture, as_grey=False)


			'''
			Generate particles

			'''

			weights = [particles[i]["weight"] for i in range(particle_number)]
			particles = [particles[i] for i in multinomial_resample(weights)]

			print("Particle number : {}".format(len(particles)))

			x = [particles[i]["x"] for i in range(particle_number)]
			y = [particles[i]["y"] for i in range(particle_number)]


			'''
			Predict

			'''


			particle_dict_aux = {"x":x, "y":y}

			'''

			We can play here with the STD parameter to handle how the algorithm can catch distant faces

			'''

			particle_dict_aux = update_particles(particle_dict_aux, std=std)


			new_x = particle_dict_aux["x"]
			new_y = particle_dict_aux["y"]

			new_particules = []
			for i in range(particle_number):
				#Position en int 
				new_x_i = int(new_x[i])
				new_y_i = int(new_y[i])
				
				'''
				old debug
				if (i<10):
					print("\n")
					print(new_x[i])
					print(new_x_i)
					print(array_first_picture.shape[0])
					print("\n")
				'''	
				
				#Deep copy pour pas avoir d'effet de bord
				particle_dict = {}
				particle_dict["x"] = particles[i]["x"]
				particle_dict["y"] = particles[i]["y"]
				particle_dict["weight"] = particles[i]["weight"]

				if new_x_i < array_first_picture.shape[0] and new_x_i >= 0:
					particle_dict["x"] = new_x_i
					
					'''
					old debug
					if i<10:
						print("New value : {}".format(new_x_i))
						print("Particle dict : {}".format(particle_dict["x"]))
					'''
					
				
				if new_y_i < array_first_picture.shape[1] and new_y_i >= 0:
 					particle_dict["y"] = new_y_i
				
				#construire la nouvelle liste
				new_particules.append(particle_dict)

				'''
				if i<10:
					print("Particle : {}".format(new_particules[i]["x"]))
				'''
 				

			#remplacer par la nouvelle liste
			particles = new_particules
			#print("10 New particles x: {}".format([particle["x"] for particle in particles[0:10]]))

			#print("-------------")


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
		import cv2

		cap = cv2.VideoCapture(0)

		particles = []
		ret,array_first_picture = cap.read()
		
		for _ in range(particle_number):
			particle = {}
			#X ET Y INVERSE POUR POUVOIR PLOT CORRECTEMENT
			particle["x"] = int(np.random.rand(1) * array_first_picture.shape[0])
			particle["y"] = int(np.random.rand(1) * array_first_picture.shape[1])
			particle["weight"] = 1/particle_number
			particles.append(particle)


		while True:
			ret,array_picture_bgr = cap.read()
			array_picture = cv2.cvtColor(array_picture_bgr, cv2.COLOR_BGR2RGB)
		

			weights = [particles[i]["weight"] for i in range(particle_number)]
			particles = [particles[i] for i in multinomial_resample(weights)]
			x = [particles[i]["x"] for i in range(particle_number)]
			y = [particles[i]["y"] for i in range(particle_number)]


			particle_dict_aux = {"x":x, "y":y}
			particle_dict_aux = update_particles(particle_dict_aux, std=std)

			new_x = particle_dict_aux["x"]
			new_y = particle_dict_aux["y"]

			new_particules = []
			for i in range(particle_number):
				#Position en int 
				new_x_i = int(new_x[i])
				new_y_i = int(new_y[i])
				
				
				#Deep copy pour pas avoir d'effet de bord
				particle_dict = {}
				particle_dict["x"] = particles[i]["x"]
				particle_dict["y"] = particles[i]["y"]
				particle_dict["weight"] = particles[i]["weight"]

				if new_x_i < array_first_picture.shape[0] and new_x_i >= 0:
					particle_dict["x"] = new_x_i
					
					
					
				
				if new_y_i < array_first_picture.shape[1] and new_y_i >= 0:
 					particle_dict["y"] = new_y_i
				
				#construire la nouvelle liste
				new_particules.append(particle_dict)

				
			particles = new_particules
			
			'''
			Compute the likelihood of each particle

			Find the best cercle of each particle

			'''
			particles = evaluate(array_picture, particles)



			'''
			Update and normalize weights with their likelihood

			'''

			particles = update_weights(particles)
			


			#Display picture/particles/ellipse
			display_picture_opencv(picture=array_picture_bgr, particles=particles)

			key = cv2.waitKey(100)



		

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
	

def display_picture_opencv(picture ,particles):
	import cv2
	

	#Plot particles with weights
	for particle in particles:

		#X ET Y inverse pour pouvoir plot correctement
		cv2.circle(picture, (particle["y"],particle["x"]), radius=int(particle["weight"]*20), color=(255,0,0), thickness=3)


		#Add circle if particle has key best_cercle
		if "best_cercle" in particle:
			#Add circle
			#ENCORE INVERSE X ET Y 
			circle = particle["best_cercle"]
			cv2.circle(picture, (circle["c"], circle["r"]), radius=int(circle["radius"]), color=(0,0,255),thickness=1)

	
	cv2.imshow('video test',picture)



##TODO : optimiser cette fonction
'''
Fixe les poids des particules en fonction de leur likelihood, et normalise pour avoir une somme égale a 1 en passant par les logarithmes.

Args:
    dictionnaire de particle (clé utilisée : "likelihood" et "weight")
    	
Returns:
	dictionnaire de particle avec clé "weight" mise à jour
'''
def update_weights(particles):
	'''
	normalizing_constant = sum([particle["likelihood"] for particle in particles])

	for particle in particles :
		particle["weight"] = particle["likelihood"] / normalizing_constant 
		print(particle["weight"])
		print("\n")
	'''

	likelihood_list = [particle["likelihood"] for particle in particles if not(particle["likelihood"]==0)]
	#print("likelihood list : {}".format(likelihood_list))

	sum_likelihood = sum(likelihood_list)
	#print("Sum of likelihood : {}".format(sum_likelihood))


	normalizing_constant = math.log(sum_likelihood)
	#print("Log of sum : {}".format(normalizing_constant))

	for particle in particles :
		if not(particle["likelihood"] == 0):
			ll = math.log(particle["likelihood"])
			#print("Log likelihood : {}".format(ll))
			
			#print("Log sum : {}".format(normalizing_constant))

			particle["weight"] = math.exp(ll - normalizing_constant)

		else:
			#print("zero")
			particle["weight"] = 0
		

		#print(particle["weight"])
		#print("\n")


	#print("Sum of weights : ")
	#print(sum([particle["weight"] for particle in particles]))
	return particles


if __name__ == "__main__":
	main(sys.argv)