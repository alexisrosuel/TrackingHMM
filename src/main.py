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

	std = 75
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


			particles,average_particle = treat_frame(array_first_picture=array_first_picture, particles=particles, array_picture=array_picture, particle_number=particle_number, std=std)
			

			#Display picture/particles/ellipse
			display_picture(picture=array_picture, particles=particles, average_particle=average_particle)




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
		
			
			particles,average_particle = treat_frame(array_first_picture=array_first_picture, particles=particles, array_picture=array_picture, particle_number=particle_number, std=std)
			

			#Display picture/particles/ellipse
			display_picture_opencv(picture=array_picture_bgr, particles=particles, average_particle=average_particle)

			key = cv2.waitKey(50)


def treat_frame(array_first_picture, particles, array_picture, particle_number, std):

	'''

	Resample each particle according to their weight

	Compute the new position of the child particle with the motion equation

	'''
	#Extract x,y and weight to fit array format
	weights = [particles[i]["weight"] for i in range(particle_number)]
	particles = [particles[i] for i in multinomial_resample(weights)]
	x = [particles[i]["x"] for i in range(particle_number)]
	y = [particles[i]["y"] for i in range(particle_number)]


	particle_dict_aux = {"x":x, "y":y}
	particle_dict_aux = update_particles(particle_dict_aux, std=std)

	#Reformat to fit standard object list
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

		#Set new position only if it does not go outside of picture shape
		if new_x_i < array_first_picture.shape[0] and new_x_i >= 0:
			particle_dict["x"] = new_x_i
			
		if new_y_i < array_first_picture.shape[1] and new_y_i >= 0:
				particle_dict["y"] = new_y_i
		
		#append particle to new particle list
		new_particules.append(particle_dict)

	#Replace
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

	'''
	Find the average particle and cercle
	'''
	
	average_particle = get_average_particle(particles)
			
	return particles, average_particle

def get_average_particle(particles):

	average_particle = {}
	average_particle["x_average"] = 0
	average_particle["y_average"] = 0
	average_particle["c_average"] = 0
	average_particle["r_average"] = 0
	average_particle["radius_average"] = 0

	for particle in particles:
		average_particle["x_average"] += particle["weight"] * particle["x"]
		average_particle["y_average"] += particle["weight"] * particle["y"]

		if "best_cercle" in particle:
			circle = particle["best_cercle"]
			average_particle["c_average"] += particle["weight"] * circle["c"]
			average_particle["r_average"] += particle["weight"] * circle["r"]
			average_particle["radius_average"] += particle["weight"] * circle["radius"]

	average_particle["x_average"] = int(average_particle["x_average"])
	average_particle["y_average"] = int(average_particle["y_average"])
	average_particle["c_average"] = int(average_particle["c_average"])
	average_particle["r_average"] = int(average_particle["r_average"])
	average_particle["radius_average"] = int(average_particle["radius_average"])
	return average_particle

def display_picture(picture ,particles, average_particle):
	
	
	fig,ax = plt.subplots(1)
	

	#Display picture
	ax.imshow(picture)

	#Deactivate autoscale to have clean window
	ax.autoscale(False)

	#Plot particles with weights
	for particle in particles:

		#X ET Y inverse pour pouvoir plot correctement
		ax.scatter(particle["y"], particle["x"], marker="x", s=particle["weight"]*100 )



		#Add circle if particle has key best_cercle
		if "best_cercle" in particle:
			#Add circle
			#ENCORE INVERSE X ET Y 
			circle = particle["best_cercle"]
			circle_patch = Circle(xy=(circle["c"], circle["r"]), radius=circle["radius"])
			circle_patch.set_facecolor("None")
			circle_patch.set_edgecolor("red")
			ax.add_patch(circle_patch)
		
	ax.scatter(average_particle["y_average"], average_particle["x_average"], marker="x", s=200, c="g",linewidth=5.0)

	circle_patch_av = Circle(xy=(average_particle["c_average"], average_particle["r_average"]), radius=average_particle["radius_average"], linewidth=5.0)
	circle_patch_av.set_facecolor("None")
	circle_patch_av.set_edgecolor("green")
	ax.add_patch(circle_patch_av)


	print("Average particle :  {}".format(average_particle))
	plt.show()
	

def display_picture_opencv(picture ,particles, average_particle):
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

	
	cv2.circle(picture, (average_particle["y_average"],average_particle["x_average"]), radius=3, color=(0,255,0), thickness=4)
	cv2.circle(picture, (average_particle["c_average"], average_particle["r_average"]), radius=average_particle["radius_average"], color=(0,255,0),thickness=4)
	cv2.imshow('video test',picture)



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