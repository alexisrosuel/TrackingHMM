# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:18:03 2016

@author: Alexis
"""

import numpy as np
from skimage import color
from skimage import feature
from skimage import io
from skimage import segmentation
from skimage import draw


def evaluate(image_array, particles):
    """ Blablabla
    
    Args:
    	-image array : matrice numpy de dimension 3 (axes 1, 2 : axes x, y 
     de l'image, axes 3 : R,G,B)
       -particles : vecteur de dictionnaires {x:, y:, weight:}
    	
    Returns:
    	likelihood...
    """
    
    # On converti l'image au format YCbCr
    image_array_YCbCr = rgb2ycbcr(image_array)
    
    # On récupère les pixels correspondant aux particules
    particles_pixels = np.array([image_array_YCbCr[particle['x'],particle['y']] for particle in particles])
    
    # on calcule le likelihood correspondant à chaque particule
    particles = skin_likelihood(particles_pixels, particles)

    # On calcule les cercles
    particles = facial_contour(image_array, particles)
    
    # on calcule les likelihood des cercle
    particles = cercle_likelihood(image_array, particles)
    
    # on conserve pour chaque particule le likelihood du cercle est max
    for particle in particles:
        cercle_likelihood_vector = [cercle['cercle_likelihood'] for cercle in particle['cercles']]
        cercle_max = np.argmax(cercle_likelihood_vector)
        particle['best_cercle'] = particle['cercles'][cercle_max]
    
    # on calcule les likelihood finaux
    lambda1 = 1
    lambda2 = 1
    threshold = 0.8
    for particle in particles:
        if particle['skin_likelihood'] < threshold:
            particle['likelihood'] = 0
        else:
            particle['likelihood'] = lambda1*particle['skin_likelihood'] + lambda2*particle['best_cercle']['cercle_likelihood']
    
    return particles

def rgb2ycbcr(image_RGB):
    """ Transforme une image au format RGB en une image au format YCbCr
    selon la méthode décrite dans l'article. Méthode absente de la 
    version stable de skimage, mais déjà terminée en dev.
    
    Args:
    	matrice numpy de dimension 3 (axes 1, 2 : axes x, y de l'image
     axes 3 : R,G,B)
    	
    Returns:
    	matrice numpy de dimension 3 (axes 1, 2 : axes x, y de l'image
     axes 3 : Y,Cb,Cr)
    """
    
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = image_RGB.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)
    
def skin_likelihood(vector_pixels, particles):
    """ Calcule pour chaque pixel du vecteur la valeur du likelihood
    qui représente la probabilité que le pixel soit de la peau. 
    La distribution utilisée est une loi normale bi-dimensionnelle, où
    la moyenne et la matrice de variance covariance est fixée (valeurs 
    classique qui fonctionnent bien)
    
    Args:
    	matrice numpy de dimension 2 (axe 1 : liste des particules, 
     axe 2 : Y, Cb, Cr)
    	
    Returns:
    	matrice numpy de dimension 1 (axe 1 : liste des likelihood)
    """
    
    # On définit les paramètres de la distribution likelihood
    m = np.matrix([120,150])
    C = np.matrix([[200,-17],
                  [-17,400]]) # a modifier plus tard avec les vrais valeurs
    C_inv = np.linalg.inv(C)

    
    likelihood = np.empty(shape=vector_pixels.shape[0])
    
    for i in range(vector_pixels.shape[0]):
        pixel_color = vector_pixels[i,1:]
        produit_mat = np.dot(np.dot(pixel_color-m,C_inv),(pixel_color-m).T)
        particles[i]['skin_likelihood'] = np.exp(-produit_mat/2)
        
    return particles
    
    
    
    
def cercle_likelihood(image_array, particles):
    # on construit l'image transformée en contour
    image_grey = color.rgb2grey(image_array)
    image_contour = feature.canny(image_grey)
    
    # On construit ensuite l'image des likelihood    
    threshold = 0.5
    image_likelihood_skin = np.empty(shape=image_grey.shape)
    for particle in particles:
        image_likelihood_skin[particle['x'],particle['y']] = 255 if particle['skin_likelihood']>threshold else 0

    # Pour gagner du temps on calcule ici le premier AND
    image_AND = image_contour*image_likelihood_skin
    
    # on constuit enfin chaque ellipse et on enregistre son likelihood associé
    for particle in particles:
        for cercle in particle['cercles']:
            image_cercle = np.zeros(shape=image_grey.shape, dtype=np.uint8)
            rr, cc = draw.circle(cercle['r'], cercle['c'], cercle['radius'])
            index = []
            for i in range(len(rr)):
                if rr[i]<0 or rr[i]>=image_grey.shape[0] or cc[i]<0 or cc[i]>=image_grey.shape[1]:
                    index.append(i)
            rr = np.delete(rr, index)
            cc = np.delete(cc, index)
            
            image_cercle[rr,cc] = 255
            
            # Et on calcule le likelihood de l'ellipse
            image_AND_cercle = image_AND*image_cercle
            cercle['cercle_likelihood'] = np.sum(image_AND_cercle)/np.sum(image_cercle)
    
    return particles
        

def facial_contour(image_array, particles):
    """ 
    
    Args:
    	matrice numpy de dimension 1 (valeur du likelihood) 
    	
    Returns:
    	-
    """
    
    # skin color threshold value
    threshold = 0
    
    # On conserve les particles qui ont un likelihood > threshold
    particles_kept = np.array([particle for particle in particles if particle['skin_likelihood']>threshold])
    
    # Nombre de cercles générées au hasard
    N_cercles = 4
    
    for particle in particles_kept:
        # On crée les objets ellipses
        cercles = np.array([{'r': particle['x'], 
                             'c': particle['y'], 
                             'radius': np.random.randint(low=10.0, high=200.0, size=1)[0], 
                             } for i in range(N_cercles)
                             ])
        particle['cercles'] = cercles

    return particles
    
image = io.imread("..\\..\\scarlett.jpeg")    
#image = io.imread("..\\data\\sequence3\\sequence10000.png")   
    
N_particles = 50

particles = np.array([{'x': np.random.randint(low=0,high=image.shape[0]),
                       'y': np.random.randint(low=0,high=image.shape[1])
                       } for i in range(N_particles)
                      ])
    
particles = evaluate(image,particles)   
