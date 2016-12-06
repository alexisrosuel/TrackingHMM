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

particles=np.array([{'x':10,'y':20,'weigh':4},{'x':150,'y':420,'weigh':10}])

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
    
    # récupère les pixels correspondant aux particules
    particles_likelihood = np.array([image_array_YCbCr[particle['x'],particle['y']] for particle in particles])
    
    # on calcule le likelihood correspondant à chaque particule
    particles_likelihood = skin_likelihood(particles_likelihood)
    
    # on ajoute cette valeur au dictionnaire des particules
    for i in range(len(particles)):
        particles[i]['likelihood'] = particles_likelihood[i]

    # On calcule les ellipses
    ellipses = facial_contour(image_array_YCbCr, particles)
    
    return ellipses#, weights

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
    
def skin_likelihood(vector_particles):
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
    m = np.matrix([118,136])
    C = np.matrix([[4296.3,-1712.1],
                  [-1712.1,1957.8]]) # a modifier plus tard avec les vrais valeurs
    C_inv = np.linalg.inv(C)
    
    likelihood = np.empty(shape=vector_particles.shape[0])
    
    for i in range(vector_particles.shape[0]):
        pixel_color = vector_particles[i,1:]
        produit_mat = np.dot(np.dot(pixel_color-m,C_inv),(pixel_color-m).T)
        likelihood[i] = np.exp(produit_mat/2)
        
    return likelihood


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
    particles_kept = np.array([particle for particle in particles if particle['likelihood']>threshold])
    
    # On crée les objets ellipses
    ellipses = np.array([{'x': particle['x'], 'y':particle['y']} for particle in particles_kept])
    
    # on crée une ellipse à partir des particles['x','y']
    M=100 #points utilisés pour construire l'ellipse
    
    #image_grey = color.rgb2grey(image)
    #image_contour = feature.canny(image_grey)
    #io.imshow(image_contour)
    #image_contour = measure.find_contours(image_grey, level=0)
    #io.imshow(image_contour)
    
    s = np.linspace(0, 2*np.pi,100)
    init = 25*np.array([np.cos(s), np.sin(s)]).T+[ellipses[1]['x'],ellipses[1]['y']]
#    print(ellipses[1]['x'])
#    print(ellipses[1]['y'])          
#    elps = np.array(draw.ellipse_perimeter(ellipses[1]['x'],ellipses[1]['y'],5,5))  
#    print(elps.shape)       

          
    image_contour = segmentation.active_contour(image_array, snake=init)
    #print(image_contour)
    return image_contour
   
    

image = skimage.io.imread("..\\data\\sequence3\\sequence10000.png")
"""
image_ycbcr = rgb2ycbcr(image)
skimage.io.imshow(image_ycbcr)
"""
points = evaluate(image,particles)
#print(points)
rr, cc = skimage.draw.polygon(points.T[0],points.T[1])
image_contour = image
image_contour[rr, cc] = 1
io.imshow(image_contour)