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


particles=np.array([{'x':150,'y':600,'weigh':4},{'x':400,'y':200,'weigh':10}])

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
    particles_likelihood = np.array([image_array_YCbCr[particle['x'],particle['y']] for particle in particles])
    
    # on calcule le likelihood correspondant à chaque particule
    particles_likelihood = skin_likelihood(particles_likelihood)
    
    # on ajoute cette valeur au dictionnaire des particules
    for i in range(len(particles)):
        particles[i]['likelihood_skin'] = particles_likelihood[i]

    # On calcule les ellipses
    #ellipses = facial_contour(image_array, particles)
    
    # on calcule les likelihood des ellipses
    
    
    # on calcule les likelihood finaux
#    lambda1 = 1
#    lambda2 = 1
#    threshold = 0
#    for particle in particles:
#        if particle['likelihood_skin'] < threshold:
#            particle['likelihood'] = 0
#        else:
#            particle['likelihood'] = lambda1*particle['likelihood_skin'] + lambda2*particle['likelihood_ellipse']
#    
    #return ellipses

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
    m = np.matrix([120,150])
    C = np.matrix([[200,-17],
                  [-17,400]]) # a modifier plus tard avec les vrais valeurs
    C_inv = np.linalg.inv(C)
    
    likelihood = np.empty(shape=vector_particles.shape[0])
    
    for i in range(vector_particles.shape[0]):
        pixel_color = vector_particles[i,1:]
        produit_mat = np.dot(np.dot(pixel_color-m,C_inv),(pixel_color-m).T)
        likelihood[i] = np.exp(produit_mat/2)
        
    return likelihood
    
    
    
    
def ellipse_likelihood(image_array, particles):
    # on construit l'image transformée en contour
    image_grey = color.rgb2grey(image_array)
    image_contour = feature.canny(image_grey)
        
    threshold = 1.5
    image_likelihood_skin = np.empty(shape=image_array.shape[0:2])
    for particle in particles:
        print(particle['likelihood_skin'])
        image_likelihood_skin[particle['x'],particle['y']] = 255 if particle['likelihood_skin']>threshold else 0
    
    image_and = image_contour*image_likelihood_skin # AND operation
    io.imshow(image_and)
    print(np.max(image_and))
    for particle in particles:
        points = particle['ellipse_contour']
        rr, cc = skimage.draw.polygon(points.T[0],points.T[1])
        image_ellipse = np.copy(image)
        image_ellipse[rr, cc] = 1
        image_test = image_ellipse*image_and
        particle['likelihood_ellipse'] = np.sum(image_test)/np.sum(image_ellipse)
        
        
image = io.imread("..\\data\\sequence3\\sequence10000.png")     
#particles[0]['ellipse_contour'] = 
evaluate(image,particles)
ellipse_likelihood(image,particles)     


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
    particles_kept = np.array([particle for particle in particles if particle['likelihood_skin']>threshold])
    
    # On crée les objets ellipses
    ellipses = np.array([{'x': particle['x'], 'y':particle['y']} for particle in particles_kept])
    
    # on crée une ellipse à partir des particles['x','y']
    M=100 #points utilisés pour construire l'ellipse
    
    s = np.linspace(0, 2*np.pi,100)
    init = 25*np.array([np.cos(s), np.sin(s)]).T+[ellipses[1]['x'],ellipses[1]['y']]         
    #elps = np.array(draw.ellipse_perimeter(ellipses[1]['x'],ellipses[1]['y'],5,5))  
    #print(elps.shape)       

          
    image_contour = segmentation.active_contour(image_array, snake=init)
    #print(image_contour)
    return image_contour
   
    
    #io.imshow(image_contour)
    #image_contour = measure.find_contours(image_grey, level=0)
    #io.imshow(image_contour)

#image = io.imread("..\\..\\scarlett.jpeg")
#image = io.imread("..\\data\\sequence3\\sequence10000.png")
"""
image_ycbcr = rgb2ycbcr(image)
skimage.io.imshow(image_ycbcr)
"""
#points = evaluate(image,particles)
#print(points)
#
#io.imshow(image_contour)