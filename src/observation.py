# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:18:03 2016

@author: Alexis
"""

import numpy as np
import skimage

def transf_RGB_to_YCbCr(vector_pixels_RGB):
    """ Transforme un vecteur de pixel au format RGB en un vecteur de
    pixel au format YCbCr selon la méthode décrite dans l'article. Méthode absente de la 
    version stable de skimage, mais déjà terminée en dev.
    
    Args:
    	matrice numpy de dimension 3 (axes 1, 2 : axes x, y de l'image
     axes 3 : R,G,B)
    	
    Returns:
    	matrice numpy de dimension 3 (axes 1, 2 : axes x, y de l'image
     axes 3 : Y,Cb,Cr)
    """
    
    matrice1 = np.matrix([[65.481, 128.553, 24.966], 
                          [-37.797, -74.203, 112.000],
                          [112.000, -93.786, -18.214]
                          ])
    
    matrice2 = np.matrix([16,128,128])
    
    vector_pixels_YCbCr = np.empty(shape=vector_pixels_RGB.shape)
    for i in range(vector_pixels_RGB.shape[0]):
        vector_pixels_YCbCr[i] = np.dot(matrice1,vector_pixels_RGB[i])+matrice2
    
    return vector_pixels_YCbCr
    """
    140<Cb<195 and 140<Cr<165
    """
    

def skin_likelihood_pixel(pixel):
    """ Calcule la valeur du likelihood d'un pixel.
    La distribution utilisée est une loi normale bi-dimensionnelle, où
    la moyenne et la matrice de variance covariance est fixée (valeurs 
    classique qui fonctionnent bien)
    
    Args:
    	matrice numpy de dimension 1 (Y, Cr, Cb) 
    	
    Returns:
    	matrice numpy de dimension 1 (valeur du likelihood)
    """
    
    m = np.matrix([100,100])
    C = np.matrix([[0.01,0.05],
                  [0.05,0.1]]) # a modifier plus tard avec les vrais valeurs
    C_inv = np.linalg.inv(C)
   
    pixel_color = pixel[1:] # on ne prend pas en compte Y
    produit_mat = np.dot(np.dot(pixel_color-m,C_inv),(pixel_color-m).T)
    
    return np.exp(produit_mat/2)
    
def skin_likelihood_image(vector_of_pixels_YCbCr):
    """ Calcule pour chaque pixel du vecteur la valeur du likelihood
    qui représente la probabilité que le pixel soit de la peau. 
    
    Args:
    	matrice numpy de dimension 2 (axe 1 : liste des particules, 
     axe 2 : Y, Cb, Cr)
    	
    Returns:
    	matrice numpy de dimension 1 (axe 1 : liste des likelihood)
    """
    
    likelihood = np.empty(shape=vector_of_pixels_YCbCr.shape)
    for x in range(vector_of_pixels_YCbCr.shape[0]):
        likelihood[x] = skin_likelihood_pixel(vector_of_pixels_YCbCr[x])
        
    return likelihood
    
""" Pour tester, lancer les lignes suivantes :
    
image = skimage.io.imread("..\\data\\sequence3\\sequence10000.png")
vector_pixels_RGB = np.array([image[1,1],image[10,10],
                            image[100,100],image[150,150]])

vector_pixels_YCbCr = transf_RGB_to_YCbCr(vector_pixels_RGB)
likelihood = skin_likelihood_image(vector_pixels_YCbCr)
print(likelihood)
"""

def facial_contour(vector_likelihood):
    """ Calcule la valeur du likelihood d'un pixel.
    La distribution utilisée est une loi normale bi-dimensionnelle, où
    la moyenne et la matrice de variance covariance est fixée (valeurs 
    classique qui fonctionnent bien)
    
    Args:
    	matrice numpy de dimension 1 (valeur du likelihood) 
    	
    Returns:
    	matrice numpy de dimension 0 (valeur du likelihood)
    """
    
    # skin color threshold value
    threshold = 10