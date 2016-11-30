# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:13:17 2016

@author: Alexis
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