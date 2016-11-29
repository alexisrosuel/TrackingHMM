# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:18:03 2016

@author: Alexis
"""

import numpy as np
import skimage

def transf_RGB_to_YCbCr(matrice_RGB):
    """ Transforme UNE image sous forme de matrice RGB vers une matrice 
    YCbCr selon la méthode décrite dans l'article.
    
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
    
    matrice2 = np.matrix([16],
                         [128],
                         [128]
                         )
    
    return skimage.color.rgb2ycbcr(matrice_rgb)
    
    
    