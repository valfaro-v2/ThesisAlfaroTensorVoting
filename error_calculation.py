import numpy as np
import math

from pycurv_testing.errors_calculation import (
    error_vector, relative_error_scalar, absolute_error_scalar)

from pycurv import (
    TriangleGraph, PointGraph, normals_directions_and_curvature_estimation,
    nice_asin)
from pycurv import nice_acos

def angular_error_vector_corrected(true_vector, estimated_vector):
    """
    Calculates the "angular error" for 3D vectors.

    Args:
        true_vector (numpy.ndarray): true / accepted 3D vector
        estimated_vector (numpy.ndarray): estimated / measured / experimental 3D
            vector

    Returns:
        acos(abs(np.dot(true_vector, estimated_vector)))
        angle in radians between two vectors
    """
    angular_error = nice_acos(np.dot(true_vector, estimated_vector))
    return angular_error

def calculate_errors(tg,true_kappa_1, true_kappa_2):
    """
    Given a triangle/point graph provided with its 
                                -original face/vertex normals
                                -true normals
                                -estimated normals
                                -true principal curvatures
                                -estimated principal curvatures
    calculates different error measures
    
    Args:
        tg: triangle/point graph object
        true_kappa_1 (float): first principal curvature
        true_kappa_2 (float): second principal curvature
        
    Returns:
        several lists indexed by the vertices of the tg object: estimated face normals, original normals, true normals,
        estimated first principal curvature, estimated second principal curvature, normal errors after the estimation,
        original normal errors, angular normal errors after the estimation, original angular normal errors,
        absolute first pricipal curvature errors, absolute second principal curvature errors,
    """
    pos=[0,1,2]

    #get initial normals
    initial_normals = tg.graph.vertex_properties['normal'].get_2d_array(pos)

    #get true normals
    true_normals = tg.graph.vertex_properties['true_normal'].get_2d_array(pos)

    #get estimated normals
    estimated_normals = tg.graph.vertex_properties['n_v'].get_2d_array(pos)
    
    #these arrays are of the size (3xn). It is necessary to transpose them so that
    #they are of the size (nx3)

    initial_normals = np.transpose(initial_normals)
    estimated_normals = np.transpose(estimated_normals)
    true_normals = np.transpose(true_normals)

    
    # calculation of normal errors
    normal_errors = np.array([error_vector(estimated_normal,true_normal) for estimated_normal,
                             true_normal in zip(estimated_normals,true_normals)])

    initial_normal_errors = np.array([error_vector(initial_normal,true_normal) for initial_normal,
                             true_normal in zip(initial_normals,true_normals)])

    angular_normal_errors = np.array([angular_error_vector_corrected(estimated_normal,true_normal) for estimated_normal,
                             true_normal in zip(estimated_normals,true_normals)])

    initial_angular_normal_errors = np.array([angular_error_vector_corrected(initial_normal,true_normal) for initial_normal,
                             true_normal in zip(initial_normals,true_normals)])

    # get estimated ppal curvatures
    estimated_kappa_1 = tg.graph.vertex_properties['kappa_1'].get_array()
    estimated_kappa_2 = tg.graph.vertex_properties['kappa_2'].get_array()
    
    # calculation of ppal curvature errors
    abs_kappa_1_errors = np.array([absolute_error_scalar(true_kappa_1,kappa)
                      for kappa in estimated_kappa_1])
    abs_kappa_2_errors = np.array([absolute_error_scalar(true_kappa_2, kappa)
                      for kappa in estimated_kappa_2])
    
    return estimated_normals, initial_normals, true_normals, estimated_kappa_1, estimated_kappa_2,normal_errors, initial_normal_errors, angular_normal_errors, initial_angular_normal_errors, abs_kappa_1_errors, abs_kappa_2_errors