import numpy as np

def sphere_normal(x,y,z):
    """
    Generates sphere normals (pointing inwards) at a given location.
    
    Parameters:
        x (int): x coordinate
        y (int): y coordinate
        z (int): z coordinate
    
    Returns:
        a numpy array
    """
    norm = (x**2+y**2+z**2)**(1/2)
    return np.array([-x/norm,-y/norm,-z/norm])


def cylinder_normal(x,y,z):
    """
    Generates cylinder normals (pointing inwards) at a given location
    
    Parameters:
        x (int): x coordinate
        y (int): y coordinate
        z (int): z coordinate
    
    Returns:
        a numpy array
    """
    norm = (x**2+z**2)**(1/2)
    return np.array([-x/norm,0,-z/norm])