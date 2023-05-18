import vtk

from pycurv_testing.synthetic_surfaces import remove_non_triangle_cells
from pycurv.surface import reverse_sense_and_normals



warnings = vtk.vtkOutputWindow()
warnings.GlobalWarningDisplayOff()

def modified_generate_plane_surface(x_half_size=10, y_half_size=10,
                                   x_res=30, y_res=30):
    """
    Generates a rectangular plane surface with triangular cells.
    
    Parameters:
        x_half_size (int): half size of the base of the rectangular
                            plane
        y_half_size (int): half size of the height of the rectangular
                      plane
        x_res (int): number of subdivisions of the plane along
                        the x axis
        y_res (int): number of subdivisions of the plane along the 
                        y axis
    
    Returns:
        a plane surface (vtk.vtkPolyData)
    """
    
    plane = vtk.vtkPlaneSource()
    
    plane.SetNormal(0,0,1) #plane is perpendicular to Z axis
    plane.SetOrigin(-x_half_size,-y_half_size,0)
    plane.SetPoint1(x_half_size,-y_half_size,0)
    plane.SetPoint2(-x_half_size, y_half_size,0)
    plane.SetResolution(x_res, y_res)
    
    #Now we have to triangulate the plane
    
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(plane.GetOutputPort())
    tri.Update()
    
    plane_surface = tri.GetOutput()
    
    print(f'The plane has {plane_surface.GetNumberOfCells()} cells')
    print(f'The center of the plane is {plane.GetCenter()}')
    
    return plane_surface







def generate_uncapped_cylinder_surface(r=10, h=20, res=100):
    """
        Generates an uncapped cylinder surface with minimal number of triangular cells.

        Args:
            r (float, optional): cylinder radius (default 10.0)
            h (float, optional): cylinder high (default 20.0)
            res (int, optional): resolution or number of facets (default 100)

        Returns:
            a cylinder surface (vtk.vtkPolyData)
        """
    cylinder = vtk.vtkCylinderSource()
    #remove caps
    cylinder.SetCapping(False)
    # the origin around which the cylinder should be centered
    cylinder.SetCenter(0, 0, 0)
    # the radius of the cylinder
    cylinder.SetRadius(r)
    # the height of the cylinder
    cylinder.SetHeight(h)
    # polygonal discretization; number of facets used to define the cylinder
    cylinder.SetResolution(res)
    
    #Now we have to triangulate the cylinder
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(cylinder.GetOutputPort())
    
    
    # The cylinder has discontinuities from the way the edges are
    # generated, we pass it through a CleanPolyDataFilter to merge any
    # points which are coincident or very close
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(tri.GetOutputPort())
    cleaner.SetTolerance(0.005)
    cleaner.Update()
    
    # Reverse normals so that they point inwards the surface
    cylinder_surface = reverse_sense_and_normals(cleaner.GetOutputPort())

    # Might contain non-triangle cells after cleaning -> remove them
    cylinder_surface = remove_non_triangle_cells(cylinder_surface)
    
    return cylinder_surface


def generate_UV_sphere_surface_with_hole(r=10.0, latitude_res=30,
                                   longitude_res=30, hole_size = 30, verbose=False):
    """
    Generates a triangulated sphere surface with a hole.
    
    Parameters:
        r (float, optional): sphere radius (default 10.0)
        latitude_res (int, optional): latitude resolution (default 30)
        longitude_res (int, optional): longitude resolution (default 30)
        hole_size (int, optional): starting longitude and latitude angle (default 30)
        verbose (boolean, optional): if True (default False), some extra
            information will be printed out
    
    Returns:
        a sphere surface with a hole (vtk.vtkPolyData)
    """

    if verbose:
        print("Generating a sphere with radius={}, latitude resolution={} "
                "and longitude resolution={}".format(r, latitude_res,
                                                       longitude_res))
    sphere = vtk.vtkSphereSource()

    # the origin around which the sphere should be centered
    sphere.SetCenter(0.0, 0.0, 0.0)

    sphere.SetThetaResolution(longitude_res)
    sphere.SetPhiResolution(latitude_res)

    #set the size of the hole
    sphere.SetEndPhi(180-hole_size)
    sphere.SetStartPhi(-(180-hole_size))

    #set radius
    sphere.SetRadius(r)


    # The sphere is made of strips, so pass it through a triangle filter
    # to get a triangle mesh
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(sphere.GetOutputPort())

    # The sphere has nasty discontinuities from the way the edges are
    # generated, so pass it though a CleanPolyDataFilter to merge any
    # points which are coincident or very close
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(tri.GetOutputPort())
    cleaner.SetTolerance(0.0005)
    cleaner.Update()


    # Reverse normals
    sphere_surface = reverse_sense_and_normals(cleaner.GetOutputPort())

    # Might contain non-triangle cells after cleaning - remove them
    sphere_surface = remove_non_triangle_cells(sphere_surface,
                                               verbose=verbose)
    return sphere_surface