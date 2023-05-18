from pycurv.vector_voting import *
import graph_tool.all as gt
from pycurv.graphs import *

def orientation_normals_directions_and_curvature_estimation(
        sg, radius_hit_normals, radius_hit_curvature, epsilon=0, eta=0, methods=['VV'],
        page_curvature_formula=False, full_dist_map=False, graph_file='temp.gt',
        area2=True, only_normals=False, poly_surf=None, cores=10, runtimes=''):
    """
    Runs the modified Normal Vector Voting algorithm (with different options for
    the second pass) to estimate surface orientation, principle curvatures and
    directions for a surface using its triangle graph.

    Args:
        sg (TriangleGraph or PointGraph): triangle or point graph generated
            from a surface of interest
        radius_hit_normals (float): radius in length unit of the graph;
            Applied to estimate normals.
        radius_hit_curvature (float): radius in length unit of the graph;
            it should be chosen to correspond to radius of smallest features of
            interest on the surface. Applied to estimate curvatures.
        epsilon (float, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease junction"
            (class 2), default 0
        eta (float, optional): parameter of Normal Vector Voting algorithm
            influencing the number of triangles classified as "crease junction"
            (class 2) and "no preferred orientation" (class 3, see Notes),
            default 0
        methods (list, optional): all methods to run in the second pass ('VV'
            and 'SSVV' are possible, default is 'VV')
        page_curvature_formula (boolean, optional): if True (default False),
            normal curvature formula from Page et al. is used in VV (see
            collect_curvature_votes)
        full_dist_map (boolean, optional): if True, a full distance map is
            calculated for the whole graph (not possible for PointGraph),
            otherwise a local distance map is calculated later for each vertex
            (default)
        graph_file (string, optional): name for a temporary graph file
            after the first run of the algorithm (default 'temp.gt')
        area2 (boolean, optional): if True (default), votes are weighted by
            triangle area also in the second step (principle directions and
            curvatures estimation; not possible for PointGraph)
        only_normals (boolean, optional): if True (default False), only normals
            are estimated, without principal directions and curvatures, only the
            graph with the orientations, normals or tangents is returned.
        poly_surf (vtkPolyData, optional): surface from which the graph was
            generated, scaled to given units (required only for SSVV, default
            None)
        cores (int, optional): number of cores to run VV in parallel (default 10)
        runtimes (str, optional): if given, runtimes and some parameters are
            added to this file (default '')

    Returns:
        a dictionary mapping the method name ('VV' and 'SSVV') to the
        tuple of two elements: TriangleGraph or PointGraph (if pg was given)
        graph and vtkPolyData surface of triangles with classified orientation
        and estimated normals or tangents, principle curvatures and directions
        (if only_normals is False)

    Notes:
        * Maximal geodesic neighborhood distance g_max for normal vector voting
          will be derived from radius_hit: g_max = pi * radius_hit / 2
        * If epsilon = 0 and eta = 0 (default), all triangles will be classified
          as "surface patch" (class 1).
    """
    t_begin = time.time()

    normals_estimation(sg, radius_hit_normals, epsilon, eta, full_dist_map, cores=cores,
                       runtimes=runtimes, graph_file=graph_file)
    
    #################################################################################
    
    #Correct normal orientation globally before estimating curvatures
    #Create the average normal vector atribute
    sg.graph.vp.avg_normals = sg.graph.new_vertex_property('vector<float>')
    
    # * Maximal geodesic neighborhood distance g_max for normal vector voting *
    # g_max is 1/4 of circle circumference with radius=radius_hit
    g_max = math.pi * radius_hit_normals / 2
    
    #to access the refined normal after voting
    n_v = sg.graph.vp.n_v
    
    for v in sg.graph.vertices():
        
        
        # Find the neighboring vertices of vertex v to be returned:
        if sg.__class__.__name__ == "TriangleGraph":
            neighbor_idx_to_dist = sg.find_geodesic_neighbors(
                v, g_max)
        else:  # PointGraph
            neighbor_idx_to_dist = sg.find_geodesic_neighbors_exact(
                v, g_max)
            
            
        pos = list(neighbor_idx_to_dist.keys())
        
        #Get neighboring vertices of v
        adj_vertices = set(u for u in v.all_neighbors() if u in pos)
        #Get neighboring normals of v
        adj_normals = [n_v[u] for u in adj_vertices]
        #Calculate the average of the neighboring normals
        avg_normal = np.mean(adj_normals, axis=0)
        #Assign it as an attribute to v, we also normalize the averaged normal
        sg.graph.vp.avg_normals[v] = avg_normal / np.linalg.norm(avg_normal)
        
        #Determine if the orientation of the normal at v should be reversed
        dot_prod = np.dot(n_v[v], sg.graph.vp.avg_normals[v])
        if dot_prod < 0:
            n_v[v] = np.array([-x for x in n_v[v]])
            
    ############################################
    
    if sg.__class__.__name__ == "PointGraph":
        vertex_based = True
        area2 = False
        full_dist_map = False
    else:
        vertex_based = False

    if only_normals is False:
        if len(methods) > 1:
            # load the graph from file, so each method starts a new
            sg = None
        results = {}
        for method in methods:
            sg_curv, surface_curv = curvature_estimation(
                radius_hit_curvature, graph_file=graph_file, method=method,
                page_curvature_formula=page_curvature_formula, area2=area2,
                poly_surf=poly_surf, full_dist_map=full_dist_map, cores=cores,
                runtimes=runtimes, vertex_based=vertex_based, sg=sg)
            results[method] = (sg_curv, surface_curv)
        if graph_file == 'temp.gt' and isfile(graph_file):
            remove(graph_file)

        t_end = time.time()
        duration = t_end - t_begin
        minutes, seconds = divmod(duration, 60)
        print('Whole method took: {} min {} s'.format(minutes, seconds))
        return results