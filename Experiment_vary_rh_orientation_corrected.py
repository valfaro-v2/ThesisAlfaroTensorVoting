from normal_generation import *
from surface_generation import *
from error_calculation import *
from correct_global_orientation import *

from pycurv_testing.synthetic_surfaces import (
    PlaneGenerator, SphereGenerator, CylinderGenerator, SaddleGenerator,
    add_gaussian_noise_to_surface, remove_non_triangle_cells)
from pycurv import (
    TriangleGraph, PointGraph, normals_directions_and_curvature_estimation,
    nice_asin)
from pycurv import pycurv_io as io

import vtk
warnings = vtk.vtkOutputWindow()
warnings.GlobalWarningDisplayOff()

import numpy as np
import pandas as pd
import math

import os.path
from os import remove
import os

import matplotlib.pyplot as plt


#####################IMPORTANT#############################################

#Please replace the variable "fold" with the local path of the folder that contains the experiment files.

fold = "/hps/software/users/uhlmann/valfaro/MasterThesis/"

###########################################################################




def single_estimation(figure, r, res, noise, noise_random, rh_normals_seq, rh_curvature,
                      df_generate=False, vertex_based=False):
    """
    Function that for every radiushit value for the normals, sequentially:
        1. Generates a surface of the given characteristics
        2. Adds noise to the surface
        3. Builds a triangle graph associated to the noisy surface if vertex_based is set to False. Builds a point graph associated to the surface if vertex_based is set to True
        4. Assigns the true normals to the noisy surface taking a smooth surface of its same characteristics as reference
        5. Estimate normals (with their orientation corrected) and principal curvatures of the noisy surface
        6. Saves the .vtp and .gt files after the estimation
        7. Calculates the error measures
        8. If df_generate is set to True, saves the estimated, true and original values of the normals (resp. ppal curvatures) along with the corresponding errors into CSV files
        9. Saves and returns a dataframe containing the mean of the principal curvature errors and normal errors for every noise
           iteration
    
    Args:
        figure (str): "cylinder", "sphere", "sphere_hole" or "plane"
        r (int): if figure="sphere" or figure="sphere_hole" r is the radius of the sphere
                 if figure="cylinder" r is the radius of the base
                 if figure="plane" r is the half size of the x and y axes
                 of the plane
        res (int): if figure="sphere" or figure="sphere_hole" res is the number of longitude 
                   and latitude divisions of the sphere
                   if figure="cylinder" res is the number of subdivisions
                   in strips of the cylinder
                   if figure="plane" res is the number of subdivisions along
                   the x and y axes
        noise (int): noise percentage to be applied
        noise_random (bool): if True adds random noise in a random direction
        rh_normals_seq (float): radius hit values list for the estimation of normals
        rh_curvature (float): radius hit value for the estimation of curvatures
        df_generate (bool): if True, two CSVs file per each resolution level will be generated. One of them will contain principal curvatures info and errors. The other one will contain normals info and errors.
        vertex_based (bool): if True, AVV method will be conducted. If False, RVV method will be conducted.
    
    Returns:
        a dataframe (pandas.DataFrame) containing the mean of the principal curvature errors and normal error reduction metrics for every radiushit level
        
    """
    

    if figure=="sphere":
        sgen = SphereGenerator()

        surf_file=f"sphere"

        #Generate sphere
        surf = sgen.generate_UV_sphere_surface(r=r, latitude_res=res,
                                                     longitude_res=res) 

        #Add gaussian noise
        surf = add_gaussian_noise_to_surface(surf,
                                 percent=noise, rand_dir=noise_random)


        #Generate a reference smooth sphere to get the true normals at the 
        #noisy locations.

        sgen = SphereGenerator()
        smooth_surf = sgen.generate_UV_sphere_surface(r=r, longitude_res=300, 
                                                        latitude_res = 300)
        #Ground truth values of principal curvatures 
        true_kappa_1=1/r
        true_kappa_2=1/r

    if figure=="sphere_hole":


        surf_file=f"sphere_hole"

        #Generate sphere
        surf = generate_UV_sphere_surface_with_hole(r=r, latitude_res=res,
                                                     longitude_res=res, hole_size=30) 

        #Add gaussian noise
        surf = add_gaussian_noise_to_surface(surf,
                                 percent=noise, rand_dir=noise_random)


        #Generate a reference smooth sphere to get the true normals at the 
        #noisy locations.

        sgen = SphereGenerator()
        smooth_surf = sgen.generate_UV_sphere_surface(r=r, longitude_res=300, 
                                                        latitude_res = 300)
        #Ground truth values of principal curvatures 
        true_kappa_1=1/r
        true_kappa_2=1/r



    if figure=="plane":

        surf_file=f"plane"
        #Generate plane
        surf = modified_generate_plane_surface(x_half_size=r, y_half_size=r,
                                            x_res=res, y_res=res)
        #Add gaussian noise
        surf = add_gaussian_noise_to_surface(surf,
                         percent=noise, rand_dir=noise_random)

        #Ground truth values of principal curvatures 
        true_kappa_1=0
        true_kappa_2=0


    if figure == "cylinder":

        surf_file=f"cylinder"

        #Generate cylinder
        surf = generate_uncapped_cylinder_surface(r=r, h=2*r, res=res) 
        #voxel cylinder may also be an option 

        #Add gaussian noise
        surf = add_gaussian_noise_to_surface(surf,
                                 percent=noise, rand_dir=noise_random)

        #Generate a reference smooth cylinder to get the true normals at the 
        #noisy locations.

        smooth_surf = generate_uncapped_cylinder_surface(r=r, h=2*r, res=500)
        #Ground truth values of principal curvatures 
        true_kappa_1=0
        true_kappa_2=1/r



    #Next, we create a corresponding Triangle graph / Point graph depending on the method
    #to run and we associate the normals to each graph vertex.

    if vertex_based:
        #Build a point graph. Every vertex represents a triangle vertex
        #and every edge connects two adjacent vertices.
        tg = PointGraph()
        tg.build_graph_from_vtk_surface(surf, scale=(1,1,1),
                                   verbose=False)
    else:
        #Build a triangle graph. Every vertex represents a triangle center
        #and every edge connects two adjacent triangles
        tg= TriangleGraph()
        tg.build_graph_from_vtk_surface(surf, scale=(1,1,1), vtk_curv=False,
                                   verbose=False)



    #Create space for a new attribute "true_normal" at a vertex level.
    #We will assign to every vertex of the noisy surface the true normal
    #it should have.

    tg.graph.vp.true_normal = tg.graph.new_vertex_property("vector<float>")

    if figure=="cylinder" or figure=="sphere" or figure == "sphere_hole":

        pointLocator = vtk.vtkPointLocator()
        pointLocator.SetDataSet(smooth_surf)
        pointLocator.SetNumberOfPointsPerBucket(100)
        pointLocator.BuildLocator()


        #For every point on the noisy surface we will assign the normal
        #of its closest point in the smooth surface. That way we assign
        #the "true normals" to the noisy surface.


        xyz = tg.graph.vp.xyz

        for v in tg.graph.vertices():

            x,y,z = xyz[v]

            #Finds ID of the closest point on the smooth surface to 
            #the coordinates of v
            closest_point_id = pointLocator.FindClosestPoint([x, y, z])

            #Create a placeholder for the closest point
            closest_true_xyz = np.zeros(shape=3)

            #copies the x,y,z components of the point given by the ID
            #to the given placeholder array
            smooth_surf.GetPoint(closest_point_id, closest_true_xyz)

            #Calculate the surface normal at the closest point in the 
            #smooth surf
            if figure=="sphere" or figure == "sphere_hole":
                true_normal = sphere_normal(closest_true_xyz[0], closest_true_xyz[1], closest_true_xyz[2])
            if figure=="cylinder":
                true_normal = cylinder_normal(closest_true_xyz[0], closest_true_xyz[1], closest_true_xyz[2])

            #Assign to the vertex attribute "true_normal" of the noisy
            #surface the true normal it should have.
            tg.graph.vp.true_normal[v]=true_normal

    if figure=="plane":

        for v in tg.graph.vertices():

            #The true normal is given always by the same vector
            true_normal = np.array([0,0,1])

            #Assign to the vertex attribute "true_normal" the true normal
            tg.graph.vp.true_normal[v]=true_normal
    
    #Prepare additional parameters to run the method
    if vertex_based:
        area2=False #RVV does not weight curvature votes by triangle area
    else:
        area2=True #AVV weights curvature votes by triangle area.


    #Run normal and curvature estimation
    #for every radiushit level for the normals
    
    kappa_1_errors_list=[]
    kappa_2_errors_list=[]
    normal_errors_list=[]
    initial_normal_errors_list=[]
    angular_normal_errors_list=[]
    initial_angular_normal_errors_list=[]

    
    
    for rh_normals in rh_normals_seq:
        
        print(f"\n\nrh_normal {rh_normals}")
        
        method_tg_surf_dict = orientation_normals_directions_and_curvature_estimation(tg, radius_hit_normals=rh_normals, radius_hit_curvature=rh_curvature, methods=["VV"], area2=area2, cores=6)



        #Parameters to save the files
        if noise_random==True:
            noise_direction = "Random"
        else:
            noise_direction="Normal"


        if area2 is False:
            method = 'RVV'
        else:
            method = 'AVV'


        subfold = "Experiment_vary_rh_orientation_corrected/" + figure + "/ParaViewFiles/" 
        subsubfold = noise_direction +"Noise/"+ method + "/" + "res"+ str(res)+"/" + "noise"+str(noise) + "/r" + str(r)+"/"

        #Set folders to save the files
        if os.path.isdir(subfold+subsubfold) == False:
            os.makedirs(subfold+subsubfold)



        for method_name in list(method_tg_surf_dict.keys()):


            (tg, surf) = method_tg_surf_dict[method_name]



            #Save the files
            gt_file = '{}{}_rh_normal{}_rh_curvature{}.gt'.format(
                fold+subfold+subsubfold, surf_file, rh_normals, rh_curvature)
            tg.graph.save(gt_file)
            surface_file = '{}{}_rh_normal{}_rh_curvature{}.vtp'.format(
                fold+subfold+subsubfold, surf_file, rh_normals, rh_curvature)
            io.save_vtp(surf, surface_file)


    
        #Error calculation in principal curvatures and normals
        estimated_normals, initial_normals, true_normals, estimated_kappa_1, estimated_kappa_2, normal_errors, initial_normal_errors, angular_normal_errors, initial_angular_normal_errors, abs_kappa_1_errors, abs_kappa_2_errors = calculate_errors(tg=tg, true_kappa_1=true_kappa_1, true_kappa_2=true_kappa_2)

        #Average the errors for the rh levels
        kappa_1_errors_list.append(abs_kappa_1_errors.mean())
        kappa_2_errors_list.append(abs_kappa_2_errors.mean())


        normal_errors_list.append(normal_errors.mean())
        initial_normal_errors_list.append(initial_normal_errors.mean())

        angular_normal_errors_list.append(angular_normal_errors.mean())
        initial_angular_normal_errors_list.append(initial_angular_normal_errors.mean())

        if df_generate:
            
            #Generate dataframe with full curvature data of the current iteration
            df = pd.DataFrame()

            df["kappa_1"] = estimated_kappa_1
            df["kappa_2"] = estimated_kappa_2
            df["true_kappa_1"] = true_kappa_1
            df["true_kappa_2"] = true_kappa_2

            df['kappa1AbsErrors'] = abs_kappa_1_errors
            df['kappa2AbsErrors'] = abs_kappa_2_errors

            
            #Set folders to save the files
            subfold = "Experiment_vary_rh_orientation_corrected/"+ figure+"/CSVFiles/"
            subsubfold = noise_direction +"Noise/"+ method + "/" + "res"+ str(res)+"/" + "noise"+str(noise) + "/r" + str(r)+"/"

            if os.path.isdir(subfold+subsubfold) == False:
                os.makedirs(subfold+subsubfold)

            #Save curvature data
            base_file=f"CurvatureData"
            eval_file = fold + subfold+subsubfold + '{}_rh_normal{}_rh_curvature{}.csv'.format(
                        base_file, rh_normals, rh_curvature)
            df.to_csv(eval_file, sep=';')


            #Generate dataframe with full normals data of the current iteration
            df = pd.DataFrame()

            df["estimatedNormals"] = estimated_normals.tolist()
            df["initialNormals"] = initial_normals.tolist()

            df['trueNormals'] = true_normals.tolist()

            df["normal_errors"]= normal_errors
            df["initial_normal_errors"] = initial_normal_errors

            df["angular_errors"] = angular_normal_errors
            df["initial_angular_errors"] = initial_angular_normal_errors


            #Set folders to save the files
            subfold = "Experiment_vary_rh_orientation_corrected/"+figure+"/CSVFiles/" 
            if os.path.isdir(subfold+subsubfold) == False:
                os.makedirs(subfold+subsubfold)


            #Save normals data
            base_file=f"NormalsData"
            eval_file = fold + subfold+subsubfold + '{}_rh_normal{}_rh_curvature{}.csv'.format(
                        base_file, rh_normals, rh_curvature)


            df.to_csv(eval_file, sep=';')

            



    #Generate dataframe containing the mean of the principal curvature errors and normal error reduction metrics 
    #for every rh level  
    
    df = pd.DataFrame()

    df["rh_seq"]=rh_normals_seq
    df["kappa_1_errors"]=kappa_1_errors_list
    df["kappa_2_errors"]=kappa_2_errors_list
    df["normal_errors"]=normal_errors_list
    df["initial_normal_errors"]=initial_normal_errors_list
    df["angular_normal_errors"]=angular_normal_errors_list
    df["initial_angular_normal_errors"]=initial_angular_normal_errors_list
    df["normal_error_reduction"] = (df["initial_normal_errors"] - df["normal_errors"]) / df["initial_normal_errors"]
    df["angular_normal_error_reduction"] = (df["initial_angular_normal_errors"] - df["angular_normal_errors"]) / df["initial_angular_normal_errors"]


    #Set folders to save the files
    subfold = "Experiment_vary_rh_orientation_corrected/"+figure+"/CSVFiles/average_errors/rh_seq/"

    if os.path.isdir(subfold) == False:
        os.makedirs(subfold)

    #Save averaged data
    base_file=f"average_errors_r{r}"
    eval_file = fold+subfold + '{}_method{}_rh_curvature{}_noise{}_random{}_res{}.csv'.format(
                    base_file, method, rh_curvature, noise,noise_random, res)

    df.to_csv(eval_file, sep=';')
    return df
    




################### TUNE PARAMETERS HERE #############################

geometries = ["plane", "sphere"]
methods_list = ["AVV"]
rh_seq= [10,12,15, 18]
# res_seq=[10, 12, 15, 17, 19]
res_seq=[10, 12, 15]
noise_random=True
noise=20
rh_curvature=10
r=10

#Legend:
colors = ["blue", "green", "red", "purple", "orange", "magenta"]
linestyles = ["solid", "solid", "solid", "solid"]
labels=[f"rh = {rh}" for rh in rh_seq]



######################################################################

for geometry in geometries:
    
    for method_chosen in methods_list:
        
        
        if method_chosen=="RVV":
            per_vertex = True
        else:
            per_vertex = False
            
            
        #Calculate dfs
        
        angular_normal_data=[]
        kappa_1_data=[]
        kappa_2_data=[]
        
        for res in res_seq:
            
            path = f"Experiment_vary_rh_orientation_corrected/{geometry}/CSVFiles/average_errors/rh_seq/"
            filename=f"average_errors_r{r}_method{method_chosen}_rh_curvature{rh_curvature}_noise{noise}_random{noise_random}_res{res}.csv"

            if os.path.isfile(fold+path+filename):
                df = pd.read_csv(fold+path+filename, delimiter = ";")
                print(f"CSV files for res {res} and {method_chosen} already exist! No need to generate it again!")
            else:
                df = single_estimation(figure=geometry,
                                    r=10,
                                    res=res,
                                    noise=noise,
                                    noise_random=noise_random,
                                    rh_normals_seq=rh_seq,
                                    rh_curvature=rh_curvature,
                                    df_generate=True,
                                    vertex_based=per_vertex)
            
            angular_normal_data.append(df["angular_normal_error_reduction"].values)
            kappa_1_data.append(df["kappa_1_errors"].values)
            kappa_2_data.append(df["kappa_2_errors"].values)
        
        
        
        
        
        
        
        # Plot size:
        fig, axs = plt.subplots(3, 1, figsize=(22,19))
        fig.patch.set_facecolor('white')
        plt.subplots_adjust(top=0.9,hspace=0.4)


        for ax in axs:
            ax.tick_params(axis='both', labelsize=16)
        
        
        
        for i, rh in enumerate(rh_seq):
            
            normals_rh=[normals[i] for normals in angular_normal_data]
            kappa_1_rh=[kappa1[i] for kappa1 in kappa_1_data]
            kappa_2_rh=[kappa2[i] for kappa2 in kappa_2_data]
            
            axs[0].plot(
                res_seq,
                normals_rh,
                color=colors[i],
                linestyle=linestyles[i % len(linestyles)],
                label=labels[i],
                linewidth=2.0)
            
            # Plot the averaged kappa_1 errors data
            
            axs[1].plot(
                res_seq,
                kappa_1_rh,
                color=colors[i],
                linestyle=linestyles[i % len(linestyles)],
                label=labels[i],
                linewidth=2.0)
            
            # Plot the averaged kappa_2 errors data
            
            axs[2].plot(
                res_seq,
                kappa_2_rh,
                color=colors[i],
                linestyle=linestyles[i % len(linestyles)],
                label=labels[i],
                linewidth=2.0)
    

        # Add a title and axis labels for the normal error reduction plot
        axs[0].set_title("Angular Normal Error Reduction vs. Resolution (Averaged)",fontsize=20)
        axs[0].set_xlabel("Resolution",fontsize=16)
        axs[0].set_ylabel("Angular Normal Error Reduction",fontsize=16)

        

        # Add gridlines for the normal error reduction plot
        axs[0].grid()





        # Add a title and axis labels for the kappa_1 errors plot
        axs[1].set_title("kappa1 errors vs. Resolution (Averaged)",fontsize=20)
        axs[1].set_xlabel("Resolution", fontsize=16)
        axs[1].set_ylabel("kappa1 errors",fontsize=16)

        

        # Add gridlines for the kappa_1 errors plot
        axs[1].grid()



        # Add a title and axis labels for the kappa_1 errors plot
        axs[2].set_title("kappa2 errors vs. Resolution (Averaged)",fontsize=20)
        axs[2].set_xlabel("Resolution", fontsize=16)
        axs[2].set_ylabel("kappa2 errors",fontsize=16)

        

        # Add gridlines for the kappa_1 errors plot
        axs[2].grid()


        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=6, handlelength=2.5, handleheight=0.7, fontsize=18)
        
        subfold = "Experiment_vary_rh_orientation_corrected_figs/"

        if os.path.isdir(subfold) == False:
            os.makedirs(subfold)

        fig.savefig(f"Experiment_vary_rh_orientation_corrected_figs/{geometry}_resolution_average_{method_chosen}_noise{noise}_random{noise_random}_rh_variation.png", dpi=300)





        

        
