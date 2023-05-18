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




def single_estimation(figure, r, res_seq, noise, noise_random, rh,
                 df_generate=False, vertex_based=False):
    """
    Function that for every resolution level sequentially:
        1. Generates a surface of the given characteristics
        2. Adds noise to the surface
        3. Builds a triangle graph associated to the noisy surface if vertex_based is set to False. Builds a point graph associated to the surface if vertex_based is set to True
        4. Assigns the true normals to the noisy surface taking a smooth surface of its same characteristics as reference
        5. Estimate normals (with their orientation corrected) and principal curvatures of the noisy surface
        6. Saves the .vtp and .gt files after the estimation
        7. Calculates the error measures
        8. If df_generate is set to True, saves the estimated, true and original values of the normals (resp. ppal curvatures) along with the corresponding errors into CSV files
        9. Saves and returns a dataframe containing the mean of the principal curvature errors and normal errors for every resolution
           iteration
    
    Args:
        figure (str): "cylinder", "sphere", "sphere_hole" or "plane"
        r (int): if figure="sphere" or figure="sphere_hole" r is the radius of the sphere
                 if figure="cylinder" r is the radius of the base
                 if figure="plane" r is the half size of the x and y axes
                 of the plane
        res_seq (list): list of approximation resolutions to be applied.
                        if figure="sphere" or "sphere_hole" res is the number of longitude and latitude divisions of the sphere
                        if figure="cylinder" res is the number of subdivisions in strips of the cylinder
                        if figure="plane" res is the number of subdivisions along the x and y axes
        noise (int): noise percentage to be applied
        noise_random (bool): if True adds random noise in a random direction
        rh (int): radius hit value
        df_generate (bool): if True, two CSVs file per each resolution level will be generated. One of them will contain principal curvatures info and errors. The other one will contain normals info and errors.
        vertex_based (bool): if True, AVV method will be conducted. If False, RVV method will be conducted.
    
    Returns:
        a dataframe (pandas.DataFrame) containing the mean of the principal curvature errors and normal error reduction metrics for every resolution level
        
    """
    
    kappa_1_errors_list=[]
    kappa_2_errors_list=[]
    normal_errors_list=[]
    initial_normal_errors_list=[]
    angular_normal_errors_list=[]
    initial_angular_normal_errors_list=[]
    
    counter=0
    for res in res_seq:
        
        print(f"\n\n\n Resolution {res} - Step {counter}/{len(res_seq)-1}\n\n\n Noise {noise}\n\n\n")
        counter+=1
        
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
        #We will assign to every vertex of the noisy sphere the true normal
        #it should have.

        tg.graph.vp.true_normal = tg.graph.new_vertex_property("vector<float>")
        
        if figure=="cylinder" or figure=="sphere" or figure == "sphere_hole":

            pointLocator = vtk.vtkPointLocator()
            pointLocator.SetDataSet(smooth_surf)
            pointLocator.SetNumberOfPointsPerBucket(100)
            pointLocator.BuildLocator()


            #For every vertex on the graph of the noisy surface we will assign the normal
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

                #Calculate the sphere surface at the closest point in the 
                #smooth surface
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
                
        if vertex_based:
            area2=False #RVV does not weight curvature votes by triangle area
        else:
            area2=True #AVV weights curvature votes by triangle area.
        
        
        
        
        #Run the normal and curvature estimation
        method_tg_surf_dict = orientation_normals_directions_and_curvature_estimation(
            tg, radius_hit_normals=rh, radius_hit_curvature = rh, methods=["VV"], area2=area2, cores=6)
        
        #Parameters to save the files 
        if noise_random==True:
            noise_direction = "Random"
        else:
            noise_direction="Normal"
            
       
        if area2 is False:
            method = 'RVV'
        else:
            method = 'AVV'
            
                    

        
        subfold = "Experiment1_orientation_corrected/" + figure + "/ParaViewFiles/" 
        subsubfold = noise_direction +"Noise/"+ method + "/" + "res"+ str(res)+"/" + "noise"+str(noise) + "/r" + str(r)+"/"
        
        #Set folders to save the files
        if os.path.isdir(subfold+subsubfold) == False:
            os.makedirs(subfold+subsubfold)
                
            

        for method_name in list(method_tg_surf_dict.keys()):
            
            
            (tg, surf) = method_tg_surf_dict[method_name]
            
            gt_file = '{}{}_rh{}.gt'.format(
                fold+subfold+subsubfold, surf_file, rh)
            tg.graph.save(gt_file)
            surf_file = '{}{}_rh{}.vtp'.format(
                fold+subfold+subsubfold, surf_file, rh)
            io.save_vtp(surf, surf_file)
            
            
        
        #Error calculation in principal curvatures and normals
        estimated_normals, initial_normals, true_normals, estimated_kappa_1, estimated_kappa_2, normal_errors, initial_normal_errors, angular_normal_errors, initial_angular_normal_errors, abs_kappa_1_errors, abs_kappa_2_errors = calculate_errors(tg=tg, true_kappa_1=true_kappa_1, true_kappa_2=true_kappa_2)
        
        #Average the errors for the res level
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
            subfold = "Experiment1_orientation_corrected/" +figure+"/CSVFiles/"
            subsubfold = noise_direction +"Noise/"+ method + "/" + "res"+ str(res)+"/" + "noise"+str(noise) + "/r" + str(r)+"/"
            
            if os.path.isdir(subfold+subsubfold) == False:
                os.makedirs(subfold+subsubfold)

            #Save curvature data
            base_file=f"CurvatureData"
            eval_file = fold + subfold+subsubfold + '{}_rh{}.csv'.format(
                        base_file, rh)


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
            subfold = "Experiment1_orientation_corrected/" +figure+"/CSVFiles/"
            if os.path.isdir(subfold+subsubfold) == False:
                os.makedirs(subfold+subsubfold)

            #Save normals data
            base_file=f"NormalsData"
            eval_file = fold + subfold+ subsubfold + '{}_rh{}.csv'.format(
                        base_file, rh)


            df.to_csv(eval_file, sep=';')
            
            
            
    #Generate dataframe containing the mean of the principal curvature errors and normal error reduction metrics 
    #for every resolution level
    
    df = pd.DataFrame()
    
    df["res"]=res_seq
    df["kappa_1_errors"]=kappa_1_errors_list
    df["kappa_2_errors"]=kappa_2_errors_list
    df["normal_errors"]=normal_errors_list
    df["initial_normal_errors"]=initial_normal_errors_list
    df["angular_normal_errors"]=angular_normal_errors_list
    df["initial_angular_normal_errors"]=initial_angular_normal_errors_list
    df["normal_error_reduction"] = (df["initial_normal_errors"] - df["normal_errors"]) / df["initial_normal_errors"]
    df["angular_normal_error_reduction"] = (df["initial_angular_normal_errors"] - df["angular_normal_errors"]) / df["initial_angular_normal_errors"]
    
    
    #Set folders to save the files
    subfold = "Experiment1_orientation_corrected/" + figure+"/CSVFiles/average_errors/res_seq/"
    
    if os.path.isdir(subfold) == False:
        os.makedirs(subfold)
        
    #Save averaged data    
    base_file=f"average_errors_r{r}"
    eval_file = fold+subfold + '{}_method{}_rh{}_noise{}_random{}.csv'.format(
                    base_file, method, rh, noise, noise_random)
    
    df.to_csv(eval_file, sep=';')
    
    return df




################### TUNE PARAMETERS HERE #############################

geometries = ["plane", "cylinder", "sphere", "sphere_hole"]
# res_seq = list(range(10, 27, 2))
res_seq = list(range(10, 15, 2))
methods_list = ["AVV"]
noise_random_direction = [True, False]
noise_seq=[5,10,20,25]
r=10
rh=10

#Legend:
colors = ["blue", "green", "red", "purple", "orange", "magenta"]
linestyles = ["solid", "solid", "solid", "solid","dashed", "dotted", "dashdot"]
labels=[f"noise={noise}" for noise in noise_seq]

######################################################################

for geometry in geometries:
    for method_chosen in methods_list:
        for noise_random in noise_random_direction:

            if method_chosen=="RVV":
                per_vertex = True
            else:
                per_vertex = False



            # Plot size:
            fig, axs = plt.subplots(6, 1, figsize=(20,25))
            fig.patch.set_facecolor('white')
            plt.subplots_adjust(top=0.9,hspace=0.4)
            
            for ax in axs:
                ax.tick_params(axis='both', labelsize=16)






            print(f"\n\n\n OBJECT: {geometry}\n\n\n")

            normal_data = []
            angular_normal_data=[]
            kappa_1_data = []
            kappa_2_data = []
            normal_errors_data=[]
            angular_normal_errors_data=[]
            # Loop over noise levels
            for i, noise in enumerate(noise_seq):

                path = f"Experiment1_orientation_corrected/{geometry}/CSVFiles/average_errors/res_seq/"
                filename=f"average_errors_r{r}_method{method_chosen}_rh{rh}_noise{noise}_random{noise_random}.csv"

                if os.path.isfile(fold+path+filename):
                    df = pd.read_csv(fold+path+filename, delimiter = ";")
                    print(f"CSV files for noise {noise} and {method_chosen} already exist! No need to generate it again!")
                else:
                    df = single_estimation(
                        figure=geometry,
                        r=r,
                        res_seq=res_seq,
                        noise=noise,
                        noise_random=noise_random,
                        rh=rh,
                        df_generate=True,
                        vertex_based=per_vertex)






                normal_data.append(df["normal_error_reduction"].values)
                angular_normal_data.append(df["angular_normal_error_reduction"].values)
                kappa_1_data.append(df["kappa_1_errors"].values)
                kappa_2_data.append(df["kappa_2_errors"].values)
                normal_errors_data.append(df["normal_errors"].values)
                angular_normal_errors_data.append(df["angular_normal_errors"].values)



            # Plot
            for i, noise in enumerate(noise_seq):
                axs[0].plot(
                    res_seq,
                    normal_data[i],
                    color=colors[i],
                    linestyle=linestyles[i % len(linestyles)],
                    label=labels[i],
                    linewidth=2.0
                )
                
                axs[1].plot(
                    res_seq,
                    angular_normal_data[i],
                    color=colors[i],
                    linestyle=linestyles[i % len(linestyles)],
                    label=labels[i],
                    linewidth=2.0
                )

                axs[2].plot(
                    res_seq,
                    normal_errors_data[i],
                    color=colors[i],
                    linestyle=linestyles[i % len(linestyles)],
                    label=labels[i],
                    linewidth=2.0
                )

                axs[3].plot(
                    res_seq,
                    angular_normal_errors_data[i],
                    color=colors[i],
                    linestyle=linestyles[i % len(linestyles)],
                    label=labels[i],
                    linewidth=2.0
                )
                
                axs[4].plot(
                    res_seq,
                    kappa_1_data[i],
                    color=colors[i],
                    linestyle=linestyles[i % len(linestyles)],
                    label=labels[i],
                    linewidth=2.0
                )

                axs[5].plot(
                    res_seq,
                    kappa_2_data[i],
                    color=colors[i],
                    linestyle=linestyles[i % len(linestyles)],
                    label=labels[i],
                    linewidth=2.0
                )

            # Add a title and axis labels for the normal error reduction plot
            axs[0].set_title("Normal Error Reduction vs. Resolution (Averaged)",fontsize=20)
            axs[0].set_xlabel("Resolution",fontsize=16)
            axs[0].set_ylabel("Normal Error Reduction",fontsize=16)

            # Add gridlines for the normal error reduction plot
            axs[0].grid()
            
            # Add a title and axis labels for the normal error reduction plot
            axs[1].set_title("Angular Normal Error Reduction vs. Resolution (Averaged)",fontsize=20)
            axs[1].set_xlabel("Resolution",fontsize=16)
            axs[1].set_ylabel("Angular Normal Error Reduction",fontsize=16)

            # Add gridlines for the normal error reduction plot
            axs[1].grid()

            # Add a title and axis labels for the kappa_1 errors plot
            axs[2].set_title("Normal errors vs. Resolution (Averaged)",fontsize=20)
            axs[2].set_xlabel("Resolution", fontsize=16)
            axs[2].set_ylabel("Normal errors",fontsize=16)

            # Add gridlines for the kappa_1 errors plot
            axs[2].grid()

            # Add a title and axis labels for the kappa_1 errors plot
            axs[3].set_title("Angular normal errors vs. Resolution (Averaged)", fontsize=20)
            axs[3].set_xlabel("Resolution",fontsize=16)
            axs[3].set_ylabel("Angular normal errors",fontsize=16)

            # Add gridlines for the kappa_1 errors plot
            axs[3].grid()
            
            # Add a title and axis labels for the kappa_1 errors plot
            axs[4].set_title("kappa1 errors vs. Resolution (Averaged)",fontsize=20)
            axs[4].set_xlabel("Resolution", fontsize=16)
            axs[4].set_ylabel("kappa1 errors",fontsize=16)

            # Add gridlines for the kappa_1 errors plot
            axs[4].grid()

            # Add a title and axis labels for the kappa_1 errors plot
            axs[5].set_title("kappa2 errors vs. Resolution (Averaged)", fontsize=20)
            axs[5].set_xlabel("Resolution",fontsize=16)
            axs[5].set_ylabel("kappa2 errors",fontsize=16)

            # Add gridlines for the kappa_1 errors plot
            axs[5].grid()

            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=6, handlelength=2.5, handleheight=0.7, fontsize=18)
            fig.subplots_adjust(top=0.95)
            fig.subplots_adjust(bottom=0.05)

            subfold = "Experiment1_orientation_corrected_figs/"

            if os.path.isdir(subfold) == False:
                os.makedirs(subfold)

            fig.savefig(f"Experiment1_orientation_corrected_figs/{geometry}_resolution_average_{method_chosen}_noiserandom{noise_random}.png", dpi=300)
