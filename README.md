# Installation Instructions
The next instructions have been tested and are compatible with Ubuntu 18.04. 

### Prerequisites
Before proceeding with the installation steps, please ensure that you have the following prerequisites installed on your system:
- Anaconda3
- Git

If your system already meets the requirements, follow the next steps in the terminal to proceed with the installation of PyCurv:

## 1. Create Environment
Create a new environment named "tensor-voting" with Python version 3.7 using the following command:
```
conda create -n tensor-voting python=3.7
```
Activate the newly created environment:
```
conda activate tensor-voting
```
## 2. Install Packages
Install the required packages using the following commands:
```
conda install scipy
conda install pandas
conda install matplotlib
conda install -c conda-forge pythran
conda install -c anaconda scikit-image
conda install -c "conda-forge/label/cf202003" graph-tool
```
## 3. Install Pyto
Clone the Pyto repository from GitHub and navigate to the Pyto folder:
```
git clone https://github.com/vladanl/Pyto.git
cd Pyto
```
**Important**: Manually edit the setup.py file before proceeding further. Change the
scipy version for 1.5 instead of 1.4.1 in line 26.
Then, install Pyto using the following command:
```
pip install .
```
## 4. Install PyCurv
Navigate back to the previous directory:
```
cd ..
```
Clone the PyCurv repository from GitHub and navigate to the folder containing PyCurv:
```
git clone https://github.com/kalemaria/pycurv.git
cd pycurv
```
Install PyCurv using the following command:
```
python setup.py install
```
## 5. Install Nibabel
Navigate back to the previous directory:
```
cd ..
```
Finally, install the specific version of Nibabel (3.0.0) using the following command:
```
pip install nibabel==3.0.0
```

If something goes wrong, please refer to the official instructions provided in https://github.com/kalemaria/pycurv/blob/master/README.md to install it.

# Project Files and Configuration

This project contains several experiment .py files. Each of these files is responsible for executing a specific experiment and generating results. The parameters of each experiment can be fine tuned after the single_estimation function of each file in a dedicated section to do so. 

## Experiments

**Experiment1**: This experiment evaluates the performance of estimating normals and curvatures for the selected geometrical figures, methods and different levels of noise when increasing resolution. The type of noise can be chosen to be random or added in normal directions to the triangular faces.

**Experiment1_orientation_corrected**: similar to Experiment1, but this experiment refines the orientation of estimated normals before curvature estimation. The intention is to improve the results obtained.

**Experiment2**: This experiment evaluates the performance of estimating normals and curvatures for the selected geometrical figures, methods and different resolution levels when increasing noise. The type of noise is also chosen. The type of noise can be chosen to be random or added in normal directions to the triangular faces.

**Experiment2_orientation_corrected**: similar to Experiment2, but this experiment refines the orientation of estimated normals before curvature estimation. The intention is to improve the results obtained.

**Experiment_vary_rh_orientation_corrected**: for the selected geometrical figures, methods, noise level, radiushit value for estimating curvatures and different radiushit values for estimating normals, the performance in estimating normals and curvatures is evaluated when increasing the resolution. Additionally, the applied method undergoes a refinement of the orientation of the estimated normals prior to curvature estimation.

## Running Experiments
Before running any experiment, it is essential to adjust the fold variable in each experiment file. Set the path to the folder containing the experiment files.
Also, make sure you have the necessary dependencies installed and the "tensor-voting" environment activated.
To run each of the experiments, follow these steps in the terminal:
- Navigate to your local folder containing the experiment files using `cd <path to folder>`, replacing `<path to folder>` with the actual path.
- Run the command `python <name of the experiment file>`, replacing `<name of the experiment file>` with the actual filename.

If you intend to run any experiment again with different parameters, please delete the corresponding experiment output folders (see below: Experiment Output) beforehand.

## Experiment Output


Upon executing an experiment file, two main folders will be created:

1. **Experiment Folder**: This folder is named after the specific experiment and contains all the generated files, such as .csv, .gt, and .vtp files. These files store the data and outputs produced during the experiments. The structure of each experiment folder is as follows:

    - Each geometric figure has its own subfolder.
    - Inside each geometric figure folder, there are two subfolders: CSVFiles and ParaViewFiles.
    - The CSVFiles folder contains all the generated .csv files containing data from the experiments.
    - The ParaViewFiles folder contains all the .gt and .vtp files generated in the experiments.
    - Both CSVFiles and ParaViewFiles folders follow a similar structure:
        - They have subfolders for the type of noise added in the experiment, NormalNoise or RandomNoise.
        - Inside these subfolders, there are folders for each method used, AVV or RVV.
        - Inside the method folders, there are folders for specific resolutions, labeled as res10, res20, etc.
        - Within each resolution folder, there are folders for noise levels, such as noise5, noise10, etc.
        - Finally, there are folders for each radius of the figure used, labeled as r10, r20, etc., where the corresponding files for that specific whole setting can be found.
    - Additionally, the CSVFiles folder always contains an average_errors folder, which stores the average normal and curvature errors for each level of the corresponding iterating sequence.
    
2. **Experiment Figures Folder**: This folder is named by appending "_figs" to the experiment's name. It contains graphs and visualizations showcasing the results obtained from the experiment.
   



