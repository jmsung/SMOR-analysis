# smor-analysis
Single More Off Rates (SMOR) analysis 


## Overview
This is an overview about this analysis tool. 


## Installation 
These steps will install smor-analysis and its dependencies in a conda environment with Python 3.7. It is recommended to use miniconda and conda environment to install the following, but experienced users can use any of their favorite method and directly go to step 6.   

1. Download and install Miniconda3 (Windows, Mac, Linux) by following the instructions in this link:    
<https://docs.conda.io/en/latest/miniconda.html>
    
2. Start a new shell / terminal. The conda base environment should be activated by default. If not, then you can activate it by running:   
`$ conda activate`

3. Create a new conda environment (smor) and install Python 3.7.  
`(base) $ conda create --name smor python=3.8`    

4. Activate the new environment (smor).  
`(base) $ conda activate smor`

5. Install git   
`(smor) $ conda install git`

6. Clone this repository.   
`(smor) $ git clone git@github.com:jmsung/smor-analysis.git`

7. Move to the folder.  
`(smor) $ cd smor-analysis`   

8. Install the setup.py that will install smor-analysis and its dependencies.  
`(smor) $ python setup.py develop`


## Instruction
1. Organize your data directory tree in the followng manner.     

	- data folder
		- data1
			- movie1.tif
			- info.txt
		- data2
			- movie2.tif
			- info.txt
		
2. Edit the parameters in the info.txt for each data in the same folder. Check the info.txt in data/example/ as an example. 

3. Change the working directory to where your data is located. All the data under the directory tree will be automatically analyzed.     
`(smor) $ cd /your_data_directory` 

4. Run the smor-analysis code.   
`(smor) $ run-smor-analysis`


## Notes:
* Notes and comments come here. 


## Reference:
Refer this paper.   
Hartooni et al. (2021). 
