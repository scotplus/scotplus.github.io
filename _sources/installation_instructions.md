# Getting Set Up

To clone our code and tutorials on your own device, travel to our GitHub repository:
https://github.com/Cbaker37/SCOOTR.git

Now that you have cloned the SCOOTR repository locally, you'll need to install some software in order to run our tutorials.
We recommend doing the below, as it will not affect your other environments, but feel free to configure it differently (as long
as all requirements are satisfied)

First, download anaconda (at least 2022.05) on your device. From here, enter these commands into your terminal:

<br></br>

<center>conda create -n scootr python=3.9</center>

<center>conda activate scootr</center>

<center>pip3 install -r requirements.txt</center>

<center>python -m ipykernel install --user --name=scootr</center>

<br></br>

Feel free to change the python version at your own risk (we programmed this repository in 3.9)

These steps create a conda environment specific to this project and walk you through the installation of all necessary packages. From here, open up any of our tutorials and select scootr as the kernel you will be using. This will ensure all necessary packages will be available when you run our tutorials. If you are doing this step in VSCode, the Select Kernel option will be in the top right, and the scootr kernel will be listed under "Python Environments."