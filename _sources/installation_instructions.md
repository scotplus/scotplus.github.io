# Getting Set Up

To clone our tutorials on your own device, travel to our GitHub repository:
https://github.com/scotplus/book_source.git

Now that you have cloned the SCOT+ repository locally, you'll need to install some software in order to run our tutorials.
We recommend doing the below, as it will not affect your other environments, but feel free to configure it differently (as long
as all requirements are satisfied)

First, install anaconda on your device. From here, enter these commands into your terminal:

<br></br>

<center>conda create -n scotplus python=3.9</center>

<center>conda activate scotplus</center>

<center>pip install scotplus</center>

<center>python -m ipykernel install --user --name=scotplus</center>

<br></br>

Feel free to change the python version at your own risk (we programmed this repository in 3.9)

These steps create a conda environment specific to this project and walk you through the installation of all necessary packages. From here, open up any of our tutorials and select scotplus as the kernel you will be using. This will ensure all necessary packages will be available when you run our tutorials. If you are doing this step in VSCode, the Select Kernel option will be in the top right, and the scotplus kernel will be listed under "Python Environments."