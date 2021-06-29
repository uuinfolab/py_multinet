# Learning material

This folder contains:

* a jupyter notebook (with additional example files used in the notebook)
* a pdf generated from the notebook
* powerpoint slides

## Usage

This is a possible and typical way to use the notebook:

Create a virtual environment (using Python at least version 3.8):

    python -m venv multinet

To activate it, on linux/mac:
 
    source multinet/bin/activate

On Windows:

    multinet\Scripts\activate.bat

Then install the following:

    pip install uunet --upgrade
    pip install pandas
    pip install matplotlib
    pip install jupyter

To check that you have the latest version, you can execute:

    pip show uunet

And you should see: Version: 1.1.2. Then you can run:
  
    ipython kernel install --name "multinet" --user
    jupyter-notebook

from this directory.

Please notice that:

* you need Python version 3.8 or higher.
* the pip package is not compatible with conda.

All the material can be freely reused[1] and modified[2]. 

All inquiries can be sent to <matteo.magnani@it.uu.se>

[1] please acknowledge the original source
[2] please make sure that your changes are not attributed to us
