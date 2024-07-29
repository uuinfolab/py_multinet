# multinet library (python version)

This repository contains the python version of the _multinet_ library for the analysis of multilayer networks, first released for the R framework in January 2017 on CRAN.

The library is available on PyPI (the Python Package Index), and can be installed using _pip_. This repository is mainly useful if you want to develop the library.

This library was originally based on the book: Multilayer Social Networks, by Dickison, Magnani & Rossi, Cambridge University Press (2016). The methods contained in the library and described in the book have been developed by many different authors: extensive references are available in the book, and in the documentation of each function we indicate the main reference we have followed for the implementation. For some methods developed after the book was published we give references to the corresponding literature.

## Requirements

* python >= 3.8

## Installation

    pip install uunet

(this provides the module: uunet.multinet)

## Usage

In the directory `tutorial/` you find example code covering most of the functions in the library. Documentation for all functions can be obtained from Python using `help()`.


## Contribute

To modify the library, one should consider that a large part of its code is written in C++ and comes from the [uunet repository](https://github.com/uuinfolab/uunet).

If you only want to modify the functions written in python, this can be done directly in this repository. These functions are in uunet/multinet.py.

After modifying the functions, you can install the new version running pip from the directory where setup-py is located (this requires the setuptools module and the environment needed to recompile uunet):

```sh
python -m pip install .
```

The directory C++/ contains the files exporting C++ functions from uunet to python, using ext/pybind11. These functions can also be updated directly in this repository:

- main.cpp contains the definitions of the python functions implemented in C++.
- py_functions.cpp contains the functions referenced in main.cpp, themselves calling functions from uunet.
- pycpp_utils.cpp contains some utility functions automating some common tasks used in py_functions.cpp.

If you need to modify any of the files in the directories eclat/, infomap/ and src/, they are imported from uunet and should modified the [uunet repository](https://github.com/uuinfolab/uunet). One can then get the latest code from uunet by running:

```sh
git submodule update --remote --merge
```

This command loads the latest code from uunet into ext/.

The repository has two workflows, publish-test and publish, generating source distribution and wheels and uploading them respectively to TestPyPI and PyPI. For PyPI we use .devN as a suffix for the library version, that is used under development with publish-test. When a new stable release is ready, publish is run on the master branch without .devN. 

## Contact

For any inquiries regarding this repository you can contact <matteo.magnani@it.uu.se>.
