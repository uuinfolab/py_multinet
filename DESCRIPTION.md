This is the Python version of the _multinet_ library for the analysis of multilayer networks, first released for the R framework in January 2017 on CRAN. _multinet_ is developed by the Uppsala University Information Laboratory (UU InfoLab - https://uuinfolab.github.io)


- **Tutorial:** https://github.com/uuinfolab/py_multinet/tree/master/tutorial

- **Source:** https://github.com/uuinfolab/py_multinet
- **R version:** https://github.com/uuinfolab/r_multinet
- **C++ library:** https://github.com/uuinfolab/uunet
- **Bug reports:** https://github.com/uuinfolab/py_multinet/issues

A simple example
--------------

Find the degree of an actor on two layers:

    >>> import uunet.multinet as ml
    >>> m = ml.data("aucs")
    >>> ml.degree(m, actors = ['U54'], layers = ['leisure', 'facebook'])
    [18]
    
Install
-------

Install the latest version of uunet:

    $ python -m pip install uunet
    
Notes:

- Python >= 3.8 is required
- The package is not compatible with conda

Credits
-------

This library was originally based on the book: Multilayer Social Networks, by Dickison, Magnani & Rossi, Cambridge University Press (2016). The methods contained in the library and described in the book have been developed by many different authors: extensive references are available in the book, and in the documentation of each function we indicate the main reference we have followed for the implementation. For some methods developed after the book was published we give references to the corresponding literature.

The package uses functions from eclat <https://borgelt.net/eclat.html>, for association rule mining, Infomap <https://www.mapequation.org>, for the Infomap community detection method, and Howard Hinnant's date and time library <https://github.com/HowardHinnant/date>. The code from these libraries has been included in our source package, and may not be the latest version released by the authors.

License
-------

Released under the GNU General Public Licence.

    Contact: Matteo Magnani <matteo.magnani@it.uu.se>
