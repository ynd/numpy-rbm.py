rbm.py
======

A pythonic library for Restricted Boltzmann Machines (RBMs). RBMs are a state
of the art method for modeling data. This library is both simple to use and
efficient. The only dependency is numpy.

Example
-------

    >>> import numpy, rbm
    >>> X = numpy.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> model = rbm.RBM(n_hiddens=2)
    >>> model.fit(X)

