========
Overview
========

An example package. Generated with cookiecutter-pylibrary.

* Free software: MIT license

Installation
============

::

    pip install bayesian-personalized-ranking

You can also install the in-development version with::

    pip install git+ssh://git@yes/jems-lee/bayesian-personalized-ranking.git@main

Documentation
=============


https://bayesian-personalized-ranking.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
