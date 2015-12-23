# Installation

This code has been tested with Python 3.5.0

First create a virtual environment 

    $: pyenv versions
      system
      2.7.11rc1
      3.5.0
    $: pyenv virtualenv 3.5.0 linear-regression
    $: pip install -r requirements.txt

You can now load the LinearRegression classes in a console:

    >>> import imp  
    >>> linear_regression = imp.load_source('linear_regression', '../lib/linear_regression.py')
    >>> from linear_regression import *
    >>> x = np.array([[-0.99768], [-0.69574], [-0.40373], [-0.10236], [0.22024], [0.47742], [0.82229]])
    >>> y = np.array([2.0885, 1.1646, 0.3287, 0.46013, 0.44808, 0.10013, -0.32952])
    >>> reg = BatchGradient()
    >>> reg.fit(x, y)

Check the experiments folder to see the code in action with a Jupyter notebook.

# Running the tests

    py.test

