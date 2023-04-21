Installation Instructions
===============================

.. toctree::
   :maxdepth: 2

Installing OpenRL
--------------

OpenRL supports popular operating systems such as Ubuntu, MacOS, Windows, CentOS etc. Currently, OpenRL only supports Python version 3.8 and above.
Currently, OpenRL is available on `PyPI <https://pypi.org/project/openrl/>`_ and `Anaconda <https://anaconda.org/openrl/openrl>`_. Users can install it using pip or conda.

To install using pip:

.. code-block:: bash

    pip install openrl

To install using conda:

.. code-block:: bash

    conda install -c openrl openrl

To install from source:

.. code-block:: bash

    git clone https://github.com/OpenRL-Lab/openrl.git
    cd openrl
    pip install .

Check the Version
--------------

You can check the current installed version of OpenRL by executing the following command in your terminal:

.. code-block:: bash

    openrl --version


Next, we will use a `simple example <../quick_start/hello_world.html>`_ to show how to train your first agent.