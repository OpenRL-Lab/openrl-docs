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


Use Docker
--------------

OpenRL currently provides Docker images with and without GPU support.
If the user's computer does not have an NVIDIA GPU, they can obtain an image without the GPU plugin using the following command:

.. code-block:: bash

    sudo docker pull openrllab/openrl-cpu

If the user wants to accelerate training with a GPU, they can obtain it using the following command:

.. code-block:: bash

    sudo docker pull openrllab/openrl


After successfully pulling the image, users can run OpenRL's Docker image using the following commands:

.. code-block:: bash

    # Without GPU acceleration
    sudo docker run -it openrllab/openrl-cpu
    # With GPU acceleration
    sudo docker run -it --gpus all --net host openrllab/openrl


Once inside the Docker container, users can check OpenRL's version and then run test cases using these commands:

.. code-block:: bash

    # Check OpenRL version in Docker container
    openrl --version
    # Run test case
    openrl --mode train --env CartPole-v1

Next, we will use a `simple example <../quick_start/hello_world.html>`_ to show how to train your first agent.