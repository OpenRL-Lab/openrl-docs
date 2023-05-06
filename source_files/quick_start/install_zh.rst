安装说明
===============================

.. toctree::
   :maxdepth: 2

安装 OpenRL
--------------

OpenRL支持目前主流的操作系统如：Ubuntu, MacOS, Windows, CentOS等。OpenRL目前仅支持Python3.8及以上版本。
目前，OpenRL发布在了 `PyPI <https://pypi.org/project/openrl/>`_ 和 `Anaconda <https://anaconda.org/openrl/openrl>`_ 上，用户可以通过pip或者conda安装。

通过pip安装：

.. code-block:: bash

    pip install openrl

通过conda安装：

.. code-block:: bash

    conda install -c openrl openrl

从源码安装：

.. code-block:: bash

    git clone https://github.com/OpenRL-Lab/openrl.git
    cd openrl
    pip install .

版本查看
--------------

在命令行执行以下命令，可以查看当前安装的OpenRL版本：

.. code-block:: bash

    openrl --version

使用Docker
--------------

OpenRL目前也提供了包含显卡支持和非显卡支持的Docker镜像。
如果用户的电脑上没有英伟达显卡，则可以通过以下命令获取不包含显卡插件的镜像：

.. code-block:: bash

    sudo docker pull openrllab/openrl-cpu


如果用户想要通过显卡加速训练，则可以通过以下命令获取：

.. code-block:: bash

    sudo docker pull openrllab/openrl


镜像拉取成功后，用户可以通过以下命令运行OpenRL的Docker镜像：

.. code-block:: bash

    # 不带显卡加速
    sudo docker run -it openrllab/openrl-cpu
    # 带显卡加速
    sudo docker run -it --gpus all --net host openrllab/openrl


进入Docker镜像后，用户可以通过以下命令查看OpenRL的版本然后运行测例：

.. code-block:: bash

    # 查看Docker镜像中OpenRL的版本
    openrl --version
    # 运行测例
    openrl --mode train --env CartPole-v1


接下来，我们将会通过一个 `简单的例子 <../quick_start/hello_world.html>`_ 来介绍如何使用OpenRL框架。



