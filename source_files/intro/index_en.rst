OpenRL Introduction
===============================

OpenRL Reinforcement Learning Framework
-------------------------------

OpenRL is a reinforcement learning research framework based on PyTorch developed by the Reinforcement Learning Team of 4Paradigm.
It provides a simple and easy-to-use interface that allows you to easily access different reinforcement learning environments.
Currently, OpenRL framework has the following features:

#. Simple and easy-to-use training interface, reducing the learning and usage costs of researchers.
#. Support both **single-agent** and **multi-agent** algorithms.
#. Support **offline RL** algorithms.
#. Support **Self-Play** training.
#. Support reinforcement learning training for **natural language tasks** (such as dialogue tasks).
#. Support `DeepSpeed <../quick_start/train_nlp.html#accelerate-training-with-deepspeed>`_ training。
#. Support **Arena** , which allows convenient evaluation of various agents in a competitive environment. Support local testing of submissions to the `JiDi platform <http://www.jidiai.cn/>`_.
#. Support model import from `Hugging Face <https://huggingface.co/models>`_. Support loading `Stable-baselines3 models from Hugging Face <https://openrl-docs.readthedocs.io/en/latest/sb3/index.html>`_ for testing and training.
#. Support models such as LSTM, GRU, Transformer, etc.
#. Support various training accelerations, such as mixed precision training, data collecting with half-precision policy network, etc.
#. Support `gymnasium <https://gymnasium.farama.org/>`_ environments.
#. Support dictionary-type observation input.
#. Support popular machine learning training visualization platforms such as `wandb <https://wandb.ai/>`_ and `tensorboardX <https://tensorboardx.readthedocs.io/en/latest/index.html>`_ .
#. Support serial and parallel training of environments while ensuring consistent performance under both scenarios.
#. Provides code coverage testing and unit testing.


In the following section on `Quick Start <../quick_start/index.html>`_ , we will introduce how to install the OpenRL framework,
and demonstrate how to use OpenRL through simple examples.

Users can also check the algorithms and environments supported by OpenRL, as well as obtain corresponding code in the `Gallery <https://github.com/OpenRL-Lab/openrl/blob/main/Gallery.md>`_.

Citing OpenRL
------------------------

If our work is helpful to you, please cite us:

.. code-block:: bibtex

    @misc{openrl2023,
        title={OpenRL},
        author={OpenRL Contributors},
        publisher = {GitHub},
        howpublished = {\url{https://github.com/OpenRL-Lab/openrl}},
        year={2023},
    }