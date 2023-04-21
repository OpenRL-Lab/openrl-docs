OpenRL Introduction
===============================

OpenRL Reinforcement Learning Framework
-------------------------------

OpenRL is a reinforcement learning research framework based on PyTorch developed by the Reinforcement Learning Team of 4Paradigm.
It provides a simple and easy-to-use interface that allows you to easily access different reinforcement learning environments.
Currently, OpenRL framework has the following features:

1. Simple and easy-to-use training interface, reducing the learning and usage costs of researchers.

2. Supports both **single-agent** and **multi-agent** algorithms.

3. Supports reinforcement learning training for **natural language tasks** (such as dialogue tasks).

4. Supports model import from `Hugging Face <https://huggingface.co/models>`_.

5. Supports models such as LSTM, GRU, Transformer, etc.

6. Supports various training accelerations, such as mixed precision training, data collecting with half-precision policy network, etc.

7. Support `gymnasium <https://gymnasium.farama.org/>`_ environments.

8. Support dictionary-type observation input.

9. Support popular machine learning training visualization platforms such as `wandb <https://wandb.ai/>`_ and `tensorboardX <https://tensorboardx.readthedocs.io/en/latest/index.html>`_ .

10. Supports serial and parallel training of environments while ensuring consistent performance under both scenarios.

11. Provides code coverage testing and unit testing.


In the following section on `Quick Start <../quick_start/index.html>`_ , we will introduce how to install the OpenRL framework,
and demonstrate how to use OpenRL through simple examples.


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