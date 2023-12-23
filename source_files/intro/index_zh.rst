OpenRL 介绍
===============================

OpenRL 强化学习框架
-------------------------------

OpenRL 是由第四范式强化学习团队开发的基于PyTorch的强化学习研究框架，它提供了一个简单易用的接口，可以让你方便地接入不同强化学习环境。
目前，OpenRL框架有以下特点：

#. 简单易用的训练接口，降低研究人员的学习和使用成本。
#. 同时支持 **单智能体** 和 **多智能体** 算法。
#. 支持 **离线强化学习** （Offline RL）算法。
#. 支持 **自博弈** （Self-Play）训练。
#. 支持 **自然语言任务** （如对话任务）的强化学习训练。
#. 支持 `DeepSpeed <../quick_start/train_nlp.html#deepspeed>`_ 训练。
#. 支持 **竞技场** 功能，可以在多智能体对抗性环境中方便地对各种智能体进行评测。支持对 `及第平台 <http://www.jidiai.cn/>`_ 的提交进行本地测试。
#. 支持 `Hugging Face <https://huggingface.co/models>`_ 上的模型导入。支持加载Hugging Face上 `Stable-baselines3的模型 <https://openrl-docs.readthedocs.io/zh/latest/sb3/index.html>`_ 来进行测试和训练。
#. 支持LSTM，GRU，Transformer等模型。
#. 支持多种训练加速，例如：混合精度训练，半精度策略网络收集数据等。
#. 支持 `gymnasium <https://gymnasium.farama.org/>`_ 环境。
#. 支持词典类型的观测输入。
#. 支持 `wandb <https://wandb.ai/>`_ (更多请查看 `wandb知乎教程 <https://www.zhihu.com/column/c_1494418493903155200>`_)和 `tensorboardX <https://tensorboardx.readthedocs.io/en/latest/index.html>`_ 等主流机器学习训练可视化平台。
#. 支持环境的串行和并行训练，同时保证两种模式下的训练效果一致。
#. 提供代码覆盖测试和单元测试。

在接下来的 `快速上手 <../quick_start/index.html>`_ 中，我们将介绍如何安装OpenRL框架，
并通过简单的例子来说明如何使用OpenRL。

用户还可以在 `Gallery <https://github.com/OpenRL-Lab/openrl/blob/main/Gallery.md>`_ 中查看OpenRL所支持的算法和环境，并获取对应的代码。

Citing OpenRL
------------------------

如果我们的工作对你有帮助，欢迎引用我们:

.. code-block:: bibtex

    @misc{openrl2023,
        title={OpenRL},
        author={OpenRL Contributors},
        publisher = {GitHub},
        howpublished = {\url{https://github.com/OpenRL-Lab/openrl}},
        year={2023},
    }
