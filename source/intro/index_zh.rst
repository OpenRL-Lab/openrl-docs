OpenRL 介绍
===============================

OpenRL 强化学习框架
-------------------------------

OpenRL 是由第四范式强化学习团队开发的基于PyTorch的强化学习研究框架，它提供了一个简单易用的接口，可以让你方便地接入不同强化学习环境。
目前，OpenRL框架有以下特点：

1. 简单易用的训练接口，降低研究人员的学习和使用成本。

2. 同时支持 **单智能体** 和 **多智能体** 算法。

3. 支持 **自然语言任务** （如对话任务）的强化学习训练。

4. 支持 `Hugging Face <https://huggingface.co/models>`_ 上的模型导入。

5. 支持LSTM，GRU，Transformer等模型。

6. 支持 `gymnasium <https://gymnasium.farama.org/>`_ 环境。

7. 支持词典类型的观测输入。

8. 支持 `wandb <https://wandb.ai/>`_ (更多请查看 `wandb知乎教程 <https://www.zhihu.com/column/c_1494418493903155200>`_)和 `tensorboardX <https://tensorboardx.readthedocs.io/en/latest/index.html>`_ 等主流机器学习训练可视化平台。

9. 支持环境的串行和并行训练，同时保证两种模式下的训练效果一致。

10. 提供代码覆盖测试和单元测试。

在接下来的 `快速上手 <../quick_start/index.html>`_ 中，我们将介绍如何安装OpenRL框架，
并通过简单的例子来说明如何使用OpenRL。
