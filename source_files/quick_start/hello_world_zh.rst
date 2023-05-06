开始智能体训练
============================

.. toctree::
   :maxdepth: 2

训练环境
-------

OpenRL为用户提供了一个简单易用的使用方式，这里我们以 `CartPole <https://gymnasium.farama.org/environments/classic_control/cart_pole/>`_ 环境为例，
展示如何使用 OpenRL 进行强化学习训练。新建一个文件 ``train_ppo.py``，输入如下代码.

.. code-block:: python

    # train_ppo.py
    from openrl.envs.common import make
    from openrl.modules.common import PPONet as Net
    from openrl.runners.common import PPOAgent as Agent

    env = make("CartPole-v1", env_num=9) # 创建环境，并设置环境并行数为9
    net = Net(env) # 创建神经网络
    agent = Agent(net) # 初始化训练器
    agent.train(total_time_steps=20000) # 开始训练，并设置环境运行总步数为20000

在终端执行 python train_ppo.py，即可开始训练。在普通笔记本上，仅需要 **几秒钟** ，便可以在完成智能体的训练。

.. tip::

    openrl还提供了命令行工具，可以通过一行命令完成智能体训练。用户只需要在终端执行以下命令即可：

    .. code-block:: bash

        openrl --mode train --env CartPole-v1

测试环境
-------

当智能体完成训练后，我们可以通过 ``agent.act()`` 方法，可以获取智能体的动作。
只需要在 ``train_ppo.py`` 文件中，添加如下代码，即可完成智能体的可视化测试：

.. code-block:: python

    # train_ppo.py
    from openrl.envs.common import make
    from openrl.modules.common import PPONet as Net
    from openrl.runners.common import PPOAgent as Agent

    env = make("CartPole-v1", env_num=9) # 创建环境，并设置环境并行数为9
    net = Net(env) # 创建神经网络
    agent = Agent(net) # 初始化训练器
    agent.train(total_time_steps=20000) # 开始训练，并设置环境运行总步数为20000

    # 创建用于测试的环境，并设置环境并行数为9，设置渲染模式为group_human
    env = make("CartPole-v1", env_num=9, render_mode="group_human")
    agent.set_env(env) # 训练好的智能体设置需要交互的环境
    obs, info = env.reset() # 环境进行初始化，得到初始的观测值和环境信息
    while True:
        action, _ = agent.act(obs) # 智能体根据环境观测输入预测下一个动作
        # 环境根据动作执行一步，得到下一个观测值、奖励、是否结束、环境信息
        obs, r, done, info = env.step(action)
        if any(done): break
    env.close() # 关闭测试环境

在终端执行 python train_ppo.py，即可开始训练并进行可视化测试。
``train_ppo.py`` 代码也可以从 `openrl/examples <https://github.com/OpenRL-Lab/openrl/blob/main/examples/cartpole/train_ppo.py>`_ 处下载。

运行演示：

.. image::
    images/train_ppo_cartpole.gif
    :width: 1000
    :align: center

.. note::

    如果用户在服务器上执行测试代码，将无法看到可视化界面。
    可以通过设置render_mode为group_rgb_array，
    然后在每次step后调用env.render()获取来环境图像。

在接下来的章节中，我们将会以一个更加复杂的多智能体强化学习任务（ `MPE <./multi_agent_RL.html>`_ ）为例子，
介绍如何进行训练超参数的设置，环境并行与串行模式的切换，wandb的使用等等。