加载Hugging Face上Stable-baselines3的模型
=====================

.. toctree::
   :maxdepth: 1

`Stable-baseline3 <https://github.com/DLR-RM/stable-baselines3>`_ 实现了很多强化学习算法，并在 `Hugging Face <https://huggingface.co/sb3>`_ 上分享了训练出来的模型。
OpenRL可以加载这些模型，然后使用OpenRL进行测试和训练。我们在 `这里 <https://github.com/OpenRL-Lab/openrl/tree/main/examples/sb3>`_ 给出了一个加载Stable-baseline3模型的示例。

环境安装
-------

首先我们需要通过 ``pip`` 来安装一些必要的包:

.. code-block:: bash

    pip install huggingface-tool rl_zoo3

下载模型
-------

安装好 ``huggingface-tool`` 工具后，我们可以通过 ``htool`` 命令来下来我们需要的模型。
这里，我们以 `sb3/ppo-CartPole-v1 <https://huggingface.co/sb3/ppo-CartPole-v1>`_ 为例，其下载命令如下:

.. code-block:: bash

    htool save-repo sb3/ppo-CartPole-v1 ppo-CartPole-v1

这里 ``sb3/ppo-CartPole-v1`` 是模型的地址， ``ppo-CartPole-v1`` 是我们保存的模型的名字。 ``htool`` 会自动下载模型并保存到 ``ppo-CartPole-v1`` 目录下。

加载Stable-baselines3模型并用于测试
---------------

模型下载完后，我们便可以用过OpenRL来加载该模型并进行测试，该部分完整代码可见 `这里 <https://github.com/OpenRL-Lab/openrl/blob/main/examples/sb3/test_model.py>`_ ：

.. code-block:: python

    # test_model.py
    import numpy as np
    import torch

    from openrl.configs.config import create_config_parser
    from openrl.envs.common import make
    from openrl.modules.common.ppo_net import PPONet as Net
    from openrl.modules.networks.policy_value_network_sb3 import (
        PolicyValueNetworkSB3 as PolicyValueNetwork,
    )
    from openrl.runners.common import PPOAgent as Agent

    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(["--config", "ppo.yaml"])
    env = make("CartPole-v1",  env_num=9, asynchronous=True)
    model_dict = {"model": PolicyValueNetwork}
    net = Net(
        env,
        cfg=cfg,
        model_dict=model_dict,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    agent = Agent(net)
    agent.set_env(env)
    obs, info = env.reset()
    done = False

    while not np.any(done):
        action, _ = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(action)
    env.close()



另外还需要写一个配置文件，用于配置模型的路径：

.. code-block:: yaml

    # ppo.yaml
    sb3_model_path: ppo-CartPole-v1/ppo-CartPole-v1.zip
    sb3_algo: ppo
    use_share_model: true

这样，我们就可以通过 ``python test_model.py`` 来加载Stable-baselines3的模型并进行测试了。


加载Stable-baselines3模型并用于训练
--------------------------------

模型下载完后，我们还可以用过OpenRL来加载该模型并用于训练，该部分完整代码可见 `这里 <https://github.com/OpenRL-Lab/openrl/blob/main/examples/sb3/train_ppo.py>`_ ：

.. code-block:: python

    # train_ppo.py
    import numpy as np
    import torch

    from openrl.configs.config import create_config_parser
    from openrl.envs.common import make
    from openrl.modules.common.ppo_net import PPONet as Net
    from openrl.modules.networks.policy_value_network_sb3 import (
        PolicyValueNetworkSB3 as PolicyValueNetwork,
    )
    from openrl.runners.common import PPOAgent as Agent

    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args(["--config", "ppo.yaml"])
    env = make("CartPole-v1", env_num=8, asynchronous=True)
    model_dict = {"model": PolicyValueNetwork}
    net = Net(
        env,
        cfg=cfg,
        model_dict=model_dict,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    agent = Agent(net)
    agent.train(total_time_steps=100000)
    agent.save("./ppo_sb3_agent")

另外还需要写一个配置文件，用于配置模型的路径和训练超参数：

.. code-block:: yaml

    # ppo.yaml
    sb3_model_path: ppo-CartPole-v1/ppo-CartPole-v1.zip
    sb3_algo: ppo
    use_share_model: true
    entropy_coef: 0.0
    gae_lambda: 0.8
    gamma: 0.98
    lr: 0.001
    episode_length: 32
    ppo_epoch: 20

这样，我们就可以通过 ``python train_ppo.py`` 来加载Stable-baselines3的模型并进行训练了。