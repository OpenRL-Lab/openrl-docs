Loading Stable-baselines3 Models from Hugging Face
=====================

.. toctree::
   :maxdepth: 1

`Stable-baseline3 <https://github.com/DLR-RM/stable-baselines3>`_ implements many reinforcement learning algorithms and has shared trained models on `Hugging Face <https://huggingface.co/sb3>`_. OpenRL can load these models and then test and train them using OpenRL. An example of loading Stable-baseline3 models is provided `here <https://github.com/OpenRL-Lab/openrl/tree/main/examples/sb3>`_.

Environment Setup
-------

First, we need to install some essential packages using ``pip``:

.. code-block:: bash

    pip install huggingface-tool rl_zoo3

Downloading the Model
-------

After installing the ``huggingface-tool`` utility, we can download the model we need using the ``htool`` command. For instance, to download `sb3/ppo-CartPole-v1 <https://huggingface.co/sb3/ppo-CartPole-v1>`_, use the following command:

.. code-block:: bash

    htool save-repo sb3/ppo-CartPole-v1 ppo-CartPole-v1

Here, ``sb3/ppo-CartPole-v1`` is the model's address, and ``ppo-CartPole-v1`` is the name we're giving to the downloaded model. ``htool`` will automatically download and save the model under the ``ppo-CartPole-v1`` directory.

Load Stable-baselines3 Model and Test
---------------

Once the model is downloaded, we can load it using OpenRL and perform testing. The complete code for this section is available `here <https://github.com/OpenRL-Lab/openrl/blob/main/examples/sb3/test_model.py>`_:


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

Moreover, we need to write a configuration file to set the model's path:

.. code-block:: yaml

    # ppo.yaml
    sb3_model_path: ppo-CartPole-v1/ppo-CartPole-v1.zip
    sb3_algo: ppo
    use_share_model: true

With this, we can test the Stable-baselines3 model using ``python test_model.py``.

Load Stable-baselines3 Model and Train
--------------------------------

After downloading the model, we can also load it using OpenRL and train it further. The complete code for this section can be found `here <https://github.com/OpenRL-Lab/openrl/blob/main/examples/sb3/train_ppo.py>`_:

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

Additionally, we need to create a configuration file to specify the model path and training hyperparameters:

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

With this setup, we can train the Stable-baselines3 model using ``python train_ppo.py``.