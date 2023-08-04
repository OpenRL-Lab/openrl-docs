Self-Play Training
====================

.. toctree::
   :maxdepth: 1

OpenRL is one of the reinforcement learning frameworks that support self-play training.
Self-play involves more complex algorithms and interaction processes, which pose higher demands on the framework's design and implementation.
OpenRL simplifies self-play training and algorithm implementation by using callback designs and abstracting and modularizing various components.

Users can get started with self-play training by using the self-play example we provide, which can be found in `examples/self_play <https://github.com/OpenRL-Lab/openrl/tree/main/examples/selfplay>`_.

Performing Self-Play Training with OpenRL
----------------------------------------

Before starting self-play training, you need to install the dependencies related to self-play training.
You can do this with the following command:

.. code-block:: bash

    pip install "openrl[selfplay]"

In the OpenRL framework, the entry code for self-play training is the same as non-self-play training.
Most of the self-play configuration is done through a YAML file.
Below is an example of our training code:

.. code-block:: python

    import numpy as np
    import torch
    from openrl.configs.config import create_config_parser
    from openrl.envs.common import make
    from openrl.envs.wrappers import FlattenObservation
    from openrl.modules.common import PPONet as Net
    from openrl.runners.common import PPOAgent as Agent
    from openrl.selfplay.wrappers.opponent_pool_wrapper import OpponentPoolWrapper
    from openrl.selfplay.wrappers.random_opponent_wrapper import RandomOpponentWrapper

    def train():
        cfg_parser = create_config_parser()
        cfg = cfg_parser.parse_args(["--config", "selfplay.yaml"])
        # Create environment
        env = make(
            "tictactoe_v3",
            env_num=10,
            asynchronous=True,
            opponent_wrappers=[OpponentPoolWrapper],
            env_wrappers=[FlattenObservation],
            cfg=cfg,
        )
        # Create agent
        agent = Agent(Net(env, cfg=cfg))
        # Begin training
        agent.train(total_time_steps=20000)
        env.close()

In this example, we use the `tictactoe_v3 <https://pettingzoo.farama.org/environments/classic/tictactoe/>`_ environment from `PettingZoo <https://pettingzoo.farama.org/index.html>`_ as our training environment.

For self-play training, we use the ``OpponentPoolWrapper`` to wrap the environment. This wrapper selects an opponent for each episode's reset based on the opponent selection strategy.
To configure the opponent selection strategy, we need to use a YAML file. In this example, we use the ``selfplay.yaml`` configuration file, which contains the following content:

.. code-block:: yaml

    globals:
      selfplay_api_host: 127.0.0.1
      selfplay_api_port: 10086

    seed: 0
    selfplay_api:
      host: {{ selfplay_api_host }}
      port: {{ selfplay_api_port }}
    lazy_load_opponent: true # if true, when the opponents are the same opponent_type, will only load the weight. Otherwise, will load the python script.
    callbacks:
      - id: "SelfplayAPI"
        args: {
            host: {{ selfplay_api_host }},
            port: {{ selfplay_api_port }},
            sample_strategy: "RandomOpponent",
        }
      - id: "SelfplayCallback"
        args: {
            "save_freq": 100, # how often to save the model
            "opponent_pool_path": "./opponent_pool/",  # where to save opponents
            "name_prefix": "opponent", # the prefix of the saved model
            "api_address": "http://{{ selfplay_api_host }}:{{ selfplay_api_port }}/selfplay/",
            "opponent_template": "./opponent_templates/tictactoe_opponent",
            "clear_past_opponents": true,
            "copy_script_file": false,
             "verbose": 2,
        }

Since most of our self-play configurations are defined in the YAML file, understanding the content of the YAML file is crucial.

First, our YAML configuration supports global variables. You can define global variables under ``globals``, and then use them elsewhere with ``{{ variable_name }}``.
Since we use ``selfplay_api_host`` and ``selfplay_api_port`` multiple times, we define them as global variables.

Next, we configure the ``selfplay_api`` section, which is used to set up the self-play API. Users can specify the API's address and port using ``host`` and ``port``.
Different environments use this API for opponent querying and selection.

Then, the ``lazy_load_opponent`` parameter. If ``lazy_load_opponent`` is set to ``true``, when a new opponent of the same opponent_type is sampled (using the same python script as the previous opponent),
we will only load the opponent's neural network weights instead of the entire python script.
When ``lazy_load_opponent`` is set to ``false``, we will load the opponent's python script, ensuring that the opponent's script is up-to-date, but this increases the loading time, which is generally unnecessary.

Next, we configure two callbacks, ``SelfplayAPI`` and ``SelfplayCallback``.

``SelfplayAPI`` is a callback used to start the self-play API. It starts the self-play API when training begins and closes it when training ends.
We need to specify the ``sample_strategy`` parameter, which is used to specify the opponent sampling strategy.
We have abstracted and modularized the sampling strategy, and users can refer to `here <https://github.com/OpenRL-Lab/openrl/tree/main/openrl/selfplay/sample_strategy>`_ to implement their own opponent sampling strategy.

``SelfplayCallback`` is a callback used to save opponents periodically. It saves opponents every ``save_freq`` steps in the directory specified by ``opponent_pool_path``.
We also specify the ``opponent_template`` parameter, which is used to specify the template for the opponent's python script. We generate the opponent's python script based on the ``opponent_template``.
If ``copy_script_file`` is set to ``true``, we directly copy all python files from ``opponent_template`` to the new opponent's directory. Otherwise, we create symbolic links to all python files from ``opponent_template`` in the new opponent's directory.
Generally, we set ``copy_script_file`` to ``false`` to save disk space and meet most requirements.

Regarding the opponent template, you can refer to `here <https://github.com/OpenRL-Lab/openrl/tree/main/examples/selfplay/opponent_templates/tictactoe_opponent>`_.
Each opponent template must contain at least two files: a json file ``info.json`` that describes the opponent's type and detailed information, and a python file ``opponent.py`` that implements an opponent class to handle observation inputs and actions.