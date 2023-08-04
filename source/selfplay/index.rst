自博弈训练
=====================

.. toctree::
   :maxdepth: 1

OpenRL是为数不多的支持自博弈的强化学习框架。自博弈由于涉及到更加复杂的算法和更加复杂的交互流程，对于框架的设计和实现都提出了更高的要求。
OpenRL通过对Callback的设计以及对于自博弈训练中各个部分的抽象与模块化，使得自博弈的训练和算法的实现变得更加简单。

用户可以通过我们提供一个自博弈例子来开始上手自博弈的训练，这个例子可以在 `examples/self_play <https://github.com/OpenRL-Lab/openrl/tree/main/examples/selfplay>`_ 中找到。

通过OpenRL进行自博弈训练
-------------------

在进行自博弈训练前，需要安装自博弈训练相关的依赖，可以通过以下命令安装：

.. code-block:: bash

    pip install "openrl[selfplay]"

在OpenRL框架中，自博弈训练的入口代码和非自博弈训练的入口代码是一样的，我们的大部分自博弈部分的配置都是通过YAML文件来进行的。以下使我们训练代码的示例：

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

在该示例中，我们使用了 `PettingZoo <https://pettingzoo.farama.org/index.html>`_ 中的 `tictactoe_v3 <https://pettingzoo.farama.org/environments/classic/tictactoe/>`_ 环境作为我们的训练环境。

为了进行自博弈训练，我们使用了 ``OpponentPoolWrapper`` 来对环境进行包装，这个包装器会在环境每次reset时根据对手选择策略选择一个对手进行对战。
而对于对手选择策略，我们需要通过YAML文件来进行配置。在实例中，我们使用了 ``selfplay.yaml`` 这个配置文件，以下是这个配置文件的内容：

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

由于我们自博弈的大部分配置都是通过YAML文件来进行的，因此了解YAML文件里面的内容非常重要。

首先，我们的YAML配置是支持全局变量的，用户可以在 ``globals`` 下定义全局变量，然后在其他地方通过 ``{{ variable_name }}`` 来使用这些全局变量。
因为我们这里多次使用了 ``selfplay_api_host`` 和 ``selfplay_api_port`` 这两个变量，因此我们将它们定义为全局变量。

然后，我们配置了 ``selfplay_api`` 这个部分，这个部分是用于配置自博弈API的，用户可以通过 ``host`` 和 ``port`` 来指定API的地址和端口。
不同环境的对手便是使用这个API来进行查询和选择的。

然后，``lazy_load_opponent`` 这个参数。如果 ``lazy_load_opponent`` 为 ``true`` ，那么当采样出来的新对手的类型和上个对手的类型相同时（都是使用同样的python脚本），我们就只会加载对手的神经网络得权重，而不会加载对手的python脚本。
当 ``lazy_load_opponent`` 为 ``false`` 时，我们会加载对手的python脚本，这样可以保证对手的python脚本是最新的，但是会增加加载对手的时间，但通常来说这是没有必要的。

然后，我们配置了两个callback， ``SelfplayAPI`` 和 ``SelfplayCallback`` 。

``SelfplayAPI`` 是一个用于启动自博弈API的callback，它会在训练开始时启动自博弈API，然后在训练结束时关闭自博弈API。
同时，我们需要指定 ``sample_strategy`` 这个参数，这个参数是用于指定对手的采样策略的。
我们对采样策略做了一定的抽象和模块化，用户可以参考 `这里 <https://github.com/OpenRL-Lab/openrl/tree/main/openrl/selfplay/sample_strategy>`_ 来实现自己的对手采样策略。

``SelfplayCallback`` 是一个用于定期保存对手的callback，它会在每 ``save_freq`` 步保存一次对手，保存的路径为 ``opponent_pool_path`` 。
我们还会指定 ``opponent_template`` 这个参数。这个参数是用于指定对手的python脚本的模板，我们会在 ``opponent_template`` 的基础上生成对手的python脚本。
如果 ``copy_script_file`` 为 ``true`` ，那么我们会直接复制 ``opponent_template`` 里面的所有python文件到新对手的目录下，否则我们会在新对手的目录下创建软链接到``opponent_template``下的所有python文件。
一般来说，我们设置 ``copy_script_file`` 为 ``false`` ，这样可以节省磁盘空间，也能满足大部分时候的需求。

关于对手模板的写法，可以参考 `这里 <https://github.com/OpenRL-Lab/openrl/tree/main/examples/selfplay/opponent_templates/tictactoe_opponent>`_ 。
每个对手模板至少包含两个文件，一个json文件 ``info.json`` 用于描述对手的类型和详细信息，以及一个python文件 ``opponent.py`` 用于实现一个对手类来完成观测输入和动作的处理。