训练自然语言对话任务
============

这部分我们将通过一个自然语言对话任务( `DailyDialog <https://arxiv.org/abs/1710.03957>`_ )来介绍如何在OpenRL中导入 `Hugging Face <https://huggingface.co/>`_   **模型和数据**、
如何使用 **自定义奖励模型** 以及如何 **自定义 wandb 的输出** 等。

DailyDialog任务介绍
-------

`DailyDialog <https://arxiv.org/abs/1710.03957>`_ 是一个英文的多轮对话数据集，一共包含1.3万条对话数据。
下图展示了DailyDialog中的一段对话示例：

.. image::
    images/dailydialog_example.png
    :width: 500
    :align: center

过去，这类自然语言任务通常都是使用监督学习来进行训练，
但最近的研究表明，强化学习也可以用于训练语言模型，
并且能显著提升模型的性能（参考[`1 <https://arxiv.org/abs/2203.02155>`_][`2 <https://arxiv.org/abs/2210.01241>`_][`3 <https://openai.com/research/gpt-4>`_]）。

接下来，我们将详细介绍如何使用OpenRL来完成自然语言任务的训练。


创建环境与加载数据
----------------

自然语言任务训练涉及到一些额外包的使用，用户可以通过以下命令完成相关包的安装：

.. code-block:: bash

    pip install "openrl[nlp]"

和前面介绍过的使用教程（`MPE <./multi_agent_RL.html>`_ ）一样，
我们首先需要编写一个 `train_ppo.py` 文件，编写以下训练代码：

.. code-block:: python

    # train_ppo.py
    from openrl.envs.common import make
    from openrl.modules.common import PPONet as Net
    from openrl.runners.common import PPOAgent as Agent
    from openrl.configs.config import create_config_parser
    def train():
        # 添加读取配置文件的代码
        cfg_parser = create_config_parser()
        cfg = cfg_parser.parse_args()
        # 创建 NLP 环境
        env = make("daily_dialog",env_num=2,asynchronous=True,cfg=cfg,)
        # 创建 神经网络
        net = Net(env, cfg=cfg, device="cuda")
        # 创建训练智能体
        agent = Agent(net)
        # 开始训练
        agent.train(total_time_steps=5000000)
        # 保存训练完成的智能体
        agent.save("./ppo_agent/")
    if __name__ == "__main__":
        train()

然后，我们可以创建一个配置文件 `nlp_ppo.yaml` ，并加入以下内容:

.. code-block:: yaml

    # nlp_ppo.yaml
    data_path: daily_dialog # 数据集路径
    env: # 环境所用到的参数
        args: {'tokenizer_path': 'gpt2'} # 读取tokenizer的路径
    seed: 0 # 设置seed，保证每次实验结果一致
    lr: 1e-6 # 设置policy模型的学习率
    critic_lr: 1e-6 # 设置critic模型的学习率
    episode_length: 20 # 设置每个episode的长度
    use_recurrent_policy: true

从上面配置文件可以看出，训练NLP任务需要额外设置 数据集的名称 data_path 和 环境参数env.args 。
其中，data_path 可以设置为 Hugging Face数据集名称 或者 本地数据集路径。
此外，环境参数中的 tokenizer_path 用于指定加载文字编码器的 Hugging Face名称 或者 本地路径。

使用 Hugging Face 的模型进行训练
----

在OpenRL中，我们可以使用 Hugging Face 上的模型来进行训练。
为了加载Hugging Face上的模型，我们首先需要在配置文件`nlp_ppo.yaml`中添加以下内容：

.. code-block:: yaml

    # nlp_ppo.yaml
    model_path: rajkumarrrk/gpt2-fine-tuned-on-daily-dialog # 预训练模型路径
    use_share_model: true # 策略网络和价值网络是否共享模型
    ppo_epoch: 5 # ppo训练迭代次数

    data_path: daily_dialog # 数据集名称或者路径
    env: # 环境所用到的参数
        args: {'tokenizer_path': 'gpt2'} # 读取tokenizer的路径
    lr: 1e-6 # 设置policy模型的学习率
    critic_lr: 1e-6 # 设置critic模型的学习率
    episode_length: 128 # 设置每个episode的长度
    num_mini_batch: 20

然后需要在train_ppo.py中添加以下代码：

.. code-block:: python

    # train_ppo.py
    from openrl.envs.common import make
    from openrl.modules.common import PPONet as Net
    from openrl.runners.common import PPOAgent as Agent
    from openrl.configs.config import create_config_parser
    from openrl.modules.networks.policy_value_network_gpt import (
        PolicyValueNetworkGPT as PolicyValueNetwork,
    )
    def train():
        # 添加读取配置文件的代码
        cfg_parser = create_config_parser()
        cfg = cfg_parser.parse_args()
        # 创建 NLP 环境
        env = make("daily_dialog",env_num=2,asynchronous=True,cfg=cfg,)
        # 创建 神经网络
        model_dict = {"model": PolicyValueNetwork}
        net = Net(env, cfg=cfg, model_dict=model_dict)
        # 创建训练智能体
        agent = Agent(net)
        # 开始训练
        agent.train(total_time_steps=5000000)
        # 保存训练完成的智能体
        agent.save("./ppo_agent/")
    if __name__ == "__main__":
        train()

通过以上简单几行的修改，用户便可以使用Hugging Face上的预训练模型进行训练。

.. note::

     上面这个例子中，我们使用了PolicyValueNetworkGPT这个模型。
     OpenRL还支持用户自定义模型（例如自定模型为CustomedPolicyValueNetwork），然后通过

     .. code-block:: python

         model_dict = {"model": CustomedPolicyValueNetwork}
         net = Net(env, model_dict=model_dict)

    的方式传入训练网络。如果想要分别实现策略网络和价值网络，可以通过

     .. code-block:: python

         model_dict = {
             "policy": CustomedPolicyNetwork,
             "critic": CustomedValueNetwork,
         }
         net = Net(env, model_dict=model_dict)

    来实现。自定义模型的实现方式可以参考 `PolicyValueNetworkGPT <https://github.com/OpenRL-Lab/openrl/blob/main/openrl/modules/networks/policy_value_network_gpt.py>`_、`PolicyNetwork <https://github.com/OpenRL-Lab/openrl/blob/main/openrl/modules/networks/policy_network.py>`_ 以及 `ValueNetwork <https://github.com/OpenRL-Lab/openrl/blob/main/openrl/modules/networks/value_network.py>`_。

使用奖励模型
------------

通常，自然语言任务的数据集中并不包含奖励信息。
因此，如果需要使用强化学习来训练自然语言任务，就需要使用额外的奖励模型来生成奖励。

在该DailyDialog任务中，我们将会使用一个复合的奖励模型，它包含以下三个部分：

- **意图奖励**：即当智能体生成的语句和期望的意图接近时，智能体便可以获得更高的奖励。
- **METEOR指标奖励**： `METEOR <https://en.wikipedia.org/wiki/METEOR>`_ 是一个用于评估文本生成质量的指标，它可以用来衡量生成的语句和期望的语句的相似程度。我们把这个指标作为奖励反馈给智能体，以达到优化生成的语句的效果。
- **KL散度奖励**：该奖励用来限制智能体生成的文本偏离预训练模型的程度，防止出现reward hacking的问题。

我们最终的奖励为以上三个奖励的加权和，其中 **KL散度奖励** 的系数是随着KL散度的大小动态变化的。

想在OpenRL中使用该奖励模型，用户无需修改训练代码，只需要在 `nlp_ppo.yaml` 文件中添加reward_class参数即可：

.. code-block:: yaml

    # nlp_ppo.yaml
    reward_class:
        id: NLPReward # 奖励模型名称
        args: {
            # 用于意图判断的模型的名称或路径
            "intent_model": rajkumarrrk/roberta-daily-dialog-intent-classifier,
            # 用于计算KL散度的预训练模型的名称或路径
            "ref_model": roberta-base, # 用于意图判断的tokenizer的名称或路径
        }

    model_path: rajkumarrrk/gpt2-fine-tuned-on-daily-dialog # 预训练模型路径
    use_share_model: true
    ppo_epoch: 5 # ppo训练迭代次数
    data_path: daily_dialog # 数据集名称或者路径
    env: # 环境所用到的参数
        args: {"tokenizer_path": gpt2} # 读取tokenizer的路径
    lr: 1e-6 # 设置policy模型的学习率
    critic_lr: 1e-6 # 设置critic模型的学习率
    episode_length: 128 # 设置每个episode的长度
    num_mini_batch: 20

.. note::

    OpenRL支持用户使用自定义的奖励模型。
    首先，用户需要编写自定义奖励模型(需要继承 `BaseReward <https://github.com/OpenRL-Lab/openrl/blob/main/openrl/rewards/base_reward.py>`_ 类)。
    接着，用户需要注册自定义的奖励模型，即在train_ppo.py添加以下代码：

    .. code-block:: python

        # train_ppo.py
        from openrl.rewards.nlp_reward import CustomedReward
        from openrl.rewards import RewardFactory
        RewardFactory.register("CustomedReward", CustomedReward)

    最后，用户需要在nlp_ppo.yaml中填写自定义的奖励模型即可：

    .. code-block:: yaml

        reward_class:
            id: "CustomedReward" # 自定义奖励模型名称
            args: {} # 用户自定义奖励函数可能用到的参数

自定义wandb输出
------------

OpenRL还支持用户自定义wandb和tensorboard的输出内容。
例如，在该任务的训练过程中，我们还需要输出各种类型奖励的信息和KL散度系数的信息，
用户可以在nlp_ppo.yaml文件中加入vec_info_class参数来实现:

.. code-block:: yaml

    # nlp_ppo.yaml
    vec_info_class:
        id: "NLPVecInfo" # 调用NLPVecInfo类以打印NLP任务中奖励函数的信息
    #设置wandb信息
    wandb_entity: openrl # 这里用于指定wandb团队名称，请把openrl替换为你自己的团队名称
    experiment_name: train_nlp # 这里用于指定实验名称
    run_dir: ./run_results/ # 这里用于指定实验数据保存的路径
    log_interval: 1 # 这里用于指定每隔多少个episode上传一次wandb数据
    # 自行填写其他参数...

修改完配置文件后，在train_ppo.py文件中启用wandb:

.. code-block:: python

    # train_ppo.py
    agent.train(total_time_steps=100000, use_wandb=True)

然后执行python train_ppo.py --config nlp_ppo.yaml，过一会儿，便可以在wandb中看到如下的输出:

.. image::
    images/nlp_wandb.png
    :width: 1000
    :align: center

从上图可以看到，wandb输出了各种类型奖励的信息和KL散度系数的信息。

如果用户还需要输出其他信息，还可以参考 `NLPVecInfo <https://github.com/OpenRL-Lab/openrl/blob/main/openrl/envs/vec_env/wrappers/vec_info.py>`_ 类
和 `VecInfo <https://github.com/OpenRL-Lab/openrl/blob/main/openrl/envs/vec_env/wrappers/vec_info.py>`_ 类来实现自己的CustomedVecInfo类。
然后，需要在train_ppo.py中注册自定义的CustomedVecInfo类:

.. code-block:: python

    # train_ppo.py
    # 注册自定义输出信息类
    VecInfoFactory.register("CustomedVecInfo", CustomedVecInfo)

最后，只需要在nlp_ppo.yaml中填写CustomedVecInfo类即可：

.. code-block:: yaml

    # nlp_ppo.yaml
    vec_info_class:
        id: "CustomedVecInfo" # 调用自定义CustomedVecInfo类以输出自定义信息

使用混合精度训练加速
--------------

OpenRL还提供了一键开启混合精度训练的功能。用户只需要在配置文件中加入以下参数即可：

.. code-block:: yaml

    # nlp_ppo.yaml
    use_amp: true # 开启混合精度训练


OpenRL训练结果
---------------

下表格展示了使用OpenRL训练该对话任务的结果。结果显示使用强化学习训练后，模型各项指标皆有所提升。
另外，从下表可以看出，相较于 `RL4LMs <https://github.com/allenai/RL4LMs>`_ ，
OpenRL的训练速度更快（提升 xx% ），最终的性能指标也更好。

=============== ======== ======== ========  ========= ============ ======= ========== ================ =========== =================
                 FPS     Rouge-1   Rouge-2   Rouge-L   Rouge-Lsum   Meteor  SacreBLEU  Intent Accuracy  perplexity  mean_pred_length
=============== ======== ========  ========  ========= ============ ======= ========== ================ =========== =================
监督学习          -       0.164     0.018     0.137     0.137        0.234   0.063      0.4265           40.91       18.95
RL4LMs           xxxx         xx     xxx    xxx     xxx        xxx   xxx      xxx           xxx       xxx
OpenRL          **xxxx **      **0.182**   **0.020**     **0.154**     **0.154**        **0.296**   **0.093**      **0.4274**           **44.03**      **18.64**
=============== ========  ========  ========= ============ ======= ========== ================ =========== =================
