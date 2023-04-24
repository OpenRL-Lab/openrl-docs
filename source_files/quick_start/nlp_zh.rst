训练对话任务
============

RLHF(Reinforcement Learning from Human Feedback)
是一种基于人类反馈数据与强化学习算法训练语言模型的方法，
也是ChatGPT等问答模型所采用的训练方法。
我们将在本教程中将以 `DailyDialog <http://yanran.li/files/ijcnlp2017dailydialog.pdf>`_ 
任务为例，介绍如何使用PPO+RLHF方法fine-tune预训练语言模型。

以下展示了DailyDialog中的一个实例，给定多轮对话，
智能体需要根据上下文回复特定语句以完成对话。
其中Prompt是模型输入，
Supervised代表预训练模型输出，
Supervised+PPO代表利用RLHF方法fine-tune过后的模型输出。

**Prompt:**

Alright . Well . We have many units without auto-reverse.Here is a good unit . Very reliable.It has an AM / FM band , built-in microphone , recording capability , and no auto-reverse . 

How much is it ? 

This one sells for $ 39.99 . 

Can I test it out ? 

Of course .  

**Supervised:** Great. It sounds great. I must say, this is the most expensive unit on the market.

**Supervised+PPO:** Great. It sounds great. I'll take one back.


创建基础对话任务
----------------

创建对话任务和创建 `MPE <./multi_agent_RL.html>`_ 环境一样，
我们需要编写train_ppo.py文件和nlp_ppo.yaml文件

在nlp_ppo.yaml配置文件中加入以下内容:

.. code-block:: yaml

    # nlp_ppo.yaml
    wandb_entity: tmarl # 这里用于指定wandb团队名称，请把openrl替换为你自己的团队名称
    experiment_name: ppo # 这里用于指定实验名称
    run_dir: ../../../exp_results/ # 这里用于指定实验数据保存的路径
    log_interval: 1 # 这里用于指定每隔多少个episode上传一次wandb数据
    seed: 0 # 设置seed，保证每次实验结果一致
    lr: 1e-6 # 设置policy模型的学习率
    critic_lr: 1e-6 # 设置critic模型的学习率
    episode_length: 20 # 设置每个episode的长度
    use_recurrent_policy: true # 设置是否使用RNN
    use_joint_action_loss: false # 设置是否使用JRPO算法
    use_valuenorm: true # 设置是否使用value normalization
    use_adv_normalize: true # 设置是否使用advantage normalization
    data_path: daily_dialog # 数据集路径
    env: # 环境所用到的参数
        args: {'model_path': 'gpt2'} # 读取tokenizer的路径

在train_ppo.py中编写训练代码

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

        # 创建 NLP 环境，使用异步环境，即每个智能体独立运行
        env = make(
            "daily_dialog",
            env_num=2,
            asynchronous=True,
            cfg=cfg,
        )
        # 创建 神经网络，传入超参数的配置
        net = Net(env, cfg=cfg, device="cuda")

        # 使用wandb
        agent = Agent(net, use_wandb=True)

        # 开始训练
        agent.train(total_time_steps=5000000)
        # 保存训练完成的智能体
        agent.save("./ppo_agent/")
    
    if __name__ == "__main__":
        train()

相比于 `MPE <./multi_agent_RL.html>`_ 环境，
NLP环境需要额外设置data_path与env.args两个参数。
其中环境会通过hugging face接口，
下载data_path下的数据集以及env.args["model_path"]下的tokenizer


加载奖励函数
============

奖励函数是RLHF中重要的元素之一，
在给定奖励函数后，RL将尝试最大化此奖励，
以达到优化预训练模型的效果

奖励函数简介
------------

此处，我们选用与 `RL4LMs <https://github.com/allenai/RL4LMs>`_ 中相同的奖励函数，包含三项:

- 第一，KL散度惩罚，即限制RL训练所得模型偏离预训练模型的程度。
- 第二，意图奖励，即当模型预测的意图与真实标签相同时，可以获得更高的奖励。
- 第三， `METEOR指标 <https://en.wikipedia.org/wiki/METEOR>`_

环境接口
--------

想在OpenRL中使用特定的奖励函数，
只需要在nlp_ppo.yaml文件中添加reward_class参数即可

.. code-block:: yaml

    # nlp_ppo.yaml
    wandb_entity: tmarl # 这里用于指定wandb团队名称，请把openrl替换为你自己的团队名称
    experiment_name: ppo # 这里用于指定实验名称
    run_dir: ../../../exp_results/ # 这里用于指定实验数据保存的路径
    log_interval: 1 # 这里用于指定每隔多少个episode上传一次wandb数据
    seed: 0 # 设置seed，保证每次实验结果一致
    lr: 1e-6 # 设置policy模型的学习率
    critic_lr: 1e-6 # 设置critic模型的学习率
    episode_length: 20 # 设置每个episode的长度
    use_recurrent_policy: true # 设置是否使用RNN
    use_joint_action_loss: false # 设置是否使用JRPO算法
    use_valuenorm: true # 设置是否使用value normalization
    use_adv_normalize: true # 设置是否使用advantage normalization
    data_path: daily_dialog # 数据集路径
    env: # 环境所用到的参数
        args: {'model_path': 'gpt2'} # 读取tokenizer的路径
    reward_class: # 奖励函数所用到的参数
        id: "NLPReward" # 奖励函数类名
        # 奖励函数会用到的模型名
        args: { 
            "reward_path": "rajkumarrrk/roberta-daily-dialog-intent-classifier",
            "model_path": "gpt2",
        }


reward_class中包含id和args两个参数。
其中id为奖励函数类的名称。
NLPReward类的args参数中有两个参数，
reward_path中传递的是意图奖励所用到的模型名称，
model_path中传递的是KL散度惩罚所用到的模型名称。
给定参数后OpenRL会从hugging face上自动下载指定模型。


使用自定义奖励函数
------------------

OpenRL支持用户使用自定义奖励模型。
首先，用户需要编写自定义奖励函数(继承于openrl/rewards/base_reward中BaseReward类)。

接着，用户需要注册自定义奖励函数，即在train_ppo.py添加以下代码

.. code-block:: python

    def train():
        # 添加读取配置文件的代码
        cfg_parser = create_config_parser()
        cfg = cfg_parser.parse_args()

        # 在创建环境前注册自定义奖励函数
        from openrl.rewards.nlp_reward import CustomedReward
        from openrl.rewards import RewardFactory
        RewardFactory.register("CustomedReward", CustomedReward)
        
        # 创建 NLP 环境，使用异步环境，即每个智能体独立运行     
        env = make(
            "daily_dialog",
            env_num=2,
            asynchronous=True,
            cfg=cfg,
        )

最终，用户需要在nlp_ppo.yaml中选择自定义的奖励函数，即在nlp_ppo.yaml添加以下代码

.. code-block:: yaml

    reward_class: # 自定义奖励函数所用到的参数
        id: "CustomedReward" # 自定义奖励函数类名
        args: {} # 其他自定义奖励函数可能用到的参数

自定义wandb输出
================

OpenRL支持用户自定义wandb输出内容。
如在RLHF训练过程中，用户希望看到各项奖励函数的变化曲线，
可以通过在nlp_ppo.yaml文件中加入以下代码实现:

.. code-block:: yaml

    # nlp_ppo.yaml
    wandb_entity: tmarl # 这里用于指定wandb团队名称，请把openrl替换为你自己的团队名称
    experiment_name: ppo # 这里用于指定实验名称
    run_dir: ../../../exp_results/ # 这里用于指定实验数据保存的路径
    log_interval: 1 # 这里用于指定每隔多少个episode上传一次wandb数据
    seed: 0 # 设置seed，保证每次实验结果一致
    lr: 1e-6 # 设置policy模型的学习率
    critic_lr: 1e-6 # 设置critic模型的学习率
    episode_length: 20 # 设置每个episode的长度
    use_recurrent_policy: true # 设置是否使用RNN
    use_joint_action_loss: false # 设置是否使用JRPO算法
    use_valuenorm: true # 设置是否使用value normalization
    use_adv_normalize: true # 设置是否使用advantage normalization
    data_path: daily_dialog # 数据集路径
    env: # 环境所用到的参数
        args: {'model_path': 'gpt2'} # 读取tokenizer的路径
    reward_class: # 奖励函数所用到的参数
        id: "NLPReward" # 奖励函数类名
        # 奖励函数会用到的模型名
        args: { 
            "reward_path": "rajkumarrrk/roberta-daily-dialog-intent-classifier",
            "model_path": "gpt2",
        }
    vec_info_class: 
        id: "NLPVecInfo" # 调用NLPVecInfo类以打印NLP任务中奖励函数的信息

使用自定义输出
---------------

此外用户也可以自定义wandb输出内容，
首先编写VecInfo类，接着在train_ppo.py中注册自定义VecInfo

.. code-block:: python

    def train():
        # 添加读取配置文件的代码
        cfg_parser = create_config_parser()
        cfg = cfg_parser.parse_args()

        # 注册自定义奖励函数
        from openrl.envs.vec_env.wrappers.vec_info import CustomedVecInfo
        from openrl.envs.vec_env.wrappers.vec_monitor import VecInfoFactory
        VecInfoFactory.register("CustomedVecInfo", CustomedVecInfo)
        
        # 创建 NLP 环境，使用异步环境，即每个智能体独立运行     
        env = make(
            "daily_dialog",
            env_num=2,
            asynchronous=True,
            cfg=cfg,
        )

最终在nlp_ppo.yaml中选用自定义VecInfo

.. code-block:: yaml

    vec_info_class: 
        id: "CustomedVecInfo" # 调用自定义VecInfo类以打印自定义信息

导入hugging face模型
====================

`Hugging_Face <https://huggingface.co/>`_ 是一个开源的机器学习模型与数据平台。
以下介绍如何从Hugging Face上导入预训练模型以加速训练。

使用hugging face模型
---------------------

为了加载预训练模型(此处以gpt2-fine-tuned-on-daily-dialog为例)，
我们需要对train_ppo.py做出以下改进:

.. code-block:: python

    import numpy as np

    from openrl.configs.config import create_config_parser
    from openrl.envs.common import make
    from openrl.modules.common import PPONet as Net
    from openrl.modules.networks.policy_value_network_gpt import (
        PolicyValueNetworkGPT as PolicyValueNetwork,
    )
    from openrl.runners.common import PPOAgent as Agent

    def train():
        debug = False
        # 创建 环境

        cfg_parser = create_config_parser()
        cfg = cfg_parser.parse_args()

        env_num = 2 if debug else 10
        env = make(
            "daily_dialog",
            env_num=env_num,
            asynchronous=not debug,
            cfg=cfg,
        )

        # 创建 神经网络
        model_dict = {"model": PolicyValueNetwork}
        net = Net(env, device="cuda", cfg=cfg, model_dict=model_dict)

        # 初始化训练器
        agent = Agent(net, use_wandb=not debug)
        # 开始训练
        agent.train(total_time_steps=100000)
        agent.save("./ppo_agent")

        env.close()
        return agent

    if __name__ == "__main__":
        agent = train()

此外，由于此模型的policy与value实现在同一个类中，
因此需要设置share_model参数，
最终nlp_ppo.yaml文件如下所示:

.. code-block:: yaml

    # nlp_ppo.yaml
    wandb_entity: tmarl # 这里用于指定wandb团队名称，请把openrl替换为你自己的团队名称
    experiment_name: ppo # 这里用于指定实验名称
    run_dir: ../../../exp_results/ # 这里用于指定实验数据保存的路径
    log_interval: 1 # 这里用于指定每隔多少个episode上传一次wandb数据
    seed: 0 # 设置seed，保证每次实验结果一致
    lr: 1e-6 # 设置policy模型的学习率
    critic_lr: 1e-6 # 设置critic模型的学习率
    episode_length: 128 # 设置每个episode的长度
    use_recurrent_policy: true # 设置是否使用RNN
    use_joint_action_loss: false # 设置是否使用JRPO算法
    use_valuenorm: true # 设置是否使用value normalization
    use_adv_normalize: true # 设置是否使用advantage normalization
    data_chunk_length: 1 # 相当于不使用RNN 
    num_mini_batch: 20 # 使得batch_size=64
    ppo_epoch: 5 # ppo训练迭代次数
    use_share_model: true # policy与value实现在同一个类中
    model_path: rajkumarrrk/gpt2-fine-tuned-on-daily-dialog # 预训练模型路径
    data_path: daily_dialog # 数据集路径
    
    env: # 环境所用到的参数
        args: {'model_path': 'gpt2'} # 读取tokenizer的路径

    vec_info_class:
        id: "NLPVecInfo" # 调用指定类以打印指定信息

    reward_class: # 奖励函数所用到的参数
        id: "NLPReward" # 奖励函数类名
        # 奖励函数会用到的模型名
        args: { 
            "reward_path": "rajkumarrrk/roberta-daily-dialog-intent-classifier",
            "model_path": "rajkumarrrk/gpt2-fine-tuned-on-daily-dialog",
        }

使用自定义模型
---------------

OpenRL支持用户使用自定义模型。
首先用户需要编写自定义模型。
接着在train_ppo.py中选择该模型，通过model_dict传入。

若用户自定义模型policy与value网络实现在同一个类中，
需要在nlp_ppo.yaml文件中设置use_share_model参数为true。
并通过以下方法选用自定义模型。

.. code-block:: python

    # train_ppo.py
    import CustomedPolicyValueNetwork
    # 创建自定义神经网络 (policy与value网络实现在同一个类中)
    model_dict = {"model": CustomedPolicyValueNetwork}
    net = Net(env, device="cuda", cfg=cfg, model_dict=model_dict)

若用户在两个类中分别实现了自定义policy与value网络，
需要在nlp_ppo.yaml文件中设置use_share_model参数为false。
并通过以下方法选用自定义模型。

.. code-block:: python

    # train_ppo.py
    import CustomedPolicyNetwork, CustomedValueNetwork 
    # 创建自定义神经网络 (policy与value网络实现在同一个类中)
    model_dict = {
        "policy": CustomedPolicyNetwork,
        "critic": CustomedValueNetwork,
    }
    net = Net(env, device="cuda", cfg=cfg, model_dict=model_dict)

评测结果
=========

此处展示OpenRL在daily_dialog任务上的各项指标，
结果显示使用RLHF fine-tune过后，模型各项指标皆有所提升。

=============== ========  ========  ========= ============ ======= ========== ================ =========== =================
algorithm       Rouge-1   Rouge-2   Rouge-L   Rouge-Lsum   Meteor  SacreBLEU  Intent Accuracy  perplexity  mean_pred_length 
=============== ========  ========  ========= ============ ======= ========== ================ =========== =================
supervised      0.164     0.018     0.137     0.137        0.234   0.063      0.4265           40.91       18.95
supervised+PPO  0.182     0.020     0.154     0.154        0.296   0.093      0.4274           44.03       18.64
=============== ========  ========  ========= ============ ======= ========== ================ =========== =================
2