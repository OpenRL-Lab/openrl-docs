接入用户自定义环境
=====================

.. toctree::
   :maxdepth: 1

由于强化学习环境多种多样，任何强化学习学习框架都不可能直接接入所有未知的环境。在本章节中，我们提供几种接入用户自定义环境的方法：

* :ref:`Gymnasium`
* :ref:`Gym`
* :ref:`PettingZoo`
* :ref:`MoreExamples1`
* :ref:`MoreExamples2`
* :ref:`MoreExamples3`


.. _Gymnasium:

接入符合Gymnasium接口的环境
-------------------

对于单智能体环境，我们推荐用户把自己的环境包装成符合 Gymnasium 接口的环境。用户可以参考 `官方文档 <https://gymnasium.farama.org/api/env/>`_ 来实现Gymnasium所需要的接口。
我们在 `这里 <https://github.com/OpenRL-Lab/openrl/blob/main/examples/custom_env/gymnasium_env.py>`_ 给出了一个简单的示例，用户可以参考该示例来实现自己的环境。

下面，我们给出该例子的详细解释。首先，我们需要创建一个名叫 ``gymnasium_env.py`` 的文件，并在里面创建自己的环境，该环境需要继承 ``gymnasium.Env`` 类。
在该类中，我们需要实现 ``reset``、``step``、``render``、``close``、``seed``、``action_space``、``observation_space`` 等方法:

.. code-block:: python

    # gymnasium_env.py
    from typing import Any, Dict, Optional

    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.envs.registration import EnvSpec
    from gymnasium.utils import seeding
    from gymnasium.envs.registration import register

    from openrl.envs.common import make

    class IdentityEnv(gym.Env):
        spec = EnvSpec("IdentityEnv")
        def __init__(self,**kwargs):
            self.dim = 2
            self.observation_space = spaces.Discrete(1)
            self.action_space = spaces.Discrete(self.dim)
            self.ep_length = 5
            self.current_step = 0
        def reset(
                self,
                *,
                seed: Optional[int] = None,
                options: Optional[Dict[str, Any]] = None,
        ):
            if seed is not None:
                self.seed(seed)
            self.current_step = 0
            self.generate_state()
            return self.state, {}
        def step(self, action) :
            reward = 1
            self.generate_state()
            self.current_step += 1
            done = self.current_step >= self.ep_length
            return self.state, reward, done, {}
        def generate_state(self) -> None:
            self.state = [self._np_random.integers(0, self.dim)]
        def render(self, mode: str = "human") -> None:
            pass
        def seed(self, seed: Optional[int] = None) -> None:
            if seed is not None:
                self._np_random, seed = seeding.np_random(seed)
        def close(self):
            pass

这样，我们就完成了一个符合 Gymnasium 接口的环境。接下来，我们需要通过 ``gymnasium.envs.registration.register`` 来注册我们的环境，这样我们就可以通过 ``make`` 来创建我们的环境并用于OpenRL框架的训练和测试。

.. code-block:: python

    # gymnasium_env.py
    from typing import Any, Dict, Optional

    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.envs.registration import EnvSpec
    from gymnasium.utils import seeding
    from gymnasium.envs.registration import register

    from openrl.envs.common import make

    from train_and_test import train_and_test

    class IdentityEnv(gym.Env):
        ... 同上 ...

    register(
         id="Custom_Env/IdentityEnv", # 这里填入自定义环境的名字，可以自己随意修改
         entry_point="gymnasium_env:IdentityEnv", # 这里填入文件名和自定义环境的类名
    )

    # 通过上面函数注册完自己的环境后，便可以通过make来创建环境了
    env = make(id = "Custom_Env/IdentityEnv", env_num = 10) # 这里id需要和上面注册的id一致，env_num表示创建10个环境

    # 得到的该env，便可以直接用于OpenRL框架的训练了！
    from openrl.modules.common.ppo_net import PPONet as Net
    from openrl.runners.common.ppo_agent import PPOAgent as Agent
    agent = Agent(Net(env)) # 直接传入该环境即可
    agent.train(5000) # 开始训练！

从上面的例子可以看出，只要用户实现了符合 Gymnasium 接口的环境，便可以轻松地接入并使用OpenRL。


.. _Gym:

接入符合OpenAI Gym接口的环境
-------------------

可能某些过去的强化学习环境是使用OpenAI Gym接口实现的，这些环境也可以接入OpenRL框架。
我们在 `这里 <https://github.com/OpenRL-Lab/openrl/blob/main/examples/custom_env/openai_gym_env.py>`_ 给出了一个简单的示例，展示了如何在OpenRL框架中使用自定义的OpenAI Gym接口的环境。

下面，我们给出该例子的详细解释。首先，我们需要创建一个名叫 ``openai_gym_env.py`` 的文件，并在里面创建自己的环境，该环境需要继承 ``gym.Env`` 类：

.. code-block:: python

    # openai_gym_env.py
    from typing import Any, Dict, Optional
    import gym
    from gym import spaces
    from gym.utils import seeding
    from gym.envs.registration import register
    from gym.envs.registration import EnvSpec

    from openrl.envs.common import make

    from train_and_test import train_and_test

    class IdentityEnv(gym.Env):
        spec = EnvSpec("IdentityEnv-v1")
        def __init__(self,**kwargs):
            self.dim = 2
            self.observation_space = spaces.Discrete(1)
            self.action_space = spaces.Discrete(self.dim)
            self.ep_length = 5
            self.current_step = 0

        def reset(
                self,
                *,
                seed: Optional[int] = None,
                options: Optional[Dict[str, Any]] = None,
        ):
            if seed is not None:
                self.seed(seed)
            self.current_step = 0
            self.generate_state()
            return self.state

        def step(self, action) :
            reward = 1
            self.generate_state()
            self.current_step += 1
            done = self.current_step >= self.ep_length
            return self.state, reward, done, {}

        def generate_state(self) -> None:
            self.state = [self._np_random.randint(0, self.dim-1)]
        def render(self, mode: str = "human") -> None:
            pass
        def seed(self, seed: Optional[int] = None) -> None:
            if seed is not None:
                self._np_random, seed = seeding.np_random(seed)

        def close(self):
            pass

可以看出，该环境完全通过 OpenAI Gym 的相关接口来实现。接下来，我们需要通过 ``gym.envs.registration.register`` 来注册我们的环境，这样我们就可以通过 ``make`` 来创建我们的环境并用于OpenRL框架的训练和测试：

.. code-block:: python

    # openai_gym_env.py
    from typing import Any, Dict, Optional
    import gym
    from gym import spaces
    from gym.utils import seeding
    from gym.envs.registration import register
    from gym.envs.registration import EnvSpec

    from openrl.envs.common import make

    class IdentityEnv(gym.Env):
        ... 同上 ...

    # 注册环境
    register(
         id="Custom_Env/IdentityEnv-v1",
         entry_point="openai_gym_env:IdentityEnv", # 这里填入文件名和自定义环境的类名
    )

    # 通过上面函数注册完自己的环境后，便可以通过make来创建环境了
    env = make(id = "GymV21Environment-v0:Custom_Env/IdentityEnv-v1",env_num=10) # 这里id需要加上GymV21Environment-v0前缀或者GymV26Environment-v0前缀，后面接上注册的id

    # 得到的该env，便可以直接用于OpenRL框架的训练了！
    from openrl.modules.common.ppo_net import PPONet as Net
    from openrl.runners.common.ppo_agent import PPOAgent as Agent
    agent = Agent(Net(env)) # 直接传入该环境即可
    agent.train(5000) # 开始训练！

从上面的例子可以看出，只要用户实现了符合 OpenAI Gym 接口的环境，便可以轻松地接入并使用OpenRL。



.. _PettingZoo:

接入符合PettingZoo接口的环境
-------------------

PettingZoo是一个多智能体环境库，用户可以通过PettingZoo来创建多智能体环境。我们在 `这里 <https://github.com/OpenRL-Lab/openrl/blob/main/examples/custom_env/pettingzoo_env.py>`_ 给出了一个简单的示例，用户可以参考该示例来实现自己的环境。

下面，我们给出该例子的详细解释。首先，我们需要创建一个名叫 ``rock_paper_scissors.py`` 的文件，并在里面创建自己的环境，该环境需要继承``pettingzoo.AECEnv`` 类。
由于PettingZoo环境定义比较复杂，我们直接在 `这里 <https://github.com/OpenRL-Lab/openrl/blob/main/examples/custom_env/rock_paper_scissors.py>`_ 给出我们一个自定义的环境，用户可以参考该环境来实现自己的环境：

.. code-block:: python

    # rock_paper_scissors.py
    from pettingzoo import AECEnv

    # 该环境具体实现请查看：https://github.com/OpenRL-Lab/openrl/blob/main/examples/custom_env/rock_paper_scissors.py
    class RockPaperScissors(AECEnv):
        def __init__(self, render_mode=None):
            ...

在上面的例子中，我们定义了一个名为RockPaperScissors的环境。接下来，我们需要在OpenRL中注册该环境，然后便可以通过 ``make`` 来创建我们的环境并用于OpenRL框架的训练和测试：

.. code-block:: python

    # pettingzoo_env.py
    from openrl.envs.common import make
    from openrl.envs.PettingZoo.registration import register
    from openrl.selfplay.wrappers.random_opponent_wrapper import RandomOpponentWrapper

    from rock_paper_scissors import RockPaperScissors

    # 注册环境
    register("RockPaperScissors",RockPaperScissors) # 这里填入自定义环境的名字和环境类
    # 通过上面函数注册完自己的环境后，便可以通过make来创建环境了
    env = make("RockPaperScissors",env_num=10, opponent_wrappers=[ RandomOpponentWrapper],)

    # 得到的该env，便可以直接用于OpenRL框架的训练了！
    from openrl.modules.common.ppo_net import PPONet as Net
    from openrl.runners.common.ppo_agent import PPOAgent as Agent
    agent = Agent(Net(env)) # 直接传入该环境即可
    agent.train(5000) # 开始训练！

从该例子可以看出，只要用户实现了符合PettingZoo接口的环境，便可以轻松地接入并使用OpenRL。


.. _MoreExamples1:

通过实现 ``make_custom_envs`` 来定制化创建环境
-------------------

对于更加复杂的环境，比如实现通过命令行或者YAML文件传递环境参数等功能，我们支持用户实现自己的环境创建函数，然后通过 ``make`` 来创建环境。
具体例子可以参考OpenRL中的 `星际环境的例子 <https://github.com/OpenRL-Lab/openrl/blob/main/examples/smac>`_ :

.. code-block:: python

    # train_ppo.py
    from openrl.configs.config import create_config_parser
    from smac_env import make_smac_envs

    cfg_parser = create_config_parser()
    cfg = cfg_parser.parse_args()
    env = make(
        "2s_vs_1sc",
        env_num=8,
        cfg=cfg,
        make_custom_envs=make_smac_envs,
    )

在这个例子中，我们自己实现了一个 ``make_smac_envs`` 函数，可以更加定制化地创建环境。用户可以参考该例子来实现自己的环境创建函数。

.. _MoreExamples2:

通过实现 ``make`` 来定制化创建环境
-------------------

通常，我们使用OpenRL框架自带的 ``make`` 函数来创建环境，``make`` 函数会帮我们自动生成多个并行(或者串行)的环境。
如果用户还有自身额外的需求，还可以参考 ``make`` 函数的实现来实现自己的 ``make`` 函数。具体例子可以参考OpenRL中的 `Retro环境的例子 <https://github.com/OpenRL-Lab/openrl/blob/main/examples/retro>`_ 。
在该环境中，我们在 `custom_registration.py <https://github.com/OpenRL-Lab/openrl/blob/main/examples/retro/custom_registration.py>`_ 中实现了自定义的 ``make`` 函数，并在创建环境时进行了调用。

.. _MoreExamples3:

通过自定义Wrapper来接入已有环境
-------------------

在有些环境中，由于更加底层的优化(例如，通过GPU进行模拟)，并不支持一个一个环境地创建然后通过Python的多进程进行并行运行，而是直接产生多个并行环境。
这类的环境通常无法通过 ``make`` 函数的形式接入OpenRL框架，但是我们可以通过自定义Wrapper来接入这类环境。

`Omniverse Isaac Gym <https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs>`_ 就是这样的一个环境，它通过GPU进行模拟，无法通过 ``make`` 函数的形式接入OpenRL框架。
我们在 `这里 <https://github.com/OpenRL-Lab/openrl/blob/main/examples/isaac>`_ 给出了一个示例，展示了如何在OpenRL框架中通过实现Wrapper接入Omniverse Isaac Gym环境。
我们在 `isaac2openrl.py <https://github.com/OpenRL-Lab/openrl/blob/main/examples/isaac/isaac2openrl.py>`_ 中实现了一个 ``Isaac2OpenRLWrapper`` 类，可以传入Omniverse Isaac Gym环境，然后返回一个可直接用于OpenRL框架训练的环境。

