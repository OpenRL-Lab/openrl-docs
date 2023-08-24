Integrate User-defined Environments
=====================

.. toctree::
   :maxdepth: 1

Given the diversity of reinforcement learning environments, no reinforcement learning framework can directly interface with all unknown environments.
In this section, we offer several methods for connecting custom user-defined environments:

* Integrate an environment compliant with the Gymnasium interface (:ref:`Gymnasium`)
* Integrate an environment compliant with the OpenAI Gym interface (:ref:`Gym`)
* Integrate an environment compliant with the PettingZoo interface (:ref:`PettingZoo`)
* Customize environment creation by implementing ``make_custom_envs`` (:ref:`MoreExamples1`)
* Customize environment creation by implementing ``make`` (:ref:`MoreExamples2`)
* Integrate existing environments through custom Wrappers (:ref:`MoreExamples3`)

.. _Gymnasium:

Integrate an Environment Compliant with the Gymnasium Interface
-------------------

For single-agent environments, we recommend users wrap their environments to be compliant with the Gymnasium interface.
Users can refer to the `official documentation <https://gymnasium.farama.org/api/env/>`_ to implement the interfaces required by Gymnasium.
We provide a simple example `here <https://github.com/OpenRL-Lab/openrl/blob/main/examples/custom_env/gymnasium_env.py>`_.
Users can refer to this example to implement their own environments.

Below, we provide a detailed explanation of this example.
Firstly, create a file named ``gymnasium_env.py`` and create your environment inside it.
This environment should inherit from the ``gymnasium.Env`` class.
Within this class, methods like ``reset``, ``step``, ``render``, ``close``, ``seed``, ``action_space``, and ``observation_space`` need to be implemented:

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

With this, we have completed an environment compliant with the Gymnasium interface.
Next, we need to register our environment using ``gymnasium.envs.registration.register``.
This allows us to create our environment with ``make`` and use it for training and testing in the OpenRL framework.

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
        ... [Continuation of the class as before] ...

    register(
         id="Custom_Env/IdentityEnv", # Fill in the name of the custom environment, it can be freely modified
         entry_point="gymnasium_env:IdentityEnv", # Fill in the filename and class name of the custom environment
    )

    # After registering your environment using the above function, you can create the environment using make
    env = make(id = "Custom_Env/IdentityEnv", env_num = 10) # The id here should match the registered id, env_num means creating 10 environments

    # The obtained environment can then be directly used for training in the OpenRL framework!
    from openrl.modules.common.ppo_net import PPONet as Net
    from openrl.runners.common.ppo_agent import PPOAgent as Agent
    agent = Agent(Net(env)) # Simply pass in the environment
    agent.train(5000) # Start training!

From the above example, it can be seen that once users implement an environment compliant with the Gymnasium interface, they can easily connect and use OpenRL.

.. _Gym:

Integrate Environments Compliant with OpenAI Gym Interface
-------------------

Some past reinforcement learning environments may have been implemented using the OpenAI Gym interface.
These environments can also be integrated into the OpenRL framework.
We provide a simple example `here <https://github.com/OpenRL-Lab/openrl/blob/main/examples/custom_env/openai_gym_env.py>`_ that demonstrates how to use a custom OpenAI Gym environment in the OpenRL framework.

Following is a detailed explanation of the example.
First, we need to create a file called ``openai_gym_env.py`` and define our environment within, inheriting from the ``gym.Env`` class:

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


You can see that the environment is fully implemented through the OpenAI Gym interface.
Next, we need to register our environment through ``gym.envs.registration.register`` so that we can create our environment via ``make`` for training and testing in the OpenRL framework:


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
        # ... [code as in the original] ...

    # Register the environment
    register(
         id="Custom_Env/IdentityEnv-v1",
         entry_point="openai_gym_env:IdentityEnv", # Fill in the filename and the class name of the custom environment here
    )

    # After registering your environment with the above function, you can use 'make' to create the environment
    env = make(id = "GymV21Environment-v0:Custom_Env/IdentityEnv-v1",env_num=10) # The id here needs to be prefixed with either "GymV21Environment-v0" or "GymV26Environment-v0", followed by the registered id

    # The obtained env can now be used directly for training in the OpenRL framework!
    from openrl.modules.common.ppo_net import PPONet as Net
    from openrl.runners.common.ppo_agent import PPOAgent as Agent
    agent = Agent(Net(env)) # Just pass in this environment
    agent.train(5000) # Start training!


From the example above, it's clear that as long as users implement environments compliant with the OpenAI Gym interface, they can easily integrate and use them in OpenRL.

.. _PettingZoo:

Integrate Environments Compliant with PettingZoo Interface
-------------------

PettingZoo is a multi-agent environment library, and users can create multi-agent environments through PettingZoo.
We provide a simple example `here <https://github.com/OpenRL-Lab/openrl/blob/main/examples/custom_env/pettingzoo_env.py>`_ for users to reference in implementing their own environments.

Below, we provide a detailed explanation of this example. First, we need to create a file named ``rock_paper_scissors.py`` and create our own environment within it.
This environment should inherit from the ``pettingzoo.AECEnv`` class.
Given the complexity of defining environments in PettingZoo, we directly provide our custom environment `here <https://github.com/OpenRL-Lab/openrl/blob/main/examples/custom_env/rock_paper_scissors.py>`_.
Users can refer to this environment to design their own:

.. code-block:: python

    # rock_paper_scissors.py
    from pettingzoo import AECEnv

    # For the detailed implementation of this environment, please see: https://github.com/OpenRL-Lab/openrl/blob/main/examples/custom_env/rock_paper_scissors.py
    class RockPaperScissors(AECEnv):
        def __init__(self, render_mode=None):
            ...

In the example above, we defined an environment named RockPaperScissors.
Next, we need to register this environment in OpenRL, and then we can use ``make`` to instantiate our environment and use it for training and testing within the OpenRL framework:


.. code-block:: python

    # pettingzoo_env.py
    from openrl.envs.common import make
    from openrl.envs.PettingZoo.registration import register
    from openrl.selfplay.wrappers.random_opponent_wrapper import RandomOpponentWrapper

    from rock_paper_scissors import RockPaperScissors

    # Registering the environment
    register("RockPaperScissors",RockPaperScissors) # Input the custom environment's name and class here
    # After registering your environment using the above function, you can instantiate it using make
    env = make("RockPaperScissors",env_num=10, opponent_wrappers=[ RandomOpponentWrapper],)

    # The obtained env can be used directly for training in the OpenRL framework!
    from openrl.modules.common.ppo_net import PPONet as Net
    from openrl.runners.common.ppo_agent import PPOAgent as Agent
    agent = Agent(Net(env)) # Simply pass in the environment
    agent.train(5000) # Start training!

From this example, it's evident that once users have developed an environment compliant with the PettingZoo interface, it can be effortlessly integrated and utilized with OpenRL.

.. _MoreExamples1:

Customize Environment Creation through ``make_custom_envs``
-------------------

For more complex environments, such as passing environment parameters through command-line or YAML files, we allow users to implement their own environment creation functions and then use ``make`` to create environments.
A specific example can be found in OpenRL's `SMAC environment example <https://github.com/OpenRL-Lab/openrl/blob/main/examples/smac>`_:

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

In this example, we implemented our own ``make_smac_envs`` function, allowing for a more customized environment creation.
Users can refer to this example to implement their own environment creation function.


.. _MoreExamples2:

Customize Environment Creation with ``make``
-------------------

Typically, we use the built-in ``make`` function from the OpenRL framework to create environments.
This ``make`` function automatically generates multiple parallel (or sequential) environments for us.
If users have additional requirements, they can also implement their own ``make`` function, drawing inspiration from the implementation of the existing ``make``.
A specific example can be seen in OpenRL's `Retro environment example <https://github.com/OpenRL-Lab/openrl/blob/main/examples/retro>`_.
In this environment, we implemented a custom ``make`` function in `custom_registration.py <https://github.com/OpenRL-Lab/openrl/blob/main/examples/retro/custom_registration.py>`_ and invoked it when creating the environment.

.. _MoreExamples3:

Integrate Existing Environments through Custom Wrappers
-------------------

In some environments, due to more fundamental optimizations (e.g., simulations via GPU), it's not feasible to create individual environments and then run them in parallel using Python's multiprocessing.
Instead, they directly generate multiple parallel environments.
Environments of this kind usually cannot be integrated into the OpenRL framework through the ``make`` function.
However, they can be integrated using custom wrappers.

`Omniverse Isaac Gym <https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs>`_ is one such environment.
It runs simulations via the GPU and cannot be integrated into the OpenRL framework through the ``make`` function.
We provided an example `here <https://github.com/OpenRL-Lab/openrl/blob/main/examples/isaac>`_ that demonstrates how to integrate the Omniverse Isaac Gym environment into the OpenRL framework by implementing a wrapper.
In `isaac2openrl.py <https://github.com/OpenRL-Lab/openrl/blob/main/examples/isaac/isaac2openrl.py>`_, we implemented an ``Isaac2OpenRLWrapper`` class.
This class accepts the Omniverse Isaac Gym environment and returns an environment ready for training in the OpenRL framework.