Callbacks
=====================

.. toctree::
   :maxdepth: 1

Callbacks are a set of functions that can be called at specific stages during training.
Users can access the internal state of the agent during training through callbacks.
It allows users to monitor the training process, automatically save models, customize stopping conditions and more.
In OpenRL, we have built-in some commonly used callbacks, and users can also customize their own callbacks according to their needs (:ref:`Customize and Register Callback <CustomCallback>`).
In OpenRL, all callbacks can be configured using **YAML files**. The usage of all callbacks can be found in `examples/cartpole <https://github.com/OpenRL-Lab/openrl/tree/main/examples/cartpole>`_ .

Built-in Callbacks
-------------------

Here are some commonly used callbacks provided by OpenRL, which can be combined for use:

- Callback for periodically saving the model (:ref:`CheckpointCallback`)
- Callback for periodically evaluating the model performance and saving the best model (:ref:`EvalCallback`)
- Callback for stopping training when a specified reward threshold is reached (:ref:`StopTrainingOnRewardThreshold <StopTrainingOnRewardThreshold>`)
- Callback for stopping training when there is no longer any improvement in model performance (:ref:`StopTrainingOnNoModelImprovement <StopTrainingOnNoModelImprovement>`)
- Callback for periodically performing tasks (:ref:`EveryNTimesteps`)
- Callback for displaying a progress bar during training (:ref:`ProgressBarCallback`)
- Callback to stop training after a specified number of environment runs have been completed (:ref:`StopTrainingOnMaxEpisodes`)

.. _CheckpointCallback:

CheckpointCallback
^^^^^^^^^^^^^^^^^^

``CheckpointCallback`` is a callback that saves the model every ``save_freq`` steps.
In addition, users need to specify a model save path ``save_path``, and can also specify the prefix of the saved model name through ``name_prefix`` (by default it is set to "rl_model").
Here's an example of using ``CheckpointCallback`` with a YAML file:

.. code-block:: yaml

    callbacks:
      - id: "CheckpointCallback"
      args: {
          "save_freq": 500, # how often to save the model
          "save_path": "./results/checkpoints/",  # where to save the model
          "name_prefix": "ppo", # the prefix of the saved model
      }


.. _EvalCallback:

EvalCallback
^^^^^^^^^^^^

``EvalCallback`` is a callback used to periodically evaluate the performance of a model.
It evaluates the agent's performance in an independent testing environment.
Users can specify the path to save the best model by setting ``best_model_save_path``, and specify the log saving path by setting ``log_path``.
Here is an example of using ``EvalCallback`` through a YAML file:

.. code-block:: yaml

    callbacks:
      - id: "EvalCallback"
      args: {
          "eval_env": { "id": "CartPole-v1","env_num": 5 }, # how many envs to set up for evaluation
          "n_eval_episodes": 5, # how many episodes to run for each evaluation
          "eval_freq": 500, # how often to run evaluation
          "log_path": "./results/eval_log_path", # where to save the evaluation results
          "best_model_save_path": "./results/best_model/", # where to save the best model
          "deterministic": True, # whether to use deterministic action
          "render": False, # whether to render the env
          "asynchronous": True, # whether to run evaluation asynchronously
      }

.. _StopTrainingOnRewardThreshold:

StopTrainingOnRewardThreshold
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When users want the model to automatically stop training when it reaches a certain level of performance, they can use this callback.
Users can set the ``reward_threshold`` to specify the reward threshold. When the reward reaches this threshold after model evaluation, training will automatically stop.
It is worth noting that ``StopTrainingOnRewardThreshold`` needs to be used together with ``EvalCallback``, as it needs to obtain the reward value through evaluation.
Here is an example of using ``StopTrainingOnRewardThreshold`` through a YAML file:

.. code-block:: yaml

    callbacks:
      - id: "EvalCallback"
      args: {
          "eval_env": { "id": "CartPole-v1","env_num": 5 }, # how many envs to set up for evaluation
          "n_eval_episodes": 5, # how many episodes to run for each evaluation
          "eval_freq": 500, # how often to run evaluation
          "log_path": "./results/eval_log_path", # where to save the evaluation results
          "best_model_save_path": "./results/best_model/", # where to save the best model
          "deterministic": True, # whether to use deterministic action
          "render": False, # whether to render the env
          "asynchronous": True, # whether to run evaluation asynchronously
          "stop_logic": "OR", # the logic to stop training, OR means training stops when any one of the conditions is met, AND means training stops when all conditions are met
          "callbacks_on_new_best": [
            {
              id: "StopTrainingOnRewardThreshold",
              args: {
                "reward_threshold": 100, # the reward threshold to stop training
              }
            } ],
      }

.. _StopTrainingOnNoModelImprovement:

StopTrainingOnNoModelImprovement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When users want to stop training when the model performance no longer improves, they can use this callback.
The training will automatically stop when the model has not improved in ``max_no_improvement_evals`` evaluations.
The ``min_evals`` parameter specifies how many evaluations need to be performed before determining whether to stop training or not.
Here is an example of using ``StopTrainingOnNoModelImprovement`` through a YAML file:

.. code-block:: yaml

    callbacks:
      - id: "EvalCallback"
      args: {
          "eval_env": { "id": "CartPole-v1","env_num": 5 }, # how many envs to set up for evaluation
          "n_eval_episodes": 5, # how many episodes to run for each evaluation
          "eval_freq": 500, # how often to run evaluation
          "log_path": "./results/eval_log_path", # where to save the evaluation results
          "best_model_save_path": "./results/best_model/", # where to save the best model
          "deterministic": True, # whether to use deterministic action
          "render": False, # whether to render the env
          "asynchronous": True, # whether to run evaluation asynchronously
          "stop_logic": "OR", # the logic to stop training, OR means training stops when any one of the conditions is met, AND means training stops when all conditions are met
          "callbacks_after_eval": [
            {
              id: "StopTrainingOnNoModelImprovement",
              args: {
                "max_no_improvement_evals": 5, # Maximum number of consecutive evaluations without a new best model.
                "min_evals": 3, # Number of evaluations before start to count evaluations without improvements.
              }
            },
          ],
      }

.. _EveryNTimesteps:

EveryNTimesteps
^^^^^^^^^^^^^^^

This callback is used to trigger other callbacks every ``n_steps`` steps.
Users only need to specify the callbacks that need to be triggered through the ``callbacks`` parameter.
For example, users can save the model every ``n_steps`` steps.
You can also customize other callbacks (:ref:`Customize and Register Callback <CustomCallback>`) to meet your own needs and register them yourself.
Here is an example of using ``EveryNTimesteps`` in a YAML file to periodically save models:

.. code-block:: yaml

    callbacks:
      - id: "EveryNTimesteps" # This is same to "CheckpointCallback"
      args: {
          "n_steps": 5000,
          "callbacks":[
            {
              "id": "CheckpointCallback",
              args: {
                "save_freq": 1,
                "save_path": "./results/checkpoints_with_EveryNTimesteps/",  # where to save the model
                "name_prefix": "ppo", # the prefix of the saved model
              }
            }
          ]
      }



.. _ProgressBarCallback:

ProgressBarCallback
^^^^^^^^^^^^^^^^^^^

This callback is used to display the training progress bar, which can show the current progress, elapsed time and estimated remaining time.
Here is an example of using ``ProgressBarCallback`` through a YAML file:

.. code-block:: yaml

    callbacks:
      - id: "ProgressBarCallback"


.. _StopTrainingOnMaxEpisodes:

StopTrainingOnMaxEpisodes
^^^^^^^^^^^^^^^^^^^^^^^^^

When using this callback, the ``total_time_steps`` setting of ``agent.train`` can be ignored.
Training will stop when each environment has run for the specified number of ``max_episodes``.
Here is an example of using ``StopTrainingOnMaxEpisodes`` through a YAML file:

.. code-block:: yaml

    callbacks:
      - id: "StopTrainingOnMaxEpisodes"
      args: {
          "max_episodes": 5, # the max number of episodes to run
      }


.. _CustomCallback:

Customize and Register Callback
-------------------------------

The callbacks mentioned above can be combined for use.
For example, if a user wants to use both ``ProgressBarCallback`` and ``StopTrainingOnMaxEpisodes`` at the same time, it can be achieved through the following method:

.. code-block:: yaml

    callbacks:
      - id: "ProgressBarCallback"
      - id: "StopTrainingOnMaxEpisodes"
        args: {
          "max_episodes": 5, # the max number of episodes to run
        }

In addition, users can implement their own callbacks by inheriting from ``BaseCallback`` or ``EventCallback``,
and the specific implementation method can refer to the implementation of other callbacks: `callbacks examples <https://github.com/OpenRL-Lab/openrl/tree/main/openrl/utils/callbacks>`_.

After implementing their own callback, users need to register it through ``CallbackFactory``, and then they can use their own callback through YAML files:

.. code-block:: python

    from openrl.utils.callbacks import CallbackFactory
    from openrl.utils.callbacks.callbacks import BaseCallback

    # custom callback
    class MyCustomCallback(BaseCallback):
        def _on_step(self) -> bool:
            print("Number of calls: ",self.n_calls)
            return True

    # register custom callback
    CallbackFactory.register("MyCustomCallback", MyCustomCallback)

Then your can call it in the YAML file:

.. code-block:: yaml

    callbacks:
      - id: "MyCustomCallback"