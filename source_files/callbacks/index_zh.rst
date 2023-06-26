Callbacks
=====================

.. toctree::
   :maxdepth: 1

callback是一组用于在训练时候的特定阶段调用的函数。用户可以通过callback来访问智能体训练时候的内部状态。
它允许用户对训练进行监控，自动保存模型，定制化停止训练条件等等。
在OpenRL中，我们内置了一些常用的callback，用户也可以根据自身需求来自定义callback (:ref:`自定义并注册callback <CustomCallback>`)。
在OpenRL中，所有的callback都可以通过 **YAML文件** 来进行配置。
所有callback的使用方法都可以在 `examples/cartpole <https://github.com/OpenRL-Lab/openrl/tree/main/examples/cartpole>`_ 中找到。

内置Callback
-------------------

以下是OpenRL提供的一些常用的callback，以下这些callback可以组合进行使用:

- 用于定期保存模型的callback (:ref:`CheckpointCallback`)
- 用于定期评估模型性能并保存最好模型的callback (:ref:`EvalCallback`)
- 用于达到指定奖励阈值便停止训练的callback (:ref:`StopTrainingOnRewardThreshold <StopTrainingOnRewardThreshold>`)
- 用于当模型性能不再有提升便停止训练的callback (:ref:`StopTrainingOnNoModelImprovement <StopTrainingOnNoModelImprovement>`)
- 用于定期执行任务的callback (:ref:`EveryNTimesteps`)
- 用于显示训练进度条的callback (:ref:`ProgressBarCallback`)
- 用于达到指定环境运行轮数后便停止训练的callback (:ref:`StopTrainingOnMaxEpisodes`)

.. _CheckpointCallback:

CheckpointCallback
^^^^^^^^^^^^^^^^^^

``CheckpointCallback`` 是一个每 ``save_freq`` 步保存一次模型的callback。
此外，用户需要指定一个模型保存路径 ``save_path``，还可以通过 ``name_prefix`` 指定模型保存名称的前缀（默认情况下为 ``rl_model`` ）。
以下是通过YAML文件使用 ``CheckpointCallback`` 的一个例子:

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

``EvalCallback`` 是一个用于定期评估模型性能的callback，它会在一个独立的测试环境中评估智能体的性能。
用户可以通过设置 ``best_model_save_path`` 来指定最好模型的保存路径，通过 ``log_path`` 来指定日志保存路径。
以下是通过YAML文件使用 ``EvalCallback`` 的一个例子:

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

当用户希望我们的模型训练到一定性能变自动停止，便可以使用这个callback。
用户可以设置 ``reward_threshold`` 来指定奖励阈值，当进行模型评测后奖励达到这个阈值时，训练便会自动停止。
值得注意的是 ``StopTrainingOnRewardThreshold`` 需要搭配 ``EvalCallback`` 来一起使用，因为它需要通过评估来获取奖励值。
以下是通过YAML文件使用 ``StopTrainingOnRewardThreshold`` 的一个例子:

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

当用户希望当模型性能不再有提升的时候便停止训练，便可以使用这个callback。
当模型在 ``max_no_improvement_evals`` 次评估中都没有提升时，训练便会自动停止。
而 ``min_evals`` 参数用于指定至少需要进行多少次评估才能开始判断是否停止训练。
以下是通过YAML文件使用 ``StopTrainingOnNoModelImprovement`` 的一个例子:

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

该callback用于在每隔 ``n_steps`` 步时触发其他callback。
用户只需要通过 ``callbacks`` 参数来指定需要触发的callback即可。
例如，用户可以每隔 ``n_steps`` 步来保存模型。
用于也可以自定义其他callback (:ref:`自定义并注册callback <CustomCallback>`) 来实现自己的需求。
以下是通过YAML文件使用 ``EveryNTimesteps`` 来定期保存模型的一个例子:

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

该callback用于显示训练进度条，可以展示当前进度、已用时间和预估剩余时间。
以下是通过YAML文件使用 ``ProgressBarCallback`` 的一个例子:

.. code-block:: yaml

    callbacks:
      - id: "ProgressBarCallback"


.. _StopTrainingOnMaxEpisodes:

StopTrainingOnMaxEpisodes
^^^^^^^^^^^^^^^^^^^^^^^^^

使用该callback的时候，可以无视 ``agent.train`` 的 ``total_time_steps`` 设置，当每个环境运行达到指定的 ``max_episodes`` 轮，便会停止训练。
以下是通过YAML文件使用 ``StopTrainingOnMaxEpisodes`` 的一个例子:

.. code-block:: yaml

    callbacks:
      - id: "StopTrainingOnMaxEpisodes"
      args: {
          "max_episodes": 5, # the max number of episodes to run
      }


.. _CustomCallback:

自定义并注册callback
-------------------

以上介绍的这些callback可以组合进行使用。
例如，当用户想同时使用 ``ProgressBarCallback`` 和  ``StopTrainingOnMaxEpisodes`` 时，可以通过以下方式来实现:

.. code-block:: yaml

    callbacks:
      - id: "ProgressBarCallback"
      - id: "StopTrainingOnMaxEpisodes"
        args: {
          "max_episodes": 5, # the max number of episodes to run
        }

此外，用户可以通过继承 ``BaseCallback`` 或者 继承 ``EventCallback`` 来实现自己的callback，具体实现方式可以参考其他callback的实现：`callbacks实现示例 <https://github.com/OpenRL-Lab/openrl/tree/main/openrl/utils/callbacks>`_ 。

当用户实现自己的callback后，需要通过 ``CallbackFactory`` 来进行注册，然后用户便可以通过YAML文件来使用自己的callback了：

.. code-block:: python

    from openrl.utils.callbacks import CallbackFactory
    from openrl.utils.callbacks.callbacks import BaseCallback

    # 自定义callback
    class MyCustomCallback(BaseCallback):
        def _on_step(self) -> bool:
            print("Number of calls: ",self.n_calls)
            return True

    # 注册自定义的callback
    CallbackFactory.register("MyCustomCallback", MyCustomCallback)

然后在yaml文件里面进行调用即可：

.. code-block:: yaml

    callbacks:
      - id: "MyCustomCallback"