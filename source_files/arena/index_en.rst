Arena
=====================

.. toctree::
   :maxdepth: 1

OpenRL provides an arena framework for competitive environments, where both intelligent agents trained through self-play by OpenRL and rule-based agents can compete and be evaluated in the arena.
Example code for the arena can be found `here <https://github.com/OpenRL-Lab/openrl/tree/main/examples/arena>`_.
OpenRL even supports local evaluation of agents submitted in JiDi platform format.
We provide examples for local evaluation on the JiDi platform for the `Snake game <https://github.com/OpenRL-Lab/openrl/tree/main/examples/snake#evaluate-jidi-submissions-locally>`_ and
`Google Research Football game <https://github.com/OpenRL-Lab/TiZero#evaluate-jidi-submissions-locally>`_.

Evaluating Agents Through the Arena
-------------------

Once the agent's self-play training is completed and saved locally, we can evaluate them through the arena.
Assuming we have saved the agents to the ``./agent_trained`` directory, and there's a random agent in the ``./random_agent`` directory, we can evaluate these two agents with the following code:

.. code-block:: python

    from openrl.arena import make_arena
    from openrl.arena.agents.local_agent import LocalAgent
    from openrl.envs.wrappers.pettingzoo_wrappers import RecordWinner

    # Create the arena and have the environment record the winner
    env_wrappers = [RecordWinner]
    arena = make_arena("tictactoe_v3", env_wrappers=env_wrappers)
    # Load the trained agent from a local file
    agent1 = LocalAgent("./agent_trained")
    # Load the random agent from a local file
    agent2 = LocalAgent("./random_agent")
    # Initialize the arena, setting a total of 100 games to be played, with a maximum of 10 games simultaneously running at a time, and a random seed of 0
    arena.reset(
        agents={"agent1": agent1, "agent2": agent2},
        total_games=100,
        max_game_onetime=10,
        seed=0,
    )
    # Run the arena, set to parallel execution (users can also set parallel=False, in which case each game will be played one by one, mainly for convenient debugging)
    result = arena.run(parallel=True)
    arena.close()
    # Print the game statistics result
    print(result)

In this example, we used the `tictactoe_v3 <https://pettingzoo.farama.org/environments/classic/tictactoe/>`_ environment from `PettingZoo <https://pettingzoo.farama.org/index.html>`_ as our arena environment. We used ``env_wrappers = [RecordWinner]`` to let the environment record the winner, thus continuously keeping track of the game's win-loss information. Next, we initialized the arena through ``arena.reset``, setting a total of 100 games to be played, with a maximum of 10 games simultaneously running at a time, and a random seed of 0. Then we ran the arena through ``arena.run``, setting parallel execution (users can also set ``parallel=False``, in which case each game will be played one by one, mainly for convenient debugging). Finally, we can print out the game statistics result.

For the writing of agents in the arena, you can refer to the examples we provided `here <https://github.com/OpenRL-Lab/openrl/tree/main/examples/arena>`_.

Performing Local Evaluation of Agents Submitted to the JiDi Platform Using OpenRL
-------------------------------------------------------------------------------

OpenRL supports local evaluation of agents submitted to the `JiDi Platform <http://www.jidiai.cn/>`_.
After users have trained their own agents and crafted their submission code, they can directly perform local evaluations within the OpenRL framework!
We provide examples for local evaluation on the JiDi platform for the `Snake game <https://github.com/OpenRL-Lab/openrl/tree/main/examples/snake#evaluate-jidi-submissions-locally>`_
and the `Google Research Football game <https://github.com/OpenRL-Lab/TiZero#evaluate-jidi-submissions-locally>`_.

For instance, in the case of the Snake game, the following code snippet allows for local evaluation of agents on the JiDi platform:

.. code-block:: python

    env_wrappers = [RecordWinner]
    player_num = 3
    arena = make_arena(
        f"snakes_{player_num}v{player_num}", env_wrappers=env_wrappers, render=render
    )

    # Locally load the agents that meet the JiDi platform submission criteria
    agent1 = JiDiAgent("./submissions/rule_v1", player_num=player_num)
    agent2 = JiDiAgent("./submissions/rl", player_num=player_num)

    arena.reset(
        agents={"agent1": agent1, "agent2": agent2},
        total_games=10,
        max_game_onetime=5,
        seed=0,
    )
    result = arena.run(parallel=True)
    arena.close()
    print(result)

In this example, we use ``JiDiAgent`` to load agents that meet the JiDi platform submission standards. The rest of the usage remains the same as before.
We load one rule-based agent and one agent trained through reinforcement learning to compete against each other and tally the win-loss records.
The complete code and submitted agents for this example can be found `here <https://github.com/OpenRL-Lab/openrl/tree/main/examples/snake>`_.

Additionally, for local evaluation of the Google Research Football game, we have implemented a simple and easy-to-use local evaluation tool in the `TiZero <https://github.com/OpenRL-Lab/TiZero>`_ project.
After installing TiZero via ``pip install tizero``, you can execute the following command to perform local evaluation of agents on the JiDi platform:

.. code-block:: bash

    tizero eval --left_agent submission_dir1 --right_agent submission_dir2 --total_game 10

Here, ``submission_dir1`` and ``submission_dir2`` are the directories for two agents that meet JiDi platform submission criteria, and ``total_game`` is the total number of games to be played.
We provide our trained TiZero agents `here <https://github.com/OpenRL-Lab/TiZero#evaluate-jidi-submissions-locally>`_, and users can directly use this command for the local evaluation.