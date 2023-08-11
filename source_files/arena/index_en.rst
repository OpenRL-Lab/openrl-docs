Arena
=====================

.. toctree::
   :maxdepth: 1

OpenRL provides an arena framework for competitive environments, where both intelligent agents trained through self-play by OpenRL and rule-based agents can compete and be evaluated in the arena.
Example code for the arena can be found `here <https://github.com/OpenRL-Lab/openrl/tree/main/examples/arena>`_.

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