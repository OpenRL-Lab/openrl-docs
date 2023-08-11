竞技场
=====================

.. toctree::
   :maxdepth: 1

OpenRL为具有竞争性的环境提供了一套竞技场框架，通过OpenRL自博弈训练后的智能体以及基于规则的智能体均可以在竞技场中进行对战和评测。
用于可以在 `这里 <https://github.com/OpenRL-Lab/openrl/tree/main/examples/arena>`_ 找到竞技场的示例代码。

通过竞技场进行智能体评测
-------------------

当用于完成了智能体的自博弈训练，并保存到到了本地后，我们可以通过竞技场来对智能体进行评测。
假设我们将智能体保存到了 ``./agent_trained`` 目录下，另外有个随机智能体在 ``./random_agent`` 目录下，我们可以通过以下代码来对这两个智能体进行评测：

.. code-block:: python

    from openrl.arena import make_arena
    from openrl.arena.agents.local_agent import LocalAgent
    from openrl.envs.wrappers.pettingzoo_wrappers import RecordWinner

    # 创建竞技场，并让环境记录胜利者
    env_wrappers = [RecordWinner]
    arena = make_arena("tictactoe_v3", env_wrappers=env_wrappers)
    # 从本地文件加载训练好的智能体
    agent1 = LocalAgent("./agent_trained")
    # 从本地文件加载随机智能体
    agent2 = LocalAgent("./random_agent")
    # 初始化竞技场，设置一共运行100局对战，同一时刻最多同时运行10局对战，随机种子为0
    arena.reset(
        agents={"agent1": agent1, "agent2": agent2},
        total_games=100,
        max_game_onetime=10,
        seed=0,
    )
    # 运行竞技场，设置为并行运行（用户也可以设置parallel=False，这时候每局对战将按照顺序一局一局来进行，主要用户进行方便调试）
    result = arena.run(parallel=True)
    arena.close()
    # 输出对局统计结果
    print(result)



在该示例中，我们使用了 `PettingZoo <https://pettingzoo.farama.org/index.html>`_ 中的 `tictactoe_v3 <https://pettingzoo.farama.org/environments/classic/tictactoe/>`_ 环境作为我们的竞技场环境。
我们通过 ``env_wrappers = [RecordWinner]`` 来让环境记录胜利者，这样我们就不断统计对局的胜负信息。
接着我们通过 ``arena.reset`` 来初始化竞技场，设置了一共运行100局对战，同一时刻最多同时运行10局对战，随机种子为0。
然后我们通过 ``arena.run`` 来运行竞技场，设置为并行运行（用户也可以设置 ``parallel=False``，这时候每局对战将按照顺序一局一局来进行，主要用户进行方便调试）。
最后我们可以打印出对局统计结果。

关于竞技场中智能体的写法，可以参考我们给的 `示例 <https://github.com/OpenRL-Lab/openrl/tree/main/examples/arena>`_ 。
