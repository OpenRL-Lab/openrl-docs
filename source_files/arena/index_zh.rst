竞技场
=====================

.. toctree::
   :maxdepth: 1

OpenRL为具有竞争性的环境提供了一套竞技场框架，通过OpenRL自博弈训练后的智能体以及基于规则的智能体均可以在竞技场中进行对战和评测。
用于可以在 `这里 <https://github.com/OpenRL-Lab/openrl/tree/main/examples/arena>`_ 找到竞技场的示例代码。
OpenRL甚至还支持及第平台提交格式智能体的本地评测，我们提供了本地评测及第平台上 `贪吃蛇游戏 <https://github.com/OpenRL-Lab/openrl/tree/main/examples/snake#evaluate-jidi-submissions-locally>`_
和 `谷歌足球游戏 <https://github.com/OpenRL-Lab/TiZero#evaluate-jidi-submissions-locally>`_ 的例子。

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


使用OpenRL对及第平台提交的智能体进行本地评测
--------------------------------------

OpenRL支持对 `及第平台 <http://www.jidiai.cn/>`_ 提交的智能体进行本地评测，用户训练完自己的智能体并构造自己的提交代码后，可以直接在OpenRL框架中对其进行本地评测！
我们提供了本地评测及第平台上 `贪吃蛇游戏 <https://github.com/OpenRL-Lab/openrl/tree/main/examples/snake#evaluate-jidi-submissions-locally>`_
和 `谷歌足球游戏 <https://github.com/OpenRL-Lab/TiZero#evaluate-jidi-submissions-locally>`_ 的例子。

以贪吃蛇游戏为例，我们可以通过以下代码来对及第平台上的智能体进行本地评测：


.. code-block:: python

    env_wrappers = [RecordWinner]
    player_num = 3
    arena = make_arena(
        f"snakes_{player_num}v{player_num}", env_wrappers=env_wrappers, render=render
    )
    # 从本地文件加载符合及第平台提交规范的智能体
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

在这个例子中，我们通过 ``JiDiAgent`` 来加载符合及第平台提交规范的智能体。然后后续的使用方式和之前的一样。
在该例子中，我们加载了一个基于规则的智能体和一个基于强化学习训练的智能体，让他们进行对战并统计胜负信息。
该例子的完整代码和提交智能体可以在 `这里 <https://github.com/OpenRL-Lab/openrl/tree/main/examples/snake>`_ 找到。

此外，对于谷歌足球游戏的本地评测，我们在 `TiZero <https://github.com/OpenRL-Lab/TiZero>`_ 项目中实现了一个简单易用的本地评测工具。
用户在通过 ``pip install tizero`` 安装好TiZero后，可以通过以下命令来对及第平台上的智能体进行本地评测：

.. code-block:: bash

    tizero eval --left_agent submission_dir1 --right_agent submission_dir2 --total_game 10

其中 ``submission_dir1`` 和 ``submission_dir2`` 分别是两个符合及第平台提交规范的智能体的目录， ``total_game`` 是总共进行的对局数。
我们在 `这里 <https://github.com/OpenRL-Lab/TiZero#evaluate-jidi-submissions-locally>`_ 中提供了我们训练的TiZero智能体，用户可以直接使用该命令来对其进行本地评测。