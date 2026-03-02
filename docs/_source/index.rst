AI Hydra: Hybrid Neural Network + Tree Search System
====================================================

Overview
--------

AI Hydra is a distributed application that includes a Textual TUI client, a 
simple router, and a headless server. Communication is over ZeroMQ. The basic
functionality is that the Client sends control messages (start, stop, reset)
to the router which forwards them to the server. When the server is running
it publishes board and other telemetry information a ZeroMQ PUB/SUB socket.
The client subscribes to the server's PUB socket and displays the game state.

The headless server uses a neural network and a policy stack to play the game.
The policy stack includes an implementation of the traditional Epsilon-Greedy
algorithm to encourage exploration at the beginning of a simulation run. 

The server also includes a *lookahead* policy where the server checks to see if
the move suggested by the neural network will result in a collision. If so,
then it "looks ahead" at alternative moves. If a *look-ahead* move results
in finding food, then that move is selected. If a move does **not** result in
a collision, then that move is selected.

The *look-ahead* policy is only enabled for a configurable probability of the
time. The training of the neural network includes the moves and game state
information that was executed. So when the *look-ahead* policy is used the
training data is enhanced. This leads to better neural network performance.

Installation
------------

.. code-block:: shell

   $ python3 -m venv hydra-venv
   $ . hydra-venv/bin/activate
   hydra-venv> pip install ai-hydra

Distributed Architecture
------------------------

The *HydraClient*, *HydraRouter*, and *HydraMgr* are run in three different
terminals. The project supports running the client, router, and server on
different machines, but at this early stage in the project, it's recommended
that all three be run on the same machine.

Startup
-------

Start the *HydraClient* in the first terminal:

.. code-block:: shell

    $ . hydra-venv/bin/activate
    hydra-venv> ai-hydra-client

Start the *HydraRouter* in a second terminal:

.. code-block:: shell

    $ . hydra-venv/bin/activate
    hydra-venv> ai-hydra-router

Finally, start the *HydraMgr* in a third terminal:

.. code-block:: shell
    
    $ . hydra-venv/bin/activate
    hydra-venv> ai-hydra-mgr


Click the `Start` button in the *HydraRouter* to start the routing functions.

Click the `Start` button in the *HydraClient*. This causes a `START_FUN` 
ZeroMQ message to be sent through the *HydraRouter* to the *HydraMgr*. The
*HydraMgr* continues to listen for `STOP` or `RESET` messages. The *HydraMgr*
starts the simulation, and publishes game telemetry information on a ZeroMQ
**PUB** socket. The *HydraClient* connects directly to the *HydraMgr* and subscribes 
to the appropriate topics. The *HydraClient* displays the game state.