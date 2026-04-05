episodes = 10
rewards = []
moving_average = []

"""
Here we are experimenting with our implementation:
- we play a certain number of episodes of the game
- for deciding each move to play at each step, we will apply our MCTS algorithm
- we will collect and plot the rewards to check if the MCTS is actually working.
- For CartPole-v0, in particular, 200 is the maximum possible reward. 
"""

for e in range(episodes):

    reward_e = 0
    game = gym.make(GAME_NAME)
    observation = game.reset()
    done = False

    new_game = deepcopy(game)
    mytree = Node(new_game, False, 0, observation, 0)

    print("episode #" + str(e + 1))

    while not done:

        mytree, action = Policy_Player_MCTS(mytree)

        observation, reward, done, _ = game.step(action)

        reward_e = reward_e + reward

        # game.render() # uncomment this if you want to see your agent in action!

        if done:
            print("reward_e " + str(reward_e))
            game.close()
            break

    rewards.append(reward_e)
    moving_average.append(np.mean(rewards[-100:]))

plt.plot(rewards)
plt.plot(moving_average)
plt.show()
print("moving average: " + str(np.mean(rewards[-20:])))
