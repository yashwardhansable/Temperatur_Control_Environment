from Environment import TemperatureWithDelay
from algorithms import TD3
from ReplayBuffer import ReplayBuffer
import numpy as np
import torch
import matplotlib.pyplot as plt
from StepChangeEnv import TemperatureWithDelayStepChange
import json


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.




# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    SP = 7.1
    current_level = 3.2
    max_action = 1.0
    expl_noise = 0.1
    action_dim = 3
    state_dim = 1
    batch_size = 256

    env = TemperatureWithDelay(7.1, 3.2)
    episode_rewards = []

    buffer_size = 512
    replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)

    state = env.reset(target_level=SP, current_level=current_level)
    TD3 = TD3(state_dim=state_dim, action_dim=action_dim, max_action=max_action)

    for episode in range(200):

        if episode > 0 and episode % 10 == 0:
            sp_range = list(np.round(np.linspace(0, 10, 21), 1))
            SP = np.float64(np.random.choice(sp_range, 1))
            current_height = np.float64(np.random.choice(sp_range, 1))
            state = env.reset(target_level=SP, current_level=current_level)

        else:
            state = env.reset(target_level=SP, current_level=current_level)

        initial_state = state
        Set_Point = SP

        episode_reward = 0

        for i in range(5000):
            action = (TD3.select_action(state=np.array(state)) + np.random.normal(0, max_action * expl_noise,
                                                                                  size=action_dim)).clip(-max_action,
                                                                                                         max_action)
            #print(action)
            next_state, reward, done = env.step(action)
            #print(next_state, reward, done)
            replay_buffer.add(state, action, next_state, reward, done)

            episode_reward += reward
            state = next_state
            if i > 500:
                #print("training...")
                TD3.train(replay_buffer, batch_size)
                #print(action)

            if done or episode_reward < -5000000:
                episode_rewards.append(episode_reward)
                break

        print(
            f"Episode {episode + 1}: Reward = {episode_reward},Initial_State = {initial_state}, Final_State = {state},Set Point = {Set_Point}")

    print(env.p, env.i, env.d)




    action_test = (env.p, env.i, env.d)
    target_level = 7.1
    current_level = 3.2

    with open('data01.json', 'w') as fjson:
        json.dump(action_test, fjson)

    TC = TemperatureWithDelayStepChange(target_level = target_level, current_level =current_level)

    for i in range(0,10000):
        TC.step(action=action_test)

    plt.figure()
    plt.plot(TC.t, TC.r, label='Reference (Target Temperature)')
    plt.plot(TC.t, TC.y, label='Temperature')
    plt.plot(TC.t, TC.u, label='Control Signal')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
