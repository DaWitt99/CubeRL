from cube import Cube
from cube_archs import DenseBN, ResidualBlock, make_model
from collections import deque
import tensorflow as tf
import numpy as np
from datetime import datetime

DEFAULT_GOAL = Cube(3,0).get_state_one_hot()

def sample_experiences(memory, batch_size=64, positive_learnings=8):
    dones = np.array([exp[2] for exp in memory])
    positive_indices = np.argwhere(dones>=0)
    size = min(len(positive_indices), positive_learnings)
    won_indices = []

    if size > 0:
        won_indices = np.random.choice(np.squeeze(positive_indices), size=size, replace=False)

    indices = np.concat((np.random.randint(low=0, high=len(memory)-1, size=batch_size-size),won_indices))
    exps = [memory[int(i)] for i in indices]
    return [np.array([experience[field_index] for experience in exps]) for field_index in range(7)]

def epsilon_greedy_policy(model, state,goal, epsilon=0):
    if np.random.random() < epsilon:
        return np.random.randint(0,18)
    else:
        app = np.append(state,goal)
        app = np.append(app,goal - state)
        Q_vals = model.predict(app.reshape(1,-1))
        return np.argmax(Q_vals)

def play_one_step(model, epsilon, cube, memory):
    state = cube.get_state_one_hot()
    action = epsilon_greedy_policy(model,state, DEFAULT_GOAL,epsilon)

    next_state, cube_entropy, move_counter, done, truncated = cube.make_move(action)
    layers_done = cube.get_layers_done()
    walls_done = cube.get_walls_done()

    layer_reward = 0
    for i in range(cube.ds):
        if layers_done[1][i] == 1 and layers_done[0][i] == 0:
            layer_reward += 5

    wall_reward = 0
    for i in range(6):
        if walls_done[1][i] == 1 and walls_done[0][i] == 0:
            wall_reward += 5

    no_repeats_rewards = 0
    for past_state in cube.get_state_history()[:-1]:
        if np.array_equal(state,past_state):
            no_repeats_rewards -=1

    reward = no_repeats_rewards + layer_reward + wall_reward + done * 20 - 1

    memory.append([state, action, reward, next_state, done, truncated, DEFAULT_GOAL])
    return state,next_state, reward, done, truncated, cube_entropy, action

def training_step(model,target_model, discount, optimizer, memory, batch_size=64):
    exps = sample_experiences(memory, batch_size)
    states, actions, rewards, next_states, dones, truncateds, goals = exps
    napp = np.append(next_states,goals,axis=2)
    napp = np.append(napp,goals - next_states,axis=2)
    next_Q_values = target_model.predict(np.squeeze(napp), verbose=0)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = rewards + discount * max_next_Q_values
    target_Q_values = target_Q_values.reshape(-1,1)
    mask = tf.one_hot(actions, 18)

    with tf.GradientTape() as tape:
        qapp = np.append(states,goals,2)
        qapp = np.append(qapp,goals - states,2)
        all_Q_values = model(np.squeeze(qapp))
        Q_values = tf.reduce_sum(all_Q_values*mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(tf.keras.losses.mse(target_Q_values, Q_values))
        print(loss)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def training_loop(model, target, episodes, discount, optimizer, memory, date):
    for episode in range(episodes):
        cube = Cube(3,3)
        episodes_memories = []
        for step in range(40):
            epsilon = max(1-episode/episodes,0.1)
            state,next_state,reward,done,truncated,entropy,action = play_one_step(model, epsilon, cube, memory)
            print("EPISODE", episode, "STEP", step, 'REWARD', reward, 'DONE', done, "ENTROPY", entropy)

            episodes_memories.append([state,action,20,next_state,1,truncated, next_state])
            if done:
                break
            if truncated:
                for epm in episodes_memories:
                    memory.append(epm)
                break

        if episode % 100 == 0:
            target.set_weights(model.get_weights())

        if episode > 10:
            training_step(model, target, discount, optimizer, memory)
            model.save('cubeRL_'+date.strftime("%d-%m-%Y_%H-%M-%S")+'.keras')

def testing_loop(model, episodes, memory):
    positive = 0
    for episode in range(episodes):
        cube = Cube(3,1)
        for step in range(40):
            state,next_state, reward, done, truncated, entropy, action = play_one_step(model, 0, cube, memory)
            print("TEST EPISODE", episode, "STEP", step, 'REWARD', reward, 'DONE', done, "ENTROPY", entropy)
            if done:
                positive+=1
                break
            if truncated:
                break
    return positive

mode = "load"

if mode == "load":
    model = tf.keras.models.load_model('cubeRL_15-05-2026_13-34-52.keras',
                                       custom_objects={'ResidualBlock': ResidualBlock, "DenseBN": DenseBN})
    target = tf.keras.models.load_model('cubeRL_15-05-2026_13-34-52.keras',
                                        custom_objects={'ResidualBlock': ResidualBlock, "DenseBN": DenseBN})
else:
    model_type = "Residual_4"
    model = make_model(model_type)
    target = make_model(model_type)

experiences_buffer = deque(maxlen=100000)
discount = 0.98
optimizer = tf.keras.optimizers.Adam(learning_rate=0.000025, clipnorm=0.1)

model.compile(loss="mse", optimizer=optimizer)
target.compile(loss="mse", optimizer=optimizer)
target.set_weights(model.get_weights())
now = datetime.now()

training_loop(model, target, 1000, discount, optimizer, experiences_buffer, now)
#rint("TESTING---------------")
val =testing_loop(model, 1000, experiences_buffer)
print(val)