from cube import Cube
from cube_archs import DenseBN, ResidualBlock, make_model
from collections import deque
import tensorflow as tf
import numpy as np

def sample_experiences(memory, batch_size=32, win_learnings=8):
    dones = np.array([exp[4] for exp in memory])
    dones_indices = np.argwhere(dones==True)
    size = min(len(dones_indices), win_learnings)
    won_indices = []

    if size > 0:
        won_indices = np.random.choice(np.squeeze(dones_indices), size=size, replace=False)

    indices = np.concat((np.random.randint(low=0, high=len(memory)-1, size=batch_size-size),won_indices))
    exps = [memory[int(i)] for i in indices]
    return [np.array([experience[field_index] for experience in exps]) for field_index in range(6)]

def epsilon_greedy_policy(model, state, epsilon=0):
    if np.random.random() < epsilon:
        return np.random.randint(0,18)
    else:
        Q_vals = model.predict(state)
        return np.argmax(Q_vals)

def play_one_step(model, epsilon, cube, memory):
    state = cube.get_state_one_hot()
    action = epsilon_greedy_policy(model,state,epsilon)

    next_state, cube_entropy, move_counter, done, truncated = cube.make_move(action)
    layers_done = cube.get_layers_done()

    layer_reward = 0
    for i in range(cube.ds):
        if layers_done[1][i] == 1 and layers_done[0][i] == 0:
            layer_reward += 5

    no_repeats_rewards = 0
    for past_state in cube.get_state_history()[:-1]:
        if np.array_equal(state,past_state):
            no_repeats_rewards -=1

    reward = no_repeats_rewards + layer_reward + done * 20 - 1

    memory.append([state, action, reward, next_state, done, truncated])
    return next_state, reward, done, truncated, cube_entropy, action

def training_step(model,target_model, discount, optimizer, memory, batch_size=32):
    exps = sample_experiences(memory, batch_size)
    states, actions, rewards, next_states, dones, truncateds = exps
    next_Q_values = target_model.predict(np.squeeze(next_states), verbose=0)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = rewards + discount * max_next_Q_values
    target_Q_values = target_Q_values.reshape(-1,1)
    mask = tf.one_hot(actions, 18)

    with tf.GradientTape() as tape:
        all_Q_values = model(np.squeeze(states))
        Q_values = tf.reduce_sum(all_Q_values*mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(tf.keras.losses.mse(target_Q_values, Q_values))
        print(loss)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def training_loop(model, target, episodes, discount, optimizer, memory):
    for episode in range(episodes):
        cube = Cube(3,1)

        for step in range(40):
            epsilon = max(1-episode/episodes,0.1)
            next_state,reward,done,truncated,entropy,action = play_one_step(model,epsilon, cube, memory)
            print("EPISODE", episode, "STEP", step, 'REWARD', reward, 'DONE', done, "ENTROPY", entropy)

            if done or truncated:
                break

        if episode % 100 == 0:
            target.set_weights(model.get_weights())

        if episode > 1000:
            training_step(model, target, discount, optimizer, memory)
            cube_solver.save('cubeRL.keras')

def testing_loop(model, episodes):
    for episode in range(episodes):
        cube = Cube(3,1)
        for step in range(40):
            next_state, reward, done, truncated, entropy, action = play_one_step(model, 0, cube)
            print("TEST EPISODE", episode, "STEP", step, 'REWARD', reward, 'DONE', done, "ENTROPY", entropy)

mode = "load"

if mode == "load":
    model = tf.keras.models.load_model('cubeRL.keras',
                                       custom_objects={'ResidualBlock': ResidualBlock, "DenseBN": DenseBN})
    target = tf.keras.models.load_model('cubeRL.keras',
                                        custom_objects={'ResidualBlock': ResidualBlock, "DenseBN": DenseBN})
else:
    model_type = "Dense_4"
    model = make_model(model_type)
    target = make_model(model_type)

experiences_buffer = deque(maxlen=100000)
discount = 0.95
optimizer = tf.keras.optimizers.Adam(learning_rate=0.000025, clipnorm=0.1)

model.compile(loss="mse", optimizer=optimizer)
target.compile(loss="mse", optimizer=optimizer)
target.set_weights(model.get_weights())

training_loop(model, target, 10000, discount, optimizer, experiences_buffer)
print("TESTING---------------")
testing_loop(model, 1000)