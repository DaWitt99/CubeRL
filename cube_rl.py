from cube import Cube
from collections import deque
import tensorflow as tf
import numpy as np

class DenseBN(tf.keras.layers.Layer):
    def __init__(self,units=32, activation='elu', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.dense = tf.keras.layers.Dense(units=self.units, kernel_initializer="he_normal")
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.bn(x)
        x= self.activation(x)
        return x

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'units': self.units, 'activation': tf.keras.activations.serialize(self.activation)}

def sample_experiences(batch_size=64):
    indices = np.random.randint(low=0, high=len(experiences_buffer)-1, size=batch_size)
    exps = [experiences_buffer[i] for i in indices]
    return [np.array([experience[field_index] for experience in exps]) for field_index in range(6)]

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.random() < epsilon:
        return np.random.randint(0,13)
    else:
        Q_vals = cube_solver.predict(state)
        return np.argmax(Q_vals)

def play_one_step(epsilon, cube, counter=-1):
    state = cube.get_state()
    init_layers_done = cube.count_layers_done()
    init_entropy = cube.get_cube_entropy()

    if counter != -1:
        action = counter
    else:
        action = epsilon_greedy_policy(state,epsilon)

    next_state, cube_entropy, move_counter, done, truncated = cube.make_move(action)
    layers_done = cube.count_layers_done()

    if counter != -1:
        reward = 5
    else:
        reward = (layers_done - init_layers_done) + (init_entropy - cube_entropy) * 10.0 + done * 100 - truncated * 100

    experiences_buffer.append([state, action, reward, next_state, done, truncated])
    return next_state, reward, done, truncated, cube_entropy, action

def training_step(batch_size=32):
    exps = sample_experiences(batch_size)
    states, actions, rewards, next_states = exps

    next_Q_values = []
    for next in next_states:
        next_Q_values.append(cube_solver.predict(np.array(next), verbose=0))
    max_next_Q_values = np.max(next_Q_values, axis=2)
    target_Q_values = rewards.reshape(batch_size,1) + discount * max_next_Q_values
    target_Q_values = target_Q_values.reshape(-1,1)
    mask = tf.one_hot(actions, 14)

    with tf.GradientTape() as tape:
        all_Q_values = [cube_solver(state) for state in states]
        Q_values = tf.reduce_max(tf.reduce_sum(all_Q_values*mask, axis=2, keepdims=True),axis=1)
        loss = tf.reduce_mean(tf.keras.losses.mse(target_Q_values, Q_values))

    gradients = tape.gradient(loss, cube_solver.trainable_variables)
    optimizer.apply_gradients(zip(gradients, cube_solver.trainable_variables))

def training_loop(episodes):
    for episode in range(episodes):
        cube = Cube(3,-1)

        for step in range(100):
            epsilon = max(1-episode/50,0.1)
            next_state,reward,done,truncated,entropy,action = play_one_step(epsilon, cube)
            print("EPISODE", episode, "STEP", step, 'REWARD', reward, 'DONE', done, "ENTROPY", entropy)

            if done or truncated:
                break

        if episode > 20:
            training_step()
            cube_solver.save('cubeRL.keras')

# train going backward on cube's moves history made of random moves
def pre_training_loop(episodes):
    for episode in range(episodes):
        cube = Cube(3,-1)
        cube.make_random_moves(episode)
        history = cube.get_history()
        counter_history =[]

        for i in range(len(history)):
            if history[i] % 2 == 0:
                counter_history.append(history[i]+1)
            else:
                counter_history.append(history[i]-1)

        for i in range(len(history)):
            next_state, reward, done, truncated, entropy, action = play_one_step(0, cube, counter_history.pop())
            print("EPISODE", episode, "STEP", i, 'REWARD', reward, 'DONE', done, "ENTROPY", entropy)

        if episode > 20:
            training_step()
            cube_solver.save('cubeRL.keras')

cube_solver = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    DenseBN(units=4096, activation='relu'),
    DenseBN(units=2048, activation='relu'),
    DenseBN(units=2048, activation='relu'),
    DenseBN(units=1024, activation='relu'),
    tf.keras.layers.Dense(14),
])

experiences_buffer = deque(maxlen=4000)
discount = 0.99
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

#cube_solver=tf.keras.models.load_model('cubeRL.keras', custom_objects={'DenseBN':DenseBN})
pre_training_loop(1000)
training_loop(1000)