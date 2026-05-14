from cube import Cube
from collections import deque
import tensorflow as tf
import numpy as np

class DenseBN(tf.keras.layers.Layer):
    def __init__(self,units=64, activation='elu', **kwargs):
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

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, units = 128, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.layers = [
            tf.keras.layers.Dense(units=self.units, kernel_initializer="he_normal"),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            tf.keras.layers.Dense(units=self.units, kernel_initializer="he_normal"),
            tf.keras.layers.BatchNormalization(),
        ]
        self.skip_layers = [
            tf.keras.layers.Dense(units=self.units, kernel_initializer="he_normal"),
            tf.keras.layers.BatchNormalization(),
        ]

    def call(self, inputs):
        Y = inputs
        for layer in self.layers:
            Y=layer(Y)

        Y_skip = inputs
        for skip_layer in self.skip_layers:
            Y_skip = skip_layer(Y_skip)

        return self.activation(Y+Y_skip)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'units': self.units, 'activation': tf.keras.activations.serialize(self.activation)}

def sample_experiences(batch_size=64):
    indices = np.random.randint(low=0, high=len(experiences_buffer)-1, size=batch_size)
    exps = [experiences_buffer[i] for i in indices]
    return [np.array([experience[field_index] for experience in exps]) for field_index in range(6)]

def epsilon_greedy_policy(model, state, epsilon=0):
    if np.random.random() < epsilon:
        return np.random.randint(0,13)
    else:
        Q_vals = model.predict(state)
        return np.argmax(Q_vals)

def play_one_step(model, epsilon, cube):
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

    reward = no_repeats_rewards + layer_reward + done * 1200 - 1

    experiences_buffer.append([state, action, reward, next_state, done, truncated])
    return next_state, reward, done, truncated, cube_entropy, action

def training_step(model, batch_size=32):
    exps = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones, truncateds = exps
    next_Q_values = []
    for next in next_states:
        next_Q_values.append(model.predict(next, verbose=0))
    max_next_Q_values = np.max(next_Q_values, axis=2)
    target_Q_values = rewards.reshape(batch_size,1) + discount * max_next_Q_values
    target_Q_values = target_Q_values.reshape(-1,1)
    mask = tf.one_hot(actions, 18)

    with tf.GradientTape() as tape:
        all_Q_values = [model(state) for state in states]
        Q_values = tf.reduce_max(tf.reduce_sum(all_Q_values*mask, axis=2, keepdims=True),axis=1)
        loss = tf.reduce_mean(tf.keras.losses.mse(target_Q_values, Q_values))
        print(loss)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def training_loop(model, episodes):
    for episode in range(episodes):
        cube = Cube(3,1)

        for step in range(200):
            epsilon = max(1-episode/200,0.1)
            next_state,reward,done,truncated,entropy,action = play_one_step(model,epsilon, cube)
            print("EPISODE", episode, "STEP", step, 'REWARD', reward, 'DONE', done, "ENTROPY", entropy)

            if done or truncated:
                break

        if episode > -1:
            training_step(model)
            cube_solver.save('cubeRL.keras')

def testing_loop(model, episodes):
    for episode in range(episodes):
        cube = Cube(3,1)
        for step in range(200):
            next_state, reward, done, truncated, entropy, action = play_one_step(model, 0, cube)
            print("TEST EPISODE", episode, "STEP", step, 'REWARD', reward, 'DONE', done, "ENTROPY", entropy)

cube_solver = tf.keras.Sequential([
    #tf.keras.layers.Flatten(),
    DenseBN(units=4096, activation='elu'),
    DenseBN(units=2048, activation='elu'),
    DenseBN(units=2048, activation='elu'),
    DenseBN(units=1024, activation='elu'),
    tf.keras.layers.Dense(14),
])

experiences_buffer = deque(maxlen=4000)
discount = 0.95
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

residual_solver = tf.keras.Sequential([
    ResidualBlock(4096,"elu"),
    ResidualBlock(2048,"elu"),
    ResidualBlock(2048,"elu"),
    ResidualBlock(1024,"elu"),
    tf.keras.layers.Dense(18)
])

#cube_solver=tf.keras.models.load_model('cubeRL.keras', custom_objects={'DenseBN':DenseBN})
training_loop(residual_solver, 1000)
print("TESTING---------------")
testing_loop(residual_solver, 1000)