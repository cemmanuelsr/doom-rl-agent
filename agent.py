#!/usr/bin/env python3

import vizdoom as vzd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import itertools as it
import os
from time import sleep, time
from collections import deque
from tqdm import trange

from constants import *
from utils import *

# Configuration file path
# config_file_path = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")
config_file_path = os.path.join(vzd.scenarios_path, "simpler_basic.cfg")
# config_file_path = os.path.join(vzd.scenarios_path, "rocket_basic.cfg")
# config_file_path = os.path.join(vzd.scenarios_path, "basic.cfg")

DEVICE = torch.device('cpu')
global_train_scores = []

# Roda todas as épocas de treinamento
def run(game, agent, actions, num_epochs, frame_repeat, steps_per_epoch=2000):
    start_time = time()

    for epoch in range(num_epochs):
        game.new_episode()
        train_scores = []
        global_step = 0
        print("\nEpoch #" + str(epoch + 1))
        
        for _ in trange(steps_per_epoch, leave=False):

            if show_labels and visible_during_train:
                _state = game.get_state()
                if _state:
                    labels = _state.labels_buffer

                    if labels is not None:
                        cv2.imshow('ViZDoom Labels Buffer', color_labels(labels))

                    screen = _state.screen_buffer
                    for l in _state.labels:
                        if l.object_name in ["Medkit", "GreenArmor"]:
                            draw_box(screen, l.x, l.y, l.width, l.height, doom_blue_color)
                        else:
                            draw_box(screen, l.x, l.y, l.width, l.height, doom_red_color)
                    cv2.imshow('ViZDoom Screen Buffer', screen)
                    cv2.waitKey(28)

            state = preprocess(game.get_state().screen_buffer)
            if show_labels:
                state = np.reshape(state, (3, 30, 45))
            action = agent.get_action(state)
            reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()
                        
            if not done:
                next_state = preprocess(game.get_state().screen_buffer)
                if show_labels:
                    next_state = np.reshape(next_state, (3, 30, 45))
            else:
                if not show_labels:
                    next_state = np.zeros((1, 30, 45)).astype(np.float32)
                else:
                    next_state = np.zeros((3, 30, 45)).astype(np.float32)

            agent.append_memory(state, action, reward, next_state, done)

            if global_step > agent.batch_size:
                agent.train()

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()

            global_step += 1

        agent.update_target_net()
        train_scores = np.array(train_scores)
        global_train_scores.append(train_scores.mean())

        print("Resultados: media: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()),
              "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

        test(game, agent, actions)
        if save_model:
            torch.save(agent.q_net, f'./agents/{model_savefile}')

        print("Tempo total = %.2f minutos" % ((time() - start_time) / 60.0))

    game.close()
    if save_model:
        import matplotlib.pyplot as plt
        plt.plot(global_train_scores)
        plt.xlabel('Episodes')
        plt.ylabel('Train scores')
        plt.title('Train scores vs Episodes')
        plt.savefig('./results/results.jpg')     
        plt.close()
    return agent, game

# Usando uma rede neural para a estratégia de Double Deep Q-Learning
# Exemplo de rede da documentação do pytorch
class DoubleQNet(nn.Module):
    def __init__(self, available_actions_count):
        super(DoubleQNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3 if show_labels else 1, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.state_fc = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.advantage_fc = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, available_actions_count)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 192)
        x1 = x[:, :96]
        x2 = x[:, 96:]
        state_value = self.state_fc(x1).reshape(-1, 1)
        advantage_values = self.advantage_fc(x2)
        x = state_value + (advantage_values - advantage_values.mean(dim=1).reshape(-1, 1))

        return x

# Exemplo de agente
# Retirado da documentação do ViZDoom
class DQNAgent:
    def __init__(self, action_size, memory_size, batch_size, discount_factor, 
                 lr, load_model, epsilon=1, epsilon_decay=0.9996, epsilon_min=0.1):
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.discount = discount_factor
        self.lr = lr
        self.memory = deque(maxlen=memory_size)
        self.criterion = nn.MSELoss()

        if load_model:
            print("Loading model from: ", f'./agents/{model_savefile}')
            self.q_net = torch.load(f'./agents/{model_savefile}')
            self.target_net = torch.load(f'./agents/{model_savefile}')
            self.epsilon = self.epsilon_min

        else:
            print("Initializing new model")
            self.q_net = DoubleQNet(action_size).to(DEVICE)
            self.target_net = DoubleQNet(action_size).to(DEVICE)

        self.opt = optim.SGD(self.q_net.parameters(), lr=self.lr)

    def get_action(self, state):
        if np.random.uniform() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            state = np.expand_dims(state, axis=0)
            state = torch.from_numpy(state).float().to(DEVICE)
            action = torch.argmax(self.q_net(state)).item()
            return action

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        batch = random.sample(self.memory, self.batch_size)
        batch = np.array(batch, dtype=object)

        states = np.stack(batch[:, 0]).astype(float)
        actions = batch[:, 1].astype(int)
        rewards = batch[:, 2].astype(float)
        next_states = np.stack(batch[:, 3]).astype(float)
        dones = batch[:, 4].astype(bool)
        not_dones = ~dones

        row_idx = np.arange(self.batch_size)

        with torch.no_grad():
            next_states = torch.from_numpy(next_states).float().to(DEVICE)
            idx = row_idx, np.argmax(self.q_net(next_states).cpu().data.numpy(), 1)
            next_state_values = self.target_net(next_states).cpu().data.numpy()[idx]
            next_state_values = next_state_values[not_dones]

        q_targets = rewards.copy()
        q_targets[not_dones] += self.discount * next_state_values
        q_targets = torch.from_numpy(q_targets).float().to(DEVICE)

        idx = row_idx, actions
        states = torch.from_numpy(states).float().to(DEVICE)
        action_values = self.q_net(states)[idx].float().to(DEVICE)

        self.opt.zero_grad()
        td_error = self.criterion(q_targets, action_values)
        td_error.backward()
        self.opt.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min


if __name__ == '__main__':
    # Gerando jogo
    game = generate_game(config_file_path)
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Inicializando agente
    agent = DQNAgent(len(actions), lr=learning_rate, batch_size=batch_size,
                     memory_size=replay_memory_size, discount_factor=discount_factor,
                     load_model=load_model)

    # Treinando agente
    if not skip_learning:
        agent, game = run(game, agent, actions, num_epochs=train_epochs, frame_repeat=frame_repeat,
                          steps_per_epoch=learning_steps_per_epoch)

        print("Treinado!")

    # Reinicializando jogo mas agora visível
    game.close()
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            if show_labels:
                state = np.reshape(state, (3, 30, 45))
            best_action_index = agent.get_action(state)

            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep entre episódios para não fritar a CPU
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)
