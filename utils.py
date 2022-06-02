import vizdoom as vzd
import cv2
import numpy as np
import skimage.transform
from tqdm import trange

from constants import *

# Muda resolução dos frames para o escolhido
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

# Cria um novo jogo
def generate_game(config_file_path):
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(visible_during_train)
    game.set_mode(vzd.Mode.PLAYER)
#    if show_labels:
#        game.set_screen_format(vzd.ScreenFormat.BGR24)
#        game.set_labels_buffer_enabled(True)
#        game.clear_available_game_variables()
#        game.add_available_game_variable(vzd.GameVariable.POSITION_X)
#        game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
#        game.add_available_game_variable(vzd.GameVariable.POSITION_Z)
#    else:
#        game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")

    return game

# Testa agente sobre um episódio de teste apenas
def test(game, agent, actions):
    print("\nTesting...")
    test_scores = []
    for test_episode in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state)

            game.make_action(actions[best_action_index], frame_repeat)
        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    print("Resultados: media: %.1f +/- %.1f," % (
        test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
          "max: %.1f" % test_scores.max())

# Desenha caixa de bordas
def draw_box(buffer, x, y, width, height, color):
    for i in range(width):
        buffer[y, x + i, :] = color
        buffer[y + height, x + i, :] = color

    for i in range(height):
        buffer[y + i, x, :] = color
        buffer[y + i, x + width, :] = color

# Colore paredes de azul e chão de vermelho
def color_labels(labels):
    tmp = np.stack([labels] * 3, -1)
    tmp[labels == 0] = [255, 0, 0]
    tmp[labels == 1] = [0, 0, 255]

    return tmp
