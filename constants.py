# Q-learning settings
learning_rate = 0.0001
discount_factor = 0.99
train_epochs = 5
learning_steps_per_epoch = 2000
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 10
model_savefile = "./doom-agent.pth"
save_model = False
load_model = True
skip_learning = True
visible_during_train = False
doom_red_color = [0, 0, 203]
doom_blue_color = [203, 0, 0]
show_labels = True
