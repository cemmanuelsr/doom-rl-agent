# Q-learning settings
learning_rate = 0.0001
discount_factor = 0.99
train_epochs = 20
learning_steps_per_epoch = 2000
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 5

# Other parameters
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 5
model_savefile = "doom-agent-20-epochs.pth"
save_model = True
load_model = False
skip_learning = False
visible_during_train = False
doom_red_color = [0, 0, 203]
doom_blue_color = [203, 0, 0]
show_labels = False
