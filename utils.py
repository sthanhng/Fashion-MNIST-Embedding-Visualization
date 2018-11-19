import matplotlib.pyplot as plt
import numpy as np


# ===================================================================
# Plot the training loss and accuracy
# ===================================================================
def plot_loss_acc(model, epochs, save_path):
    plt.style.use('ggplot')
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), model.history['loss'], label='train_loss')
    plt.plot(np.arange(0, N), model.history['acc'], label='train_acc')
    plt.title('Training loss and accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend(loc='upper left')
    plt.savefig(save_path)
