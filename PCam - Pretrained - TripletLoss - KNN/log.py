import matplotlib.pyplot as plt


def plot_save(Training_loss_values, Validation_loss_values, iters):

    plt.figure()
    plt.plot(iters, Validation_loss_values, label="Validation loss")
    plt.plot(iters, Training_loss_values, label="Training loss")
    plt.xlabel("Iterations")
    plt.ylabel("Training and validation loss")
    plt.legend()
    plt.savefig('log/Training and validation losses.png')
    plt.show()

def print_clear():
    f = open("log/log.txt", "w+")
    f.close()

def print_save(txt):
    print(txt)
    f = open("log/log.txt", "a+")
    f.write(txt+"\n")
    f.close()

