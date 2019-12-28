import pickle
import time

import numpy as np

from log import plot_save, print_save, print_clear


class Training:

    def __init__(self, model, data_loader, batch_size=10, epochs=20, model_ckpt_file='model/PCamNet'):
        self.model = model
        self.data_loader = data_loader  # List of tuple: (left_cam, right_cam, disp_map) filenames
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.num_batch = len(self.data_loader)
        self.model_ckpt_file = model_ckpt_file
        self.train_stats = []

    def train_model(self):
        print_clear()
        print_save('Training Model: %s ... ' % self.model.get_name())

        Validation_loss_values = []
        Training_loss_values = []
        iters = []

        # train model for one epoch - call fn model.train_batch(data, label) for each batch
        for epoch in range(self.num_epochs):
            training_loss = 0.0
            training_count = 0

            validation_loss = 0.0
            validation_count = 0
            validation_predict = []

            true_labels = []
            t1 = time.time()
            for batch_data, batch_labels in self.data_loader:

                if training_count <= 0.8 * self.num_batch:
                    training_loss += self.model.train_batch(batch_data, batch_labels)
                    training_count += 1
                else:
                    validation_loss += self.model.train_batch(batch_data, batch_labels)
                    validation_count += 1

            t2 = time.time()
            self.train_stats.append([epoch, training_loss.numpy(), training_count,
                                     validation_loss.numpy(), validation_count,
                                     validation_predict, true_labels, t2 - t1])

            print_save('====================================')

            print_save('epoch: %4d    train loss: %20.6f     val loss: %20.6f' %
                       (epoch, training_loss / training_count,
                        validation_loss / validation_count))

            Validation_loss_values.append(float(validation_loss / validation_count))
            Training_loss_values.append(float(training_loss / training_count))
            iters.append(epoch)

            print_save('epoch time:' + str(np.round(t2 - t1, 2)) + 's')
            print_save('time for completion:' + str(np.round((t2 - t1) * (self.num_epochs - epoch - 1) / 60, 2)) + 'm')
            print_save('')

            self.model.save_model(self.model_ckpt_file + '.pth')

            pickle.dump(self.train_stats, open(self.model_ckpt_file + '_stats.pkl', 'wb'))

        print_save('Training Model: %s ... Complete' % self.model.get_name())
        plot_save(Training_loss_values, Validation_loss_values, iters)
        return 0

    def get_model(self):
        return self.model

    def set_model_save(self, filename):
        self.model_ckpt_file = filename
