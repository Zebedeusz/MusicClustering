from keras.layers import Input, Dense, Conv2D, UpSampling2D, MaxPooling2D, Flatten, Reshape, Dropout, Activation
from keras.models import Model, Sequential
from keras import backend as K
from keras.regularizers import l1,l2,l1_l2
from keras import metrics
from keras.optimizers import Adagrad, rmsprop, Adam, Nadam, Adadelta, SGD
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

import numpy

class ConvAutoEncoder():

    numpy.random.seed(42)
    autoencoder = Sequential()
    decoder = Sequential()
    w_act_reg = 0.00001
    w_kernel_reg = 0.00001
    w_bias_reg = 0.00001

    def fbeta_score(y_true, y_pred, beta=1):

        def precision(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        def recall(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        if beta < 0:
            raise ValueError('The lowest choosable beta is zero (only precision).')

        if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
            return 0

        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        bb = beta ** 2
        fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())

        return fbeta_score

    def initialiseNet(self, rows, cols, print_layers_size, normalize_batch, optimizer, num_features, loss_function):

        self.optimizer_str = optimizer

        if optimizer == "Adam":
            optimizer = Adam
        elif optimizer == "rmsprop":
            optimizer = rmsprop
        elif optimizer == "Nadam":
            optimizer = Nadam
        elif optimizer == "Adadelta":
            optimizer = Adadelta
        elif optimizer == "sgd":
            optimizer = SGD

        print("Building network")
        self.autoencoder = Sequential()
        self.decoder = Sequential()

        self.decoder.add(Conv2D(1, 3, padding='same',
                                    activity_regularizer=l1_l2(self.w_act_reg, self.w_act_reg),
                                    kernel_regularizer=l2(self.w_kernel_reg), bias_regularizer=l2(self.w_bias_reg),
                                    input_shape=(rows, cols, 1)))
        self.decoder.add(LeakyReLU(alpha=0.3))
        if normalize_batch: self.decoder.add(BatchNormalization())
        if print_layers_size:
            print("1st conv in: {}".format(self.decoder.get_input_shape_at(0)))
            print("1st conv out: {}".format(self.decoder.get_output_shape_at(0)))
        self.decoder.add(MaxPooling2D((2, 2), padding='same'))
        if print_layers_size: print("1st max pool out: {}".format(self.decoder.get_output_shape_at(0)))
        self.decoder.add(Conv2D(1, 3,
                                    activity_regularizer=l1_l2(self.w_act_reg, self.w_act_reg),
                                    kernel_regularizer=l2(self.w_kernel_reg), bias_regularizer=l2(self.w_bias_reg),
                                    padding='same'))
        self.decoder.add(LeakyReLU(alpha=0.3))
        if normalize_batch:self.decoder.add(BatchNormalization())
        if print_layers_size:print("2nd conv out: {}".format(self.decoder.get_output_shape_at(0)))
        self.decoder.add(MaxPooling2D((2, 2), padding='same'))
        if print_layers_size:print("2nd max pool out: {}".format(self.decoder.get_output_shape_at(0)))



        self.decoder.add(Flatten())
        if print_layers_size: print("flatten out: {}".format(self.decoder.get_output_shape_at(0)))
        self.decoder.add(Dense(400,
                                   activity_regularizer=l1_l2(self.w_act_reg, self.w_act_reg),
                                   kernel_regularizer=l2(self.w_kernel_reg), bias_regularizer=l2(self.w_bias_reg)))
        self.decoder.add(LeakyReLU(alpha=0.3))
        if normalize_batch: self.decoder.add(BatchNormalization())
        if print_layers_size: print("dense 1 out: {}".format(self.decoder.get_output_shape_at(0)))
        #self.decoder.add(Dropout(0.3))
        self.decoder.add(Dense(num_features,
                                   activity_regularizer=l1_l2(self.w_act_reg, self.w_act_reg),
                                   kernel_regularizer=l2(self.w_kernel_reg), bias_regularizer=l2(self.w_bias_reg)))
        self.decoder.add(LeakyReLU(alpha=0.3))
        if normalize_batch: self.decoder.add(BatchNormalization())
        if print_layers_size: print("dense 2 out: {}".format(self.decoder.get_output_shape_at(0)))

        self.autoencoder.add(self.decoder)
        self.autoencoder.add(Dense(400,
                                   activity_regularizer=l1_l2(self.w_act_reg, self.w_act_reg),
                                   kernel_regularizer=l2(self.w_kernel_reg), bias_regularizer=l2(self.w_bias_reg)))
        self.autoencoder.add(LeakyReLU(alpha=0.3))
        if normalize_batch: self.autoencoder.add(BatchNormalization())
        if print_layers_size: print("dense 4 out: {}".format(self.autoencoder.get_output_shape_at(0)))
        self.autoencoder.add(Dense(625,
                                   activity_regularizer=l1_l2(self.w_act_reg, self.w_act_reg),
                                   kernel_regularizer=l2(self.w_kernel_reg), bias_regularizer=l2(self.w_bias_reg)))
        self.autoencoder.add(LeakyReLU(alpha=0.3))
        if normalize_batch: self.autoencoder.add(BatchNormalization())
        if print_layers_size: print("dense 3 out: {}".format(self.autoencoder.get_output_shape_at(0)))
        self.autoencoder.add(Reshape((25,25,1),input_shape=(625,)))
        if print_layers_size: print("reshape out: {}".format(self.autoencoder.get_output_shape_at(0)))



        self.autoencoder.add(Conv2D(1, 3,
                                    activity_regularizer=l1_l2(self.w_act_reg, self.w_act_reg),
                                    kernel_regularizer=l2(self.w_kernel_reg), bias_regularizer=l2(self.w_bias_reg),
                                    padding='same'))
        self.autoencoder.add(LeakyReLU(alpha=0.3))
        if normalize_batch: self.autoencoder.add(BatchNormalization())
        if print_layers_size: print("1st deconv out: {}".format(self.autoencoder.get_output_shape_at(0)))
        self.autoencoder.add(UpSampling2D((2, 2)))
        if print_layers_size: print("1st upsample out: {}".format(self.autoencoder.get_output_shape_at(0)))
        self.autoencoder.add(Conv2D(1, 3,
                                    activity_regularizer=l1_l2(self.w_act_reg, self.w_act_reg),
                                    kernel_regularizer=l2(self.w_kernel_reg), bias_regularizer=l2(self.w_bias_reg),
                                    padding='same',
                                    ))
        self.autoencoder.add(LeakyReLU(alpha=0.3))
        if normalize_batch: self.autoencoder.add(BatchNormalization())
        if print_layers_size: print("2rd deconv out: {}".format(self.autoencoder.get_output_shape_at(0)))
        self.autoencoder.add(UpSampling2D((2, 2)))
        if print_layers_size: print("2rd upsample out: {}".format(self.autoencoder.get_output_shape_at(0)))
        self.autoencoder.add(Conv2D(1, 3,
                                    activity_regularizer=l1_l2(self.w_act_reg, self.w_act_reg),
                                    kernel_regularizer=l2(self.w_kernel_reg), bias_regularizer=l2(self.w_bias_reg),
                                    padding='same'))
        if normalize_batch: self.autoencoder.add(BatchNormalization())
        self.autoencoder.add(Activation('sigmoid'))
        if print_layers_size: print("decoded out: {}".format(self.autoencoder.get_output_shape_at(0)))

        self.autoencoder.compile(optimizer=optimizer(), loss=loss_function, metrics=[metrics.mean_absolute_error, metrics.mean_squared_error])
        print("Network built")

    def trainNet(self, trainData):
        self.train_history = self.autoencoder.fit(trainData, trainData,
                        epochs=50,
                        batch_size=50,
                        shuffle=True,
                        validation_split=0.05
                        )

    def get_train_history(self):
        return self.train_history

    def visualise_history(self, save_path, file_name):
        import matplotlib.pyplot as plt
        import matplotlib.pylab as pylab
        import numpy

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(self.train_history.history['loss'])
        ax.plot(self.train_history.history['val_loss'])
        plt.ylabel('wartość błędu')
        plt.xlabel('nr epoki')
        plt.legend(['zbiór uczący', 'zbiór walidacyjny'], loc='upper left')
        fig.savefig(save_path + file_name + ".png")
        fig.clear()
        plt.close()

    def test_net(self, test_data):
        return self.autoencoder.evaluate(test_data, test_data)

    def get_features(self, data):
        return self.decoder.predict(data)

    def cross_validate_net(self, data, n_splits):
        from sklearn.model_selection import KFold
        skfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for train_index, test_index in skfold.split(data):
            self.initialiseNet(data.shape[1], data.shape[2], False, False, "Adam")
            self.trainNet(data[train_index])
            scores.append(self.test_net(data[test_index]))
        scores = numpy.asarray(scores)
        scores = numpy.reshape(scores, [scores.shape[0], len(scores[0])])
        print(scores)
        f = open("/home/michal/PycharmProjects/AudioFeatureExtraction/charts/cross_valid_results_100f_50e_743.txt", "w")
        f.write("mean of {} binary crossentropies: {}\n".format(n_splits, numpy.mean(scores[:,0]).astype('str')))
        f.write("std of {} binary crossentropies: {}\n".format(n_splits, numpy.std(scores[:,0]).astype('str')))
        f.write("mean of {} absolute errors: {}\n".format(n_splits, numpy.mean(scores[:,1]).astype('str')))
        f.write("std of {} absolute errors: {}\n".format(n_splits, numpy.std(scores[:,1]).astype('str')))
        f.write("mean of {} squared errors: {}\n".format(n_splits, numpy.mean(scores[:,2]).astype('str')))
        f.write("std of {} squared errors: {}\n".format(n_splits, numpy.std(scores[:,2]).astype('str')))
        f.close()


