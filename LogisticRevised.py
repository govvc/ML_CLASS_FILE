import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

X_train = np.loadtxt("F:/ML_DATA/Exam/train/x.txt")
y_train = np.loadtxt("F:/ML_DATA/Exam/train/y.txt")

X_test = np.loadtxt("F:/ML_DATA/Exam//test/x.txt")
y_test = np.loadtxt("F:/ML_DATA/Exam/test/y.txt")


def standardize(z):
    return (z - np.mean(z)) / np.std(z)


class logistic:
    def __init__(self, lr, epochs, X, y):
        self.lr = lr
        self.epochs = epochs
        self.X = X
        self.y = y

    def sigmoid(self, z):
        return 1 / (1 + np.exp(z))

    def fit(self):
        """
        train the dataset
        """

        x = np.insert(self.X, 0, 1, axis=1)
        np.random.seed(4)
        self.loss = []
        self.theta = []
        w = np.random.rand(x.shape[1])

        for i in range(0, self.epochs):
            h = self.sigmoid(np.dot(x, w.T))
            J = np.sum(np.dot(self.y, np.log(h)) + np.dot(1 - self.y, np.log(1 - h)))

            # print('---------------------\n', -J)
            w = w - self.lr * np.dot(self.y - h, x)

            self.loss.append(-J)
            self.theta.append(w)

    def predict(self, x, y):
        """
        :return: prediction
        """

        predict_, ac_ = [], []
        for i in range(self.epochs):
            h = self.sigmoid(np.dot(x, self.theta[i]))

            for j in range(len(h)):
                if h[j] > (1 - h[j]):
                    predict_.append(1)
                else:
                    predict_.append(0)

            ac_.append(np.sum(predict_ == y) / len(predict_))

            predict_.clear()

        return ac_


if __name__ == '__main__':
    X_train_1 = standardize(X_train)
    log = logistic(0.001, 1000, X_train_1, y_train)
    log.fit()

    # Preparation for drawing
    X_train_1 = np.insert(X_train_1, 0, 1, axis=1)
    ac_train = log.predict(X_train_1, y_train)
    X_test_1 = standardize(X_test)
    X_test_1 = np.insert(X_test_1, 0, 1, axis=1)
    ac_test = log.predict(X_test_1, y_test)

    fig, axes = plt.subplots(1, 3)

    axes[0].set_xlim([0, log.epochs])
    axes[0].set_ylim([10, 80])
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')

    axes[1].set_xlim([0, log.epochs])
    axes[1].set_ylim([0.2, 1])
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')

    axes[2].set_xlim([0, 80])
    axes[2].set_ylim([40, 90])
    axes[2].set_xlabel('X1')
    axes[2].set_ylabel('X2')
    axes[2].set_title('Classify')

    epochs = np.arange(0, log.epochs, 10)
    line, = axes[0].plot([], [])

    scatter1 = axes[1].scatter([], [], s=4, c='b')
    scatter2 = axes[1].scatter([], [], s=4, c='r')
    axes[1].legend(["training set", "test set"])

    scatter3 = axes[2].scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='s')
    scatter4 = axes[2].scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='^')
    axes[2].legend(["training set", "test set"])
    line0, = axes[2].plot([], [])
    log.theta = np.asarray(log.theta)

x = np.arange(0, 80, 0.8)


def animate(i):
    line.set_data(range(0,i), log.loss[:i])  # update data

    scatter1.set_offsets(np.stack((range(0,i), ac_train[:i]), axis=1))  # update data
    scatter2.set_offsets(np.stack((range(0,i), ac_test[:i]), axis=1))

    line0.set_data(x, -(log.theta[i][0] + log.theta[i][1] * (x - np.mean(X_train[:, 0])) / np.std(X_train[:, 0])) /
                   log.theta[i][2] * np.std(X_train[:, 1]) + np.mean(X_train[:, 1]))


ani = animation.FuncAnimation(fig, animate, interval=1, frames=epochs, repeat=False)
# ani.save('Logistic.gif', writer='pillow', fps=100) #save the animation
plt.show()
