import matplotlib.pyplot as plt
import numpy as np
import warnings
from matplotlib import animation

warnings.simplefilter("error")

X_train = np.loadtxt("D:/ML_DATA/iris/train/x.txt")
y_train = np.loadtxt("D:/ML_DATA/iris/train/y.txt").astype('int64')

X_test = np.loadtxt("D:/ML_DATA/iris//test/x.txt")
y_test = np.loadtxt("D:/ML_DATA/iris/test/y.txt").astype('int64')




class softmax:
    def __init__(self, lr, epochs):
        self.lr = lr  # learning rate
        self.epochs = epochs  # circulation times

    def softmax(self, z):
        """
        compute the sum of each row and reshape the matrix as one row
        :param z:
        :return:
        """

        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)

    def fit(self, X, y):

        self.X = np.insert(X, 0, 1, axis=1)
        self.y = y
        n_classes = len(np.unique(y))  # number of classes
        np.random.seed(2)
        self.w = [np.random.rand(self.X.shape[1], n_classes)]  # initialize weights

        error = []  # list of the deviation of the real value and predicted value
        self.loss = []  # list of loss

        y = np.eye(len(y), np.sum(np.unique(y)))[y]  # one_hot encoding

        for i in range(self.epochs):
            y_ = self.softmax(np.dot(self.X, self.w[i]))

            error.append(y - y_)  # add new error to the error list
            J = -np.sum(np.multiply(np.log(y_), y))  # compute the loss
            print("------------------")

            self.loss.append(J)  # add new loss to the loss list
            print("loss: ",J)
            self.w.append(self.w[i] + self.lr * (np.dot(self.X.T, error[-1])))

    # @jit
    def predict(self, x, y):
        """

        :return: prediction
        """

        predict_, ac_ = self.y, []
        for i in np.arange(self.epochs):
            predict_ = (np.argmax(self.softmax(np.dot(x, self.w[i])), axis=1))
            ac_.append(np.sum(predict_ == y) / len(y))

        return ac_


if __name__ == '__main__':

    s = softmax(0.001, 500)

    s.fit(X_train, y_train)
    X_train_1 = np.insert(X_train, 0, 1, axis=1)
    ac_train = s.predict(X_train_1, y_train)
    X_test_1 = np.insert(X_test, 0, 1, axis=1)
    ac_test = s.predict(X_test_1, y_test)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Preparation for drawing
    axes[0].set_xlim([0, s.epochs])
    axes[0].set_ylim([10, 150])
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')

    axes[1].set_xlabel('X1')
    axes[1].set_ylabel('X2')
    axes[1].set_title('Classify')

    axes[2].set_xlim([0, s.epochs])
    #
    axes[2].set_ylim([0.2, 1])
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Accuracy')

    line0, = axes[0].plot([], [])

    x1_min, x1_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    x2_min, x2_max = X_train[:, 1].min() - 0.1, X_train[:, 1].max() + 0.1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                           np.arange(x2_min, x2_max, 0.01))

    X = np.array([xx1.ravel(), xx2.ravel()]).T
    X = np.insert(X, 0, 1, axis=1)

    axes[1].scatter(s.X[:, 1], s.X[:, 2], c=s.y, s=10, marker='s')
    axes[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=10, marker='^')
    axes[1].legend(["training set", "test set"])
    epochs = np.arange(0,s.epochs,10)

    Z = np.argmax(s.softmax(np.dot(X, s.w[0])), axis=1)

    Z = Z.reshape(xx1.shape)
    cont = axes[1].contourf(xx1, xx2, Z, alpha=0.2, cmap='brg')

    scatter2 = axes[2].scatter([], [], s=5, c='b')
    scatter3 = axes[2].scatter([], [], s=5, c='r')
    axes[2].legend(["training set", "test set"])


    def animate(i):

        global cont
        for c in cont.collections:
            c.remove()
        line0.set_data(range(0,i), s.loss[:i])  # update data

        scatter2.set_offsets(np.stack((range(0,i), ac_train[:i]), axis=1))  # update data
        scatter3.set_offsets(np.stack((range(0,i), ac_test[:i]), axis=1))
        Z = np.argmax(s.softmax(np.dot(X, s.w[i])), axis=1) #compute new hypothesis
        Z = Z.reshape(xx1.shape)
        cont = axes[1].contourf(xx1, xx2, Z, alpha=0.2, cmap='brg')



    # plt.get_current_fig_manager().full_screen_toggle()

    ani = animation.FuncAnimation(fig, animate, interval=1, frames=epochs, repeat=False)
    #ani.save('softmax.gif', writer='pillow', fps=100) #save the animation
    plt.show()

 
