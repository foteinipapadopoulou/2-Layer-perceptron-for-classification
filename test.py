import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import load_cifar10_data as load_cifar

global branch

def convert_number_to_category(val):
    my_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
               5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
    return my_dict.get(val)

# h function
def h(a, ax=1):
    if (branch == 1):
        m = np.exp(a)
        return np.log(1 + m)
    elif (branch == 2):
        m = np.exp(a)
        n = np.exp(-a)
        numerator = np.subtract(m, n)
        denominator = np.add(m, n)
        return np.divide(numerator, denominator)
    else:
        return np.cos(a)


def dH(a, ax=1):
    if (branch == 1):
        m = np.exp(a)
        n = np.add(m, 1)
        return np.divide(m, n)
    elif (branch == 2):
        p = np.multiply(a, 2)
        m = np.exp(p)
        num = np.multiply(m, 4)
        d = np.add(m, 1)
        den = np.power(d, 2)
        return num / den
    else:
        return -1 * np.sin(a)


# use by default ax=1, when the array is 2D
# use ax=0 when the array is 1D
def softmax(x, ax=1):
    m = np.max(x, axis=ax, keepdims=True)  # max per row
    p = np.exp(x - m)
    return (p / np.sum(p, axis=ax, keepdims=True))

    """
    Load the MNIST dataset. Reads the training and testing files and create matrices.
    :Expected return:
    train_data:the matrix with the training data
    test_data: the matrix with the data that will be used for testing
    y_train: the matrix consisting of one
                        hot vectors on each row(ground truth for training)
    y_test: the matrix consisting of one
                        hot vectors on each row(ground truth for testing)
    """


def load_data():
    # load the train files
    df = None

    y_train = []

    for i in range(10):
        tmp = pd.read_csv('data/mnist/train%d.txt' % i, header=None, sep=" ")
        # build labels - one hot vector
        hot_vector = [1 if j == i else 0 for j in range(0, 10)]

        for j in range(tmp.shape[0]):
            y_train.append(hot_vector)
        # concatenate dataframes by rows
        if i == 0:
            df = tmp
        else:
            df = pd.concat([df, tmp])

    train_data = df.to_numpy()
    y_train = np.array(y_train)

    # load test files
    df = None

    y_test = []

    for i in range(10):
        tmp = pd.read_csv('data/mnist/test%d.txt' % i, header=None, sep=" ")
        # build labels - one hot vector

        hot_vector = [1 if j == i else 0 for j in range(0, 10)]

        for j in range(tmp.shape[0]):
            y_test.append(hot_vector)
        # concatenate dataframes by rows
        if i == 0:
            df = tmp
        else:
            df = pd.concat([df, tmp])

    test_data = df.to_numpy()
    y_test = np.array(y_test)

    return train_data, test_data, y_train, y_test

def gradcheck_softmax(W1init, W2init, X, t, lamda):
    W1 = np.random.rand(*W1init.shape)
    # print("W1:",pd.DataFrame(W1).head())
    W2 = np.random.rand(*W2init.shape)
    # print("W2:",pd.DataFrame(W2).head())

    epsilon = 1e-4

    _list = np.random.randint(X.shape[0], size=5)
    x_sample = np.array(X[_list, :])
    t_sample = np.array(t[_list, :])

    Ew, gradEw1, gradEw2 = cost_grad_softmax(W1, W2, x_sample, t_sample, lamda)

    print("gradEw1 shape: ", gradEw1.shape)
    print("gradEw2 shape: ", gradEw2.shape)
    numericalGrad1 = np.zeros(gradEw1.shape)
    numericalGrad2 = np.zeros(gradEw2.shape)
    # Compute all numerical gradient estimates and store them in
    # the matrix numericalGrad

    for k in range(numericalGrad1.shape[0]):
        for d in range(numericalGrad1.shape[1]):
            # add epsilon to the w[k,d]
            w_tmp1 = np.copy(W1)
            w_tmp1[k, d] += epsilon
            e_plus, _, a = cost_grad_softmax(w_tmp1, W2, x_sample, t_sample, lamda)

            # subtract epsilon to the w[k,d]
            w_tmp1 = np.copy(W1)
            w_tmp1[k, d] -= epsilon
            e_minus, _, a = cost_grad_softmax(w_tmp1, W2, x_sample, t_sample, lamda)

            # approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
            numericalGrad1[k, d] = (e_plus - e_minus) / (2 * epsilon)
            # print("Actual1:",numericalGrad1[k,d])
            # print("Computed1:",gradEw1[k,d])

    for k in range(numericalGrad2.shape[0]):
        for d in range(numericalGrad2.shape[1]):
            # add epsilon to the w[k,d]
            w_tmp2 = np.copy(W2)
            w_tmp2[k, d] += epsilon
            e_plus, _, a = cost_grad_softmax(W1, w_tmp2, x_sample, t_sample, lamda)

            # subtract epsilon to the w[k,d]
            w_tmp2 = np.copy(W2)
            w_tmp2[k, d] -= epsilon
            e_minus, _, a = cost_grad_softmax(W1, w_tmp2, x_sample, t_sample, lamda)

            # approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
            numericalGrad2[k, d] = (e_plus - e_minus) / (2 * epsilon)

    return gradEw1, gradEw2, numericalGrad1, numericalGrad2


def cost_grad_softmax(W1, W2, X, t, lamda):
    z1 = X.dot(W1.T)
    Z = h(z1)
    Z = np.hstack([Z, np.ones((Z.shape[0], 1))])
    z2 = Z.dot(W2.T)
    max_error = np.max(z2, axis=1)
    y = softmax(z2)

    # Compute the cost function to check convergence
    # Using the logsumexp trick for numerical stability
    Ew = np.sum(t * z2) - np.sum(max_error) - \
         np.sum(np.log(np.sum(np.exp(z2 - np.array([max_error, ] * z2.shape[1]).T), 1))) - \
         (0.5 * lamda) * np.sum(np.square(W2)) - (0.5 * lamda) * np.sum(np.square(W1))

    # calculate gradient
    gradEw2 = (t - y).T.dot(Z) - lamda * W2
    gradEw1 = np.dot(dH(z1).T * np.dot(W2[:, :-1].T, (t - y).T), X) - lamda * W1
    return Ew, gradEw1, gradEw2


def ml_softmax_train(t, X, lamda, W1init, W2init, options):
    """inputs :
      t: N x 1 binary output data vector indicating the two classes
      X: N x (D+1) input data vector with ones already added in the first column
      lamda: the positive regularizarion parameter
      winit: D+1 dimensional vector of the initial values of the parameters
      options: options(1) is the maximum number of iterations
               options(2) is the tolerance
               options(3) is the learning rate eta
    outputs :
      w: the trained D+1 dimensional vector of the parameters"""

    W1 = W1init
    W2 = W2init

    # Maximum number of iteration of gradient ascend
    _iter = options[0]

    # Tolerance
    tol = options[1]

    # Learning rate
    eta = options[2]

    Ewold = -np.inf
    batch_size = 100
    m = len(t)
    cost_history = np.zeros(_iter)
    y = np.copy(t)
    for it in range(1, _iter):
        costs = 0.0
        indices = np.random.permutation(m)
        X = X[indices]
        y = y[indices]
        for i in range(0, m, batch_size):
            X_i = X[i:i + batch_size]
            y_i = y[i:i + batch_size]
            Ew, gradEw1, gradEw2 = cost_grad_softmax(W1, W2, X_i, y_i, lamda)
            # save cost
            costs += Ew
            # Show the current cost function on screen
            if i % 50 == 0:
                print('Iteration : %d, Cost function :%f' % (it, Ew))
            # Break if you achieve the desired accuracy in the cost function
            if np.abs(Ew - Ewold) < tol:
                break

            # Update parameters based on gradient ascend
            W1 = W1 + eta * gradEw1
            W2 = W2 + eta * gradEw2
            Ewold = Ew
        print(costs)
        cost_history[it] = costs
    return W1, W2, cost_history

"""
X_train, X_test, y_train, y_test = load_data()
# plot 5 random images from the training set

n = 100
sqrt_n = int( n**0.5 )
samples = np.random.randint(X_train.shape[0], size=n)

plt.figure( figsize=(11,11) )

cnt = 0
for i in samples:
    cnt += 1
    plt.subplot( sqrt_n, sqrt_n, cnt )
    plt.subplot( sqrt_n, sqrt_n, cnt ).axis('off')
    plt.imshow( X_train[i].reshape(28,28), cmap='gray'  )

plt.show()

X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255
"""
cifar_10_dir = 'cifar-10-batches-py'
# Transform label indices to one-hot encoded vectors
(X_train, y_train), (X_test, y_test) = load_cifar.load_data(cifar_10_dir)
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
# Transform images from (32,32,3) to 3072-dimensional vectors (32*32*3)

X_train = np.reshape(X_train, (50000, 3072))
X_test = np.reshape(X_test, (10000, 3072))
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalization of pixel values (to [0-1] range)

X_train /= 255
X_test /= 255



X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

N, D = X_train.shape
M = 300  # try 100 or 200 or 300
K = 10  # num of classes

# initialize w for the gradient ascent
W1init=np.zeros((M, D))
W2init=np.zeros((K, M+1))

branch = 3
# regularization parameter
lamda = 0.1

# options for gradient descent
options = [500, 1e-6, 0.5*10/N]  # Maximum number of iteration of gradient ascend, Tolerance, Learning rate
"""
#unccoment these lines above to run gradient
gradEw1, gradEw2, numericalGrad1, numericalGrad2 = gradcheck_softmax(W1init, W2init, X_train, y_train, lamda)
# Absolute norm
print("The difference estimate for gradient of w1 is : ", np.max(np.abs(gradEw1 - numericalGrad1)))
print("The difference estimate for gradient of w2 is : ", np.max(np.abs(gradEw2 - numericalGrad2)))
print(pd.DataFrame( gradEw1 ).head())
print(pd.DataFrame( gradEw2 ).head())
print(pd.DataFrame( numericalGrad1 ).head())
print(pd.DataFrame( numericalGrad2 ).head())
"""

# Train the model
W1, W2, costs = ml_softmax_train(y_train, X_train, lamda, W1init, W2init, options)

plt.plot(np.squeeze(costs[1:]))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(format(options[2], 'f')))
plt.show()


def ml_softmax_test(W1, W2, X_test):
    z1 = X_test.dot(W1.T)
    Z = h(z1)
    Z = np.hstack([Z, np.ones((Z.shape[0], 1))])
    z2 = Z.dot(W2.T)
    ytest = softmax(z2)
    # Hard classification decisions
    ttest = np.argmax(ytest, 1)
    return ttest


pred = ml_softmax_test(W1, W2, X_test)
#pred=keras.utils.to_categorical(pred,num_classes=10)
print("Accuracy:", np.mean(pred == np.argmax(y_test, 1)))

faults = np.where(np.not_equal(np.argmax(y_test, 1), pred))[0]
# plot n misclassified examples from the Test set
n = 25
samples = np.random.choice(faults, n)
sqrt_n = int(n ** 0.5)
"""
#Uncomment these lines above for mnist data
plt.figure(figsize=(11, 13))

cnt = 0
for i in samples:
    cnt += 1
    plt.subplot(sqrt_n, sqrt_n, cnt)
    plt.subplot(sqrt_n, sqrt_n, cnt).axis('off')
    plt.imshow(X_test[i, 1:].reshape(28, 28) * 255, cmap='gray')
    plt.title("True: " + str(np.argmax(y_test, 1)[i]) + "\n Predicted: " + str(pred[i]))
plt.show()
print("Iteration=", options[0], ",lr=", options[2], ",batch_size=100")
"""
#For cifar10 data
plt.figure(figsize=(11, 13))
a = X_test[:, 1:]
a = np.reshape(a, (10000, 32, 32, 3))

cnt = 0
for i in samples:
    cnt += 1
    plt.subplot(sqrt_n, sqrt_n, cnt)
    plt.subplot(sqrt_n, sqrt_n, cnt).axis('off')
    plt.imshow(a[i, 1:])
    plt.title("True: " + convert_number_to_category(np.argmax(y_test, 1)[i]) + "\n Predicted: " + convert_number_to_category(pred[i]))
plt.show()
print("Iteration=", options[0], ",lr=", options[2], ",batch_size=100")
