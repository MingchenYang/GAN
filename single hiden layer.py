import numpy as np
import scipy

train_dataset_dir = 'C:/Users/yangm90/Desktop/hw5/largeTrain.csv'
test_dataset_dir = 'C:/Users/yangm90/Desktop/hw5/largeTest.csv'

Learning_rate = 0.01
Hidden_layer_node = 50

Epochs = 100


def loadFile(input_dir):
    dataset = np.loadtxt(input_dir, dtype=np.str, delimiter=',')
    data = dataset[0:, 1:].astype(np.float32)
    label = dataset[0:, 0].astype(np.int8)
    return data, label


def convert_to_onehot(x):
    onehot_x = np.eye(10)[x]
    return onehot_x


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def softmax(x):
    shift_x = x - np.max(x)
    exp_x = np.exp(shift_x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def cross_entropy(a, b):
    return np.sum(np.nan_to_num(-b*np.log(a) - (1-b)*np.log(1-a)))


def derivative_w2(target, layer4, layer2):
    derivative_w2 = np.zeros([Hidden_layer_node, 10])
    target = np.transpose(target)
    layer4 = np.transpose(layer4)
    layer2 = np.transpose(layer2)

    for n in range(1):
        for m in range(50):
            for k in range(10):
                derivative_w2 += (target[n, k] - layer4[n, k]) * layer2[n, m]

    return derivative_w2


def derivative_b2(target, layer4):
    return np.sum(target - layer4, axis = 0)


def derivative_w1(target, layer4, w2, layer2, x):
    derivative_w1 = np.zeros([128, Hidden_layer_node])
    target = np.transpose(target)
    layer4 = np.transpose(layer4)
    layer2 = np.transpose(layer2)
    w2 = np.transpose(w2)
    x = np.transpose(x)

    for n in range(1):
        for m in range(50):
            for d in range(128):
                for k in range(10):
                    derivative_w1[d, m] += (target[n, k] - layer4[n, k]) * w2[m, k] * layer2[n, m] * (1 - layer2[n, m]) * x[n, d]

    return derivative_w1


def derivatibe_b1(target, layer4, w2, layer2):
    derivative_b1 = np.zeros(Hidden_layer_node)
    target = np.transpose(target)
    layer4 = np.transpose(layer4)
    layer2 = np.transpose(layer2)
    w2 = np.transpose(w2)

    for n in range(1):
        for m in range(50):
            for k in range(10):
                derivative_b1 += (target[n, k] - layer4[n, k]) * w2[m, k] * layer2[n, m] * (1 - layer2[n, m]) * 1

    return derivative_b1


def train():
    train_data, train_label = loadFile(train_dataset_dir)
    test_data, test_label = loadFile(test_dataset_dir)

    train_label_onehot = convert_to_onehot(train_label)
    test_label_onehot = convert_to_onehot(test_label)

    train_data_shape = np.shape(train_data)
    #(9000, 128)
    train_data_num = train_data_shape[0]
    train_data_node = train_data_shape[1]
    test_data_shape = np.shape(test_data)
    test_data_num = test_data_shape[0]
    test_data_node = test_data_shape[1]

    W_1 = np.random.normal(0, 1.0, [Hidden_layer_node, train_data_node])
    B_1 = np.zeros([Hidden_layer_node, 1])
    W_2 = np.random.normal(0, 1.0, [10, Hidden_layer_node])
    B_2 = np.zeros([10, 1])

    k = 0
    for epoch in range(Epochs):
        for i in range(train_data_num):
            input = np.array(train_data[i], ndmin=2).T
            target = np.array(train_label_onehot[i], ndmin=2).T
            layer_1 = np.dot(W_1, input) + B_1
            # (50, 1)
            layer_2 = sigmoid(layer_1)
            # (50, 1)
            layer_3 = np.dot(W_2, layer_2) + B_2
            # (10, 1)
            layer_4 = softmax(layer_3)
            # (10, 1)
            loss = cross_entropy(layer_4, target)
            if k%100 == 0:
                print("Step:{} Loss:{:.10f}".format(k, loss))
            # (10, 50)
            # W_2 = W_2 + Learning_rate * np.dot((loss * (layer_4-target)), np.transpose(layer_2))
            deri_w2 = derivative_w2(target, layer_4, layer_2)
            W_2 = W_2 + Learning_rate * np.transpose(deri_w2)
            deri_b2 = derivative_b2(target, layer_4)
            #B_2 = B_2 + Learning_rate * np.transpose(deri_b2)
            deri_w1 = derivative_w1(target, layer_4, W_2, layer_2, input)
            W_1 = W_1 + Learning_rate * np.transpose(deri_w1)
            deri_b1 = derivatibe_b1(target, layer_4, W_2, layer_2)
            #B_1 = B_1 + Learning_rate * np.transpose(deri_b1)
            k += 1



def main(argv=None):
    train()


if __name__ == '__main__':
    main()