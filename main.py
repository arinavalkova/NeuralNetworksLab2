import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def read_dataset():
    df = pd.read_csv('withCorel.csv', sep=';', encoding='windows-1251')
    return df


def norm(df):
    arr = df.to_numpy()
    for i in range(
            arr.shape[1]):
        max = np.nanmax(arr[:, i])
        min = np.nanmin(arr[:, i])
        for j in range(arr.shape[0]):
            arr[j, i] = (arr[j, i] - min) / (max - min)
    return arr


first_neir = 19
second_neir = 80
third_neir = 60
fourth_neir = 2

ep = 1000
coeff = 0.02


def layer_activation(input, w):
    return np.dot(input, w)


def test(test_dataset, w1, w2, w3, plot=False):
    kgf_result = []
    g_total_result = []
    kgf_true = []
    g_total_true = []
    g_total_mse = 0
    kgf_mse = 0
    for data_row in test_dataset:
        input_data = np.nan_to_num(data_row[1:-2])
        neir1 = layer_activation(input_data, w1)
        sigm1 = sigmoid(neir1)
        neir2 = layer_activation(sigm1, w2)
        sigm2 = sigmoid(neir2)
        neir3 = layer_activation(sigm2, w3)

        g_total_result.append(neir3[0])
        kgf_result.append(neir3[1])
        g_total_true.append(data_row[-2])
        kgf_true.append(data_row[-1])
        g_total_mse += np.nan_to_num((g_total_true[-1] - g_total_result[-1]))**2/len(test_dataset)
        kgf_mse += np.nan_to_num((kgf_true[-1] - kgf_result[-1]))**2/len(test_dataset)

    if plot:
        print('Test data mse:')
        print('\tG_total:', g_total_mse)
        print('\tKGF:', kgf_mse)
        x = np.arange(len(test_dataset))
        fig, ax = plt.subplots()
        ax.plot(x, g_total_result, label='g_total_result')
        ax.plot(x, g_total_true, label='g_total_true')
        ax.legend()
        plt.show()
        fig, ax = plt.subplots()
        ax.plot(x, kgf_result, label='kgf_result')
        ax.plot(x, kgf_true, label='kgf_true')
        ax.legend()
        plt.show()
    return [g_total_mse, kgf_mse]


def draw_mse(mse, test_mse):
    kgf_mse = []
    g_total_mse = []
    test_kgf_mse = []
    test_g_total_mse = []
    for row in mse:
        kgf_mse.append(row[1])
        g_total_mse.append(row[0])
    for row in test_mse:
        test_kgf_mse.append(row[1])
        test_g_total_mse.append(row[0])

    x = np.arange(len(kgf_mse))
    fig, ax = plt.subplots()
    ax.plot(x, kgf_mse, label='train kgf')
    ax.plot(x, test_kgf_mse, label='test kgf')
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x, g_total_mse, label='train g_total')
    ax.plot(x, test_g_total_mse, label='test g_total')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    arr = norm(read_dataset())

    w1 = np.random.normal(0, 1, (first_neir, second_neir))
    w2 = np.random.normal(0, 1, (second_neir, third_neir))
    w3 = np.random.normal(0, 1, (third_neir, fourth_neir))

    train_num = int(0.8 * arr.shape[0])
    np.random.shuffle(arr)
    train_dataset = arr[:train_num]
    test_dataset = arr[train_num:]
    mse_list = []
    test_mse_list = []

    for i in range(ep):
        dW1 = np.zeros_like(w1)
        dW2 = np.zeros_like(w2)
        dW3 = np.zeros_like(w3)

        mse = np.zeros(2)

        for j in range(train_num):
            input_data = np.nan_to_num(train_dataset[j][1:-2])
            neir1 = layer_activation(input_data, w1)
            sigm1 = sigmoid(neir1)
            neir2 = layer_activation(sigm1, w2)
            sigm2 = sigmoid(neir2)
            neir3 = layer_activation(sigm2, w3)

            err3 = np.nan_to_num(neir3 - train_dataset[j][-2:])
            mse += err3 ** 2 / train_num
            grad3 = err3 * 2
            dW3 += np.dot(sigm2.reshape(third_neir, 1), grad3.reshape(1, fourth_neir))
            grad2 = np.dot(w3, grad3) * sigm2 * (1 - sigm2)
            dW2 += np.dot(sigm1.reshape(second_neir, 1), grad2.reshape(1, third_neir))
            grad1 = np.dot(w2, grad2) * sigm1 * (1 - sigm1)
            dW1 += np.dot(input_data.reshape(first_neir, 1), grad1.reshape(1, second_neir))

        if i % 10 == 0:
            print('train mse:', mse)
        if i > 20:
            test_mse = test(test_dataset, w1, w2, w3)
            test_mse_list.append(test_mse)
            mse_list.append(mse)
        w1 -= dW1 * coeff / train_num
        w2 -= dW2 * coeff / train_num
        w3 -= dW3 * coeff / train_num

    draw_mse(mse_list, test_mse_list)
    test(test_dataset, w1, w2, w3, plot=True)
