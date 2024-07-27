import cunumeric as np
from flexflow.keras.datasets import mnist


def main(output_file):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape)

    # x_train = x_train.reshape(60000, 784)
    # x_train = x_train.astype('float32')
    x_train = np.reshape(x_train, (60000, 784))
    x_train = x_train.astype(np.float32)
    x_train /= 255
    # y_train = y_train.astype('int32')
    y_train = y_train.astype(np.int32)
    y_train = np.reshape(y_train, (len(y_train), 1))
    np.savez(output_file, x_train=x_train, y_train=y_train)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', required=True)
    args = parser.parse_args()
    main(args.output_file)
