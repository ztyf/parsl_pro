from flexflow.core import *
# import cunumeric as np
import numpy as np
from flexflow.keras.datasets import mnist
import argparse, json


def top_level_task(data_file):
    ffconfig = FFConfig()
    print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" % (
        ffconfig.batch_size, ffconfig.workers_per_node, ffconfig.num_nodes))
    ffmodel = FFModel(ffconfig)

    dims_input = [ffconfig.batch_size, 784]
    input_tensor = ffmodel.create_tensor(dims_input, DataType.DT_FLOAT)

    num_samples = 60000

    kernel_init = UniformInitializer(12, -1, 1)
    t = ffmodel.dense(input_tensor, 512, ActiMode.AC_MODE_RELU,
                      kernel_initializer=kernel_init)
    t = ffmodel.dense(t, 512, ActiMode.AC_MODE_RELU)
    t = ffmodel.dense(t, 10)

    t = ffmodel.softmax(t)

    ffoptimizer = SGDOptimizer(ffmodel, 0.01)
    ffmodel.optimizer = ffoptimizer
    ffmodel.compile(loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics=[
                    MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY])
    label_tensor = ffmodel.label_tensor

    data = np.load(data_file)
    x_train = data['x_train']
    y_train = data['y_train']

    dataloader_input = ffmodel.create_data_loader(input_tensor, x_train)
    dataloader_label = ffmodel.create_data_loader(label_tensor, y_train)

    ffmodel.init_layers()

    epochs = ffconfig.epochs

    ts_start = ffconfig.get_current_time()

    ffmodel.fit(x=dataloader_input, y=dataloader_label, epochs=epochs)
    ffmodel.eval(x=dataloader_input, y=dataloader_label)

    ts_end = ffconfig.get_current_time()
    run_time = 1e-6 * (ts_end - ts_start)
    print("epochs %d, ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n" %
          (epochs, run_time, num_samples * epochs / run_time))

    perf_metrics = ffmodel.get_perf_metrics()

    return perf_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config-file",
        help="The path to a JSON file with the configs. If omitted, a sample model and configs will be used instead.",
        type=str,
        default=None,
    )
    parser.add_argument('--data_file', required=True)
    
    args, unknown = parser.parse_known_args()
    configs_dict = None
    if args.config_file is not None:
        with open(args.config_file) as f:
            configs_dict = json.load(f)
    init_flexflow_runtime(configs_dict)
    print("mnist mlp")
    top_level_task(args.data_file)
