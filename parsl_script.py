import parsl
from parsl import python_app, bash_app, File
import json

parsl.load()

@bash_app
def install_cunumeric():
    return f"""
    cd /home/ubuntu/data/parsl/legate.core
    ./install.py --cuda
    cd /home/ubuntu/data/parsl/legate.core/cunumeric
    ./install.py
    """

@bash_app
def preprocess_data(stdout='preprocess_data.out', stderr='preprocess_data.err'):
    return f"""
    cd /home/ubuntu/data/parsl/project
    legate preprocess_data.py --output_file mnist_preprocessed.npz
    """

@bash_app
def install_flexflow():
    return f"""
    cd /home/ubuntu/data/parsl/FlexFlow/build
    make install
    """

@bash_app
def train_model( stdout='train_model.out', stderr='train_model.err'):
    return f"""
   /home/ubuntu/data/parsl/FlexFlow/build/flexflow_python /home/ubuntu/data/parsl/project/train_script.py -ll:py 1 -ll:gpu 1 -ll:fsize 20000 -ll:zsize 20000 --data_file /home/ubuntu/data/parsl/project/mnist_preprocessed.npz
    """


if __name__ == "__main__":
    # install_cunumeric_task = install_cunumeric()
    # preprocess_data_task = preprocess_data()
    # preprocess_data_task.depends_on(install_cunumeric_task)

    # install_flexflow_task = install_flexflow()
    # install_flexflow_task.depends_on(preprocess_data_task)

    train_model_task = train_model()
    # train_model_task.depends_on(install_flexflow_task)

    # train_model_task = train_model(preprocess_data_task.outputs[0])
    # train_model_task.depends_on(install_flexflow_task)

    train_model_task.result()
    print("Model training completed.")
