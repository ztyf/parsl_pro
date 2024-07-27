import parsl
from parsl import python_app, bash_app, File
import json

parsl.load()

@bash_app
def install_cunumeric(stdout='install_cunumeric.out', stderr='install_cunumeric.err'):
    return f"""
    cd $FF_HOME/legate.core/
    ./install.py --cuda
    cd $FF_HOME/legate.core/cunumeric
    ./install.py
    """

@bash_app
def preprocess_data(stdout='preprocess_data.out', stderr='preprocess_data.err'):
    return f"""
    cd $FF_HOME/parsl_pro
    legate preprocess_data.py --output_file mnist_preprocessed.npz
    """

@bash_app
def install_flexflow(stdout='install_flexflow.out', stderr='install_flexflow.err'):
    return f"""
    cd $FF_HOME/FlexFlow/build
    make install
    """

@bash_app
def train_model(stdout='train_model.out', stderr='train_model.err'):
    return f"""
    $FF_HOME/FlexFlow/build/flexflow_python $FF_HOME/parsl_pro/train_script.py -ll:py 1 -ll:gpu 1 -ll:fsize 20000 -ll:zsize 20000 --data_file $FF_HOME/parsl_pro/mnist_preprocessed.npz
    """


if __name__ == "__main__":
    try:
        # # 安装cuNumeric
        # install_cunumeric_task = install_cunumeric()
        # print("Waiting for cuNumeric installation to complete...")
        # install_cunumeric_task.result()  # 等待任务完成
        # print("cuNumeric installation completed.")

        # 预处理数据
        preprocess_data_task = preprocess_data()
        print("Waiting for data preprocessing to complete...")
        preprocess_data_task.result()  # 等待任务完成
        print("Data preprocessing completed.")

        # 安装FlexFlow
        install_flexflow_task = install_flexflow()
        print("Waiting for FlexFlow installation to complete...")
        install_flexflow_task.result()  # 等待任务完成
        print("FlexFlow installation completed.")

        # 训练模型
        train_model_task = train_model()
        print("Waiting for model training to complete...")
        train_model_task.result()  # 等待任务完成
        print("Model training completed.")

    except Exception as e:
        print(f"Error during execution: {e}")
