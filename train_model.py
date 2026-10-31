from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
import joblib
# MLPClassifier：创建神经网络分类器。
from sklearn.neural_network import MLPClassifier
# accuracy_score：计算预测正确的比例
from sklearn.metrics import accuracy_score


def main():
    # 获取当前脚本所在的文件夹（E:\吴恩达lab\handwritten_digit_recognition）
    project_dir = Path(__file__).resolve().parent
    # 路径拼接(E:\吴恩达lab\handwritten_digit_recognition\data)
    data_dir = project_dir / "data"
    # 如果文件夹存在，也不报错
    data_dir.mkdir(exist_ok=True)

    # (E:\吴恩达lab\handwritten_digit_recognition\data\mnist.npz)
    dataset_path = data_dir / "mnist.npz"

    # 初次安装
    if not dataset_path.exists():
        print("首次运行，正在下载 MNIST 数据，请稍后...", flush=True)
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

        # （E:\吴恩达lab\handwritten_digit_recognition\data\mnist.download）
        temporary_path = data_dir / "mnist.download"

        urlretrieve(url,temporary_path)
        temporary_path.replace(dataset_path)
        print("下载完成。")
    else:
        print("发现本地数据文件，跳过下载。")

    with np.load(dataset_path, allow_pickle=False) as dataset:
        x_train = dataset["x_train"]
        y_train = dataset["y_train"]
        x_test = dataset["x_test"]
        y_test = dataset["y_test"]

    print("训练图片形状：", x_train.shape)
    print("训练标签形状：", y_train.shape)
    print("测试图片形状：", x_test.shape)
    print("测试标签形状：", y_test.shape)
    print("像素最小值：", x_train.min())
    print("像素最大值：", x_train.max())
    print("第一张训练图片的正确数字：", y_train[0])

    # 添加保存样本图片的代码
    sample_path = data_dir / "sample_digit.png"
    # 把第一张训练图片的 NumPy 数组转换成 Pillow 图片
    sample_image = Image.fromarray(x_train[0])
    # 图片从 28 × 28 放大为 280 × 280，方便观察
    sample_image = sample_image.resize(
        (280,280),
        resample=Image.Resampling.NEAREST
    )
    # 把图片写入磁盘
    sample_image.save(sample_path)
    print("样本图片已保存到：", sample_path)

    # 将每张二维图片展开为一行，并将像素缩放到0-1
    # astype(np.float32):把原来的整数像素转换为 32 位浮点数
    x_train = x_train.reshape(-1,784).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1,784).astype(np.float32) / 255.0

    print("转换后的训练输入形状:", x_train.shape)
    print("转换后的测试输入形状:", x_test.shape)
    print("转换后的数据类型：", x_train.dtype)
    print("转换后的像素范围：", x_train.min(), x_train.max())

    # 创建模型
    # MLPClassifier是一个神经网络分类器，解决二分类、多分类、多标签问题
    model = MLPClassifier(
        # 使用一个含有128神经元的隐藏层，有几层就会写几个参数
        hidden_layer_sizes=(128,),
        # ReLU函数激活
        activation="relu",
        # Adam方法更新模型参数
        solver="adam",
        # 每批次256个样本
        batch_size=256,
        # 初始学习率
        learning_rate_init=0.001,
        # 训练轮数
        max_iter=40,
        # 效果没有明显改善时提前停止
        early_stopping=True,
        # 选10%的数据作留出验证集，不参与更新参数
        validation_fraction=0.1,
        # 连续训练如果没有足够改善就停止
        n_iter_no_change=5,
        # 固定随机数种子即便复现
        random_state=42,
        # 在运行窗口显示验证过程
        verbose=True
    )

    # 开始训练
    print("开始训练模型...", flush=True)
    # x_train是图片像素
    # y_train是图片对应的正确数字
    model.fit(x_train, y_train)
    print("训练完成。")

    # 测试模型表现
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test,predictions)

    print(f"测试集准确集:{accuracy:.2%}")
    print("前 10 张测试图片的预测：", predictions[:10])
    print("前 10 张测试图片的答案：", y_test[:10])

    # 保存训练好的模型及其参数
    # 以后启动界面时加载它，就可以识别图片
    model_dir = project_dir / "models"
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "digit_model.joblib"
    joblib.dump(model, model_path) # 保存模型
    # 加载模型用 model = joblib.load(model_path)
    print("模型已保存到：", model_path)



if __name__ == "__main__":
    main()