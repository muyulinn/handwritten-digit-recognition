# 手写数字识别系统

我的第一个完整机器学习项目：使用 Tkinter 构建图形界面，使用 MNIST 数据集训练神经网络，实现单个手写数字的识别。

## 功能

- 手写识别：使用鼠标在画板上书写数字并识别。
- 图片识别：选择本地图片，预览并识别。
- 显示预测数字及模型预测概率。
- 清空画板并重新书写。

## 技术与模型

- Python 3.11
- Tkinter：图形界面
- Pillow：图片处理
- NumPy：像素数组处理
- scikit-learn：神经网络训练与预测
- joblib：模型保存与加载

模型使用一个包含 128 个神经元的隐藏层，输入为 784 个归一化像素特征，输出为数字 0～9。

MNIST 训练集包含 60,000 张图片，其中 10% 留作验证集。模型在独立的 10,000 张测试图片上的准确率为 97.75%。

## 项目结构

- `main.py`：图形界面与识别功能
- `preprocessing.py`：图片预处理
- `train_model.py`：下载数据、训练、评估及保存模型
- `requirements.txt`：依赖清单
- `models/digit_model.joblib`：训练好的模型
- `check_env.py`：开发时使用的环境检查脚本

## 安装与运行

安装 Python 3.11，下载本项目并进入项目目录。

在 Windows PowerShell 中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

也可以在 PyCharm 中打开项目，选择虚拟环境解释器，安装依赖后运行 `main.py`。

## 重新训练模型

仓库已包含训练好的模型，正常使用时不需要重新训练。

如需重新训练，执行：

```powershell
.\.venv\Scripts\python.exe train_model.py
```

首次运行会尝试下载 MNIST 数据集。如果下载失败，可以手动下载以下文件，保存为项目目录下的 `data/mnist.npz`，再重新运行训练脚本：

[下载 MNIST 数据文件](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz)

训练完成后会覆盖 `models/digit_model.joblib`。

## 使用范围与限制

- 每次识别一个数字，建议背景干净、四周留有空白。
- 支持简单的白底黑字和黑底白字图片。
- 部分个人手写数字可能误识别，测试中发现 4、5、9 比较容易混淆。
- MNIST 测试集准确率不代表个人手写图片具有相同准确率。
- 界面显示的预测概率不等于实际识别正确率。
- 当前不支持多位数字、文字或复杂场景识别。

## 数据来源

本项目使用 MNIST 手写数字数据集，未将数据集上传到仓库。

[MNIST 数据及加载说明](https://keras.io/api/datasets/mnist/)