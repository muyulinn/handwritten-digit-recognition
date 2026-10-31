import numpy as np
from PIL import Image, ImageOps

# 此函数用于接收一张 Pillow 图片对象
def prepare_digit(image):
    # 转为灰度图片，L指灰度
    gray = image.convert('L')

    # 灰度图片转成 NumPy 数组
    pixels = np.asarray(gray)
    # 取出图片四条边的像素并拼接
    # 作用为判断图片背景是黑还是白
    border = np.concatenate((
        pixels[0, :],
        pixels[-1, :],
        pixels[:, 0],
        pixels[:, -1],
    ))
    # 如果边缘平均亮度较高，就假设背景偏白，执行反色
    if border.mean() > 127:
        gray = ImageOps.invert(gray)

    # 找到笔迹
    gray = ImageOps.autocontrast(gray)
    # 区分笔迹和背景的操作
    mask = gray.point(lambda value: 255 if value > 40 else 0)
    bounding_box = mask.getbbox()

    # 找不到笔迹时
    if bounding_box is None:
        raise ValueError("图片中没有检测到清晰的数字笔迹。")

    # 裁剪、缩放笔迹并放到中央
    digit = gray.crop(bounding_box)
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # 创建28*28的黑色背景，把笔记放上去
    normalized = Image.new("L",(28, 28), color=0)
    left = (28 - digit.width) // 2
    top = (28 - digit.height) // 2
    normalized.paste(digit, (left, top))

    # 转换成模型输入并返回
    features = np.asarray(normalized, dtype=np.float32)
    features = features.reshape(1, 784) / 255.0

    return features, normalized

