# 导入界面工具 tkinter
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps, ImageDraw
# 计算模型文件的位置
from pathlib import Path
# 加载保存的模型
import joblib
# 引入预处理函数
from preprocessing import prepare_digit

def main():
    # 创建程序的主窗口
    window = tk.Tk()

    # 设置标题和大小
    window.title("手写数字识别系统")
    window.geometry("900x600")
    window.minsize(700,500)

    # 定位并加载训练好的模型
    project_dir = Path(__file__).resolve().parent
    model_path = project_dir / "models" / "digit_model.joblib"

    if not  model_path.is_file():
        messagebox.showerror(
            "模型不存在",
            f"没有找到模型文件：\n{model_path}\n请先运行 train_model.py。",
            parent=window
        )
        window.destroy()
        return

    # 加载模型到内存中
    try:
        model = joblib.load(model_path)
    except Exception as error:
        messagebox.showerror(
            "模型加载失败",
            f"无法加载模型: \n{error}",
            parent=window
        )
        window.destroy()
        return

    print("模型加载成功。")
    print("模型输入特征数：", model.n_features_in_)
    print("模型可以识别的类别：", model.classes_)

    # 创建窗口内的文字标题
    title_label = ttk.Label(window,
                            text="手写数字识别系统",
                            font=("Microsoft YaHei",20)
    )
    title_label.pack(pady=15)

    # 管理多个选项卡页面的容器
    notebook = ttk.Notebook(window)
    notebook.pack(fill="both",
                  expand=True,
                  padx=20,
                  pady=(0,20))

    # 创建选项卡容器
    handwriting_tab = ttk.Frame(notebook) # 以后画板放进 handwriting_tab
    image_tab = ttk.Frame(notebook) # 图片相关组件放进 image_tab

    # 页面加入选项卡
    notebook.add(handwriting_tab, text="手写识别")
    notebook.add(image_tab, text="图片识别")

    # 提示文字和画板
    handwriting_hint = ttk.Label(
        handwriting_tab,
        text="请在黑色画板中写一个数字")
    handwriting_hint.pack(pady=10)

    canvas = tk.Canvas(
        handwriting_tab,
        width=280,
        height=280,
        bg="black",
        highlightthickness=0 # 去掉画布外围的焦点高亮边框(不重要)
    )
    canvas.pack(pady=10)

    # 在内存中保存与画板对应的笔迹
    # 通过 handwriting_draw 绘画，会直接
    # 修改 handwriting_image 的像素。
    handwriting_image = Image.new("L", (280, 280),color=0)
    handwriting_draw = ImageDraw.Draw(handwriting_image)


    # 绘画函数
    # 记录鼠标上一次的位置
    last_x = None
    last_y = None
    brush_width = 18

    def start_draw(event):
        nonlocal last_x, last_y
        last_x = event.x
        last_y = event.y
        handwriting_result.configure(text="识别结果：等待识别")

        radius = brush_width / 2
        canvas.create_oval(
            event.x - radius,
            event.y - radius,
            event.x + radius,
            event.y + radius,
            fill="white",
            outline="white",
            width=0
        )

        # 同步起点
        handwriting_draw.ellipse(
            (
                event.x - radius,
                event.y - radius,
                event.x + radius,
                event.y + radius
            ),
            fill=255
        )

    def draw(event):
        nonlocal last_x, last_y
        if last_x is None or last_y is None:
            return

        canvas.create_line(
            last_x, last_y,
            event.x, event.y,
            fill="white",
            width=brush_width,
            capstyle=tk.ROUND
        )

        handwriting_draw.line(
            (last_x, last_y, event.x, event.y),
            fill=255,
            width=brush_width
        )

        radius = brush_width / 2
        handwriting_draw.ellipse(
            (
                event.x - radius,
                event.y - radius,
                event.x + radius,
                event.y + radius
            ),
            fill=255
        )

        last_x = event.x
        last_y = event.y

    def stop_draw(event):
        nonlocal last_x, last_y
        last_x = None
        last_y = None

    canvas.bind("<Button-1>", start_draw)
    canvas.bind("<B1-Motion>", draw)
    canvas.bind("<ButtonRelease-1>", stop_draw)



    # 清空功能
    def clear_canvas():
        # 删除画布的全部图形
        nonlocal last_x, last_y
        canvas.delete("all")

        handwriting_image.paste(0, (0, 0, 280, 280))

        last_x = None
        last_y = None
        handwriting_result.configure(text="识别结果：等待识别")

    clear_button = ttk.Button(
        handwriting_tab,
        text="清空画板",
        command=clear_canvas # 点按钮用清空功能
    )

    clear_button.pack(pady=10)

    def recognize_handwriting():
        try:
            features, _ = prepare_digit(handwriting_image)
        except ValueError:
            handwriting_result.configure(text="请先在画板中写一个数字")
            return

        probabilities = model.predict_proba(features)[0]
        best_index = probabilities.argmax()
        digit = int(model.classes_[best_index])
        probability = float(probabilities[best_index])

        handwriting_result.configure(
            text=f"识别结果：{digit}    模型预测概率：{probability:.1%}"
        )

    recognize_handwriting_button = ttk.Button(
        handwriting_tab,
        text="识别手写数字",
        command=recognize_handwriting
    )
    recognize_handwriting_button.pack(pady=5)

    handwriting_result = ttk.Label(
        handwriting_tab,
        text="识别结果：等待识别"
    )
    handwriting_result.pack(pady=5)

    # 图片识别页面
    image_hint = ttk.Label(
        image_tab,
        text="选择一张包含一个数字的图片")
    image_hint.pack(pady=10)

    # 图片状态变量
    selected_image = None
    preview_photo = None

    def choose_image():
        nonlocal selected_image, preview_photo

        # 打开文件选择框
        file_path = filedialog.askopenfilename(
            parent=window,
            title="choose image",
            filetypes=[
                ("图片文件","*.png *.jpg *.jpeg *.bmp"),
                ("所有文件","*.*")
            ]
        )

        # 处理取消操作
        if not file_path:
            return

        try:
            # 打开并处理图片
            with Image.open(file_path) as source:
                loaded_image = ImageOps.exif_transpose(source).convert("RGB")

            # 创建预览
            preview = loaded_image.copy()
            preview.thumbnail((280, 280))
            new_photo = ImageTk.PhotoImage(preview, master=window)
        except (OSError, ValueError):
            messagebox.showerror(
                "打开失败",
                "无法读取这张图片，请选择有效的 PNG、JPG 或 BMP 图片。",
                parent=window
            )
            return

        # 更新界面
        selected_image = loaded_image
        preview_photo = new_photo
        image_preview.configure(image=preview_photo, text="")
        image_status.configure(text=f"已选择：{file_path}")

        image_result.configure(text="识别结果：等待识别")

    choose_button = ttk.Button(
        image_tab,
        text="选择图片",
        command=choose_image
    )
    choose_button.pack(pady=10)

    image_preview = ttk.Label(
        image_tab,
        text="尚未选择图片",
        anchor="center"
    )
    image_preview.pack(pady=10)

    image_status = ttk.Label(
        image_tab,
        text="请选择一张图片",
        wraplength=600
    )
    image_status.pack(pady=10)

    # 识别函数
    def recognize_image():
        if selected_image is None:
            messagebox.showinfo(
                "尚未选择图片",
                "请先点击“选择图片”，再进行识别。",
                parent=window
            )
            return

        try:
            features, normalized = prepare_digit(selected_image)
        except ValueError as error:
            image_result.configure(text="识别结果：无法识别")
            messagebox.showwarning(
                "图片无法识别",
                str(error),
                parent=window
            )
            return

        # 获取该图片属于各个数字的概率
        # 用[0]因为每次只传一张图
        probabilities = model.predict_proba(features)[0]
        # 获取最大概率的数字
        best_index = probabilities.argmax()
        digit = int(model.classes_[best_index])
        # 获取最大概率的数字对应的概率
        probability = probabilities[best_index]

        image_result.configure(
            text=f"识别结果：{digit}    模型预测概率：{probability:.1%}",
        )

    recognize_image_button = ttk.Button(
        image_tab,
        text="识别图片",
        command=recognize_image
    )
    recognize_image_button.pack(pady=5)

    image_result = ttk.Label(
        image_tab,
        text="识别结果：等待识别",
        font=("Microsoft YaHei", 14)
    )
    image_result.pack(pady=5)


    # 启动界面的事件循环
    window.mainloop()


if __name__ == '__main__':
    main()