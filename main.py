import os
import cv2
import numpy as np

def image_read(path):
    """
    读取图片
    :param path: 图片路径
    :return image: 读取的图片
    """
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"图片路径错误，无法读取图片: {path}")
    return image

def image_show(name, image, scale_percent=50):
    """
    显示图片
    :param name: 窗口名称
    :param image: 要显示的图片
    :param scale_percent: 缩放百分比，默认为50%
    """
    # 获取图像的宽度和高度
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    # 缩放图像
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imshow(name, resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def image_process(image):
    """
    对输入图像进行灰度化和阈值分割处理
    :param image: 输入的彩色或灰度图像（numpy数组格式）
    :return: 处理后的二值图像（numpy数组格式）
    """
    if len(image.shape) == 2:
        image_gry = image.copy()
    else:
        image_gry = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_gauss = cv2.GaussianBlur(image_gry, (5, 5), sigmaX=1)
    img_joint_bi_filter = cv2.ximgproc.jointBilateralFilter(img_gauss, image_gry, d=5, sigmaColor=10, sigmaSpace=5)
    img_median_blur = cv2.medianBlur(img_joint_bi_filter, 7)

    _, img_thresh = cv2.threshold(img_median_blur, 254, 255, cv2.THRESH_BINARY)
    img_open = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, np.ones((19, 19), dtype=np.uint8))


    # imgOpenGrad = cv2.morphologyEx(img_open, cv2.MORPH_GRADIENT, np.ones((3,3), dtype=np.uint8))  # 形态学梯度


    return img_open

def image_contours(image):
    """
    返回图像中的最外层轮廓
    :param image: 输入的单通道二值图像（numpy数组格式）
    :return: 最外层轮廓列表
    """
    if len(image.shape) > 2:
        raise ValueError("输入图像必须为单通道二值图")

    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[1:]
    if not contours:
        return None
    return contours

def image_process_workflow(file_path,filename):
    image_original = image_read(file_path)
    image_original_copy = image_original.copy()
    imgBin = image_process(image_original_copy)



    # 显示原图和处理后的图像
    image_show(f"Processed - {filename}", imgBin)

    # cv2.imwrite(f"./processed/{filename}", image_processed)

def process_folder(folder_path):
    """
    处理指定文件夹中的所有图像
    :param folder_path: 文件夹路径
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否为图像
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 构建文件完整路径
            file_path = os.path.join(folder_path, filename)
            try:
                image_process_workflow(file_path,filename)
            except Exception as e:
                print(f"处理文件 {file_path} 时发生错误: {e}")

def main():
    folder_path = "./data/image_old/"  # 指定文件夹路径
    process_folder(folder_path)
# 调用 main 函数
if __name__ == "__main__":
    main()