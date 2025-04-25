import cv2
import numpy as np
import matplotlib.pyplot as plt


def image_read(path):
    """
    读取图片
    :param path: 图片路径
    :return image: 读取的图片
    """
    image = cv2.imread(path)
    if image is None:
        raise ValueError("图片路径错误，无法读取图片")
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


def image_select(image, scale_percent=50):
    """
    选择感兴趣的区域ROI
    :param image: 要选择ROI的图片
    :param scale_percent: 缩放百分比，默认为50%
    :return image_roi: 选择的ROI图片
    """
    # 获取图像的宽度和高度
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    # 缩放图像
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    x, y, w, h = cv2.selectROI("Select ROI", resized_image, False, False)
    if w == 0 or h == 0:
        raise ValueError("ROI选区无效，请重新选择")
    cv2.destroyWindow("Select ROI")
    image_roi = image[int(y * 100 / scale_percent):int((y + h) * 100 / scale_percent),
                      int(x * 100 / scale_percent):int((x + w) * 100 / scale_percent)]
    return image_roi


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
    _, img_thresh = cv2.threshold(img_median_blur, 250, 255, cv2.THRESH_BINARY)

    imgBinInv = cv2.bitwise_not(img_thresh)  # 二值图像的补集
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 构造 3×3 十字形结构元
    F = np.zeros(img_thresh.shape, np.uint8)  # 构建阵列 F，并写入 BinInv 的边界值
    F[:, 0] = imgBinInv[:, 0]
    F[:, -1] = imgBinInv[:, -1]
    F[0, :] = imgBinInv[0, :]
    F[-1, :] = imgBinInv[-1, :]

    # 循环迭代：对 F 进行膨胀，膨胀结果与 BinInv 进行 AND 操作
    Flast = F.copy()
    for i in range(1000):
        F_dilation = cv2.dilate(F, kernel)
        F = cv2.bitwise_and(F_dilation, imgBinInv)
        if (F == Flast).all():
            break  # 结束迭代算法
        else:
            Flast = F.copy()
        if i == 100: imgF100 = F  # 中间结果

    print("iter ={}".format(i))  # 迭代次数



    # img_open = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, np.ones((25,25), dtype=np.uint8))

    return F


def image_contours(image):
    """
    返回图像中的最外层轮廓
    :param image: 输入的单通道二值图像（numpy数组格式）
    :return: 最外层轮廓列表
    """
    if len(image.shape) > 2:
        raise ValueError("输入图像必须为单通道二值图")

    contours, _ = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return contours


def plot_histogram(image, contours):
    """
    绘制每个轮廓的颜色直方图
    :param image: 原始彩色图像
    :param contours: 轮廓列表
    """
    plt.figure(figsize=(16, 9))
    for i, contour in enumerate(contours):
        # 创建掩模
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)
        # 计算颜色直方图
        hist_b = cv2.calcHist([image], [0], mask, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], mask, [256], [0, 256])
        hist_r = cv2.calcHist([image], [2], mask, [256], [0, 256])
        # 归一化直方图
        cv2.normalize(hist_b, hist_b)
        cv2.normalize(hist_g, hist_g)
        cv2.normalize(hist_r, hist_r)
        # 扩展直方图的形状以便绘制
        hist_b_flat = hist_b.flatten()
        hist_g_flat = hist_g.flatten()
        hist_r_flat = hist_r.flatten()
        # 绘制直方图
        ax = plt.subplot(len(contours), 1, i + 1)
        ax.plot(hist_b_flat, color='b', label='Blue')
        ax.plot(hist_g_flat, color='g', label='Green')
        ax.plot(hist_r_flat, color='r', label='Red')
        ax.set_title(f'Histogram of Contour {i + 1}')
        ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    image_original = image_read("./data/image_old/green heart 2.bmp")
    image_original_copy = image_original.copy()

    image_processed = image_process(image_original)
    image_show("image_processed", image_processed)

    contours = image_contours(image_processed)
    if contours:
        cv2.drawContours(image_original_copy, contours, -1, (0, 0, 255), 3)

    #image_show("image_original_copy",image_original_copy)


# 调用 main 函数
if __name__ == "__main__":
    main()