from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import os
from osgeo import gdal
from shutil import copy, rmtree
import random
import re

"""
处理数据集
"""

'''
裁剪图像大小为256×256
'''
def cutImage():
    dataset_input = "D:/太原科技大学/数据集/LEVIR-CD/before_val/B/"
    dataset_output = "D:/太原科技大学/数据集/LEVIR-CD/val/B/"
    imagelist =os.listdir(dataset_input)
    for image in imagelist:
        img = Image.open(dataset_input + image)
        img_num = int((image.split('_')[1]).split('.')[0])
        cnt = 1
        for i in range(4):
            for j in range(4):
                x, y, w, h = i*256, j*256, 256, 256
                region = img.crop((x, y, x + w, y + h))
                num = (img_num - 1) * 16 + cnt
                region.save(dataset_output + 'val_' + str(num) + '.png')
                cnt += 1

'''
裁剪TIF图像大小为256×256
'''
def cutTIFImage():
    dataset_input = "D:/太原科技大学/数据集/WHU Building数据集/change_label.tif"
    dataset_output = "D:/太原科技大学/数据集/WHU Building数据集/label_cut/"
    img = Image.open(dataset_input)
    cnt = 1
    for i in range(126):
        for j in range(59):
            x, y, w, h = i*256, j*256, 256, 256
            region = img.crop((x, y, x + w, y + h))
            region.save(dataset_output + 'label_' + str(cnt) + '.tif')
            cnt += 1

'''
TIF→PNG
'''
def TIFToPNG():
    file_path = "D:/太原科技大学/数据集/WHU Building数据集/after_1.tif"
    ds = gdal.Open(file_path)
    driver = gdal.GetDriverByName('PNG')
    dst_ds = driver.CreateCopy('D:/太原科技大学/数据集/WHU Building数据集/example.png', ds)
    dst_ds = None
    src_ds = None

'''
划分数据集→train，val，test
'''
def splitData():
    # 保证随机可复现
    random.seed(0)
    # 将数据集中10%的数据划分到验证集中
    split_val_rate = 0.1
    # 将数据集中20%的数据划分到测试集中
    split_test_rate = 0.2

    # 保存训练集的文件夹
    train_root = "D:/太原科技大学/数据集/WHU Building数据集/train/A/"

    # 保存验证集的文件夹
    val_root = "D:/太原科技大学/数据集/WHU Building数据集/val/A/"

    # 保存测试集的文件夹
    test_root = "D:/太原科技大学/数据集/WHU Building数据集/test/A/"

    dataset_path = "D:/太原科技大学/数据集/WHU Building数据集/before_cut/"
    datalist = os.listdir(dataset_path)
    num = len(datalist)
    # 随机采样验证集的索引
    eval_index = random.sample(datalist, k=int(num * split_val_rate))
    # 随机采样测试集的索引
    etest_index = random.sample(datalist, k=int(num * split_test_rate))
    for index, image in enumerate(datalist):
        if image in eval_index:
            # 将分配至验证集中的文件复制到相应目录
            image_path = dataset_path + image
            copy(image_path, val_root)
        elif image in etest_index:
            # 将分配至测试集中的文件复制到相应目录
            image_path = dataset_path + image
            copy(image_path, test_root)
        else:
            # 将分配至训练集中的文件复制到相应目录
            image_path = dataset_path + image
            copy(image_path, train_root)

'''
复制图像
'''
def copyImage():
    test_input_path = "D:/太原科技大学/数据集/WHU Building数据集/train/A/"
    test_output_path = "D:/太原科技大学/数据集/WHU Building数据集/train/B/"
    after_path = "D:/太原科技大学/数据集/WHU Building数据集/after_cut/"
    imglist = os.listdir(test_input_path)
    for img in imglist:
        num = int((img.split('_')[1]).split('.')[0])
        copy_path = after_path + 'after_' + str(num) + '.tif'
        new_path = test_output_path
        copy(copy_path, new_path)

'''
重命名
'''
#  将元素中的数字转换为int后再排序
def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s
#  将元素中的数字转换为int后再排序
def str2int(v_str):
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

#  以分割后的list为单位进行排序
def sort_humanly(v_list):
    return sorted(v_list, key=str2int)

def rename():
    basepath = "D:/太原科技大学/数据集/WHU Building数据集/val/B/"
    filelist = os.listdir(basepath)
    total_num = len(filelist)  # 获取文件夹内所有文件个数
    filelist = sort_humanly(filelist)
    i = 1  # 表示文件的命名是从1开始的
    for item in filelist:
        if item.endswith('.png'):
            # 初始的图片的格式为png格式的
            src = os.path.join(os.path.abspath(basepath), item)
            dst = os.path.join(os.path.abspath(basepath), 'val_' + str(i) + '.png')
            try:
                os.rename(src, dst)
                print('converting %s to %s ...' % (src, dst))
                i = i + 1
            except:
                continue
    print('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
    # cutImage()
    # cutTIFImage()
    # TIFToPNG()
    # splitData()
    # copyImage()
    rename()