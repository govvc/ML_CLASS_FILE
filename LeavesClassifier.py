from PIL import Image
import os
import numpy as np

image_width = 256
image_height = 500

label = np.zeros(831)
data_ = np.zeros((831, 500, 256))


def fixed_size(filePath, savePath):
    im = Image.open(filePath).convert('L')
    out = im.resize((image_width, image_height), Image.ANTIALIAS)
    out.save(savePath)


def walk_files(openPath):
    train = []
    c = -1
    i = 0
    for root, dirs, files in os.walk(openPath):
        for file in files:
            file_path = os.path.join(root, file)

            img = Image.open(file_path).convert('L').resize((image_width, image_height), Image.ANTIALIAS)
            img = np.array(img, dtype=np.float32)
            data = np.zeros((500, 256))
            data[0:, 0:] = img
            data_[i, :, :] = data[0:, 0:]

            label[i] = c
            i += 1
        c += 1

    print(label)
    train.append(label)
    train.append(data_)


    return train


if __name__ == '__main__':
    path = r'C:\Classify\train_831'
    walk_files(path)
