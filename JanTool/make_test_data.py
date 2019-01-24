import os
import csv
import cv2
import shutil
#root directory
root = '/home/bong6/data/mrcnn_cer/stage1_train1/images'
csv_root = '/home/bong6/data/csv/output.csv'
mask_data_dir = '/home/bong6/data/mrcnn_cer/stage1_train1/mask'
result_dir = '/home/bong6/data/mrcnn_cer/stage1_train1/val2/images'

image_dict = {}
mask_dict = {}

#import image file
for (path, dir, files) in os.walk(root):
    for filename in files:
        ext = os.path.splitext(filename)[1]
        if ext == '.png' or ext == '.jpg':
            image_path = os.path.join(path, filename)
            image_dict[filename] = image_path





for (path, dir, files) in os.walk(mask_data_dir):
    for filename in files:
        ext = os.path.splitext(filename)[1]
        if ext == '.jpg' or ext == '.png':
            mask_path = os.path.join(path, filename)
            # key: filename, value: image path
            mask_dict[filename] = mask_path


print(mask_dict[filename])
for dirName, subdirList, fileList in os.walk(root):
    for filename in fileList:
        file_path = os.path.join(dirName, filename)
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        result_dir = os.path.join(result_dir)
        print(file_path)
        if file_path in mask_dict[filename]:

            if 'Type_1' in file_path:
                type1_result = os.path.join(result_dir,'Type_1')
                shutil.move(file_path, type1_result)
                print('move file type_1')
            if 'Type_2' in file_path:
                type2_result = os.path.join(result_dir, 'Type_2')
                shutil.move(file_path, type2_result)
                print('move file type_2')
            if 'Type_3' in file_path:
                type3_result = os.path.join(result_dir, 'Type_3')
                shutil.move(file_path, type3_result)
                print('move file type_3')


