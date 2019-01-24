import os
import shutil
root ='/home/bong6/data/mrcnn_cer/crop_image'
for dirName, subdirList, fileList in os.walk(root):
     for filename in fileList:
         file_path = os.path.join(dirName, filename)
         if 'a' in filename:
             shutil.move(file_path, os.path.join(root,"Type_1"))
         elif 'b' in filename:
             shutil.move(file_path, os.path.join(root, "Type_2"))

         else:
             shutil.move(file_path, os.path.join(root, "Type_3"))


