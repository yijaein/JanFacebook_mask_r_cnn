import os
import argparse
import shutil
parser = argparse.ArgumentParser()
parser.add_argument("--label_path",default='/home/bong6/data/mrcnn_cer/classificationdataset_512/train',help="label_path")
parser.add_argument("--crop_image_path", default='/home/bong6/lib/robin_cer/results/nucleus/submit_20190106T234614', help='crop image path ')
parser.add_argument('--result_dir',default='/home/bong6/lib/robin_cer/results/crop_classified')
args = parser.parse_args()

label_path = args.label_path
label_path = os.path.expanduser(label_path)

crop_path = args.crop_image_path
crop_path = os.path.expanduser(crop_path)
label_dict = {}
crop_dict = {}


if not os.path.isdir(label_path):
    print("can not import label")
else:
    for (path, dir, files) in os.walk(label_path):
        for filename in files:
            ext = os.path.splitext(filename)[1]
            if ext == '.png' or ext == '.jpg':
                label_path = os.path.join(path, filename)
                label_dict[filename.replace(ext, '')] = label_path

                #print(label_dict)





    for (path, dir, files) in os.walk(crop_path):
        for filename in files:
            ext = os.path.splitext(filename)[1]
            if ext == '.png' or ext == '.jpg':
                crop_path = os.path.join(path, filename)
                crop_dict[filename.replace(ext, '')] = crop_path


    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir)

    for id, image_path in label_dict.items():
        if not id in crop_dict:
            continue
        if "Type_1" in image_path:
            copy_path = os.path.join(args.result_dir, "Type_1")



        if "Type_2" in image_path:
            copy_path = os.path.join(args.result_dir, "Type_2")

        if "Type_3" in image_path:
            copy_path = os.path.join(args.result_dir, "Type_3")

        if not os.path.exists(copy_path):
            os.makedirs(copy_path)
        shutil.copy(image_path, copy_path )

        #shutil.copy(image_path, args.result_dir)
        #print(label_path)










