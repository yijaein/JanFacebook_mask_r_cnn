import os
import shutil

import cv2
import numpy as np

from Tools.dicom_physical_size import object_wh, read_dicom_pixel_size_csv, name_dict

'''
result_CropKidney
  * Prec@1 93.843 at Epoch 71 
  * auc@1 0.964 
  acc of label 0: 95.053% 
  acc of label 1: 86.245% 
/media/bong07/895GB/result/result_KorNK/result_CropKidney/checkpoint_72.pth


result_CropKidneyShape
  * Prec@1 92.335 at Epoch 77 
  * auc@1 0.952 
  acc of label 0: 93.928% 
  acc of label 1: 82.342% 
/media/bong07/895GB/result/result_KorNK/result_CropKidneyShape/checkpoint_78.pth


result_CropKidneyShapeWithColor
  * Prec@1 92.310 at Epoch 69 
  * auc@1 0.946 
  acc of label 0: 94.254% 
  acc of label 1: 80.112% 
/media/bong07/895GB/result/result_KorNK/result_CropKidneyShapeWithColor/checkpoint_70.pth


앙상블 정확도 94.32805314256515

'''


def norm_path(path):
    path = os.path.normcase(path)
    path = os.path.normpath(path)
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    return path


def validation(evalutes_csv_list, ground_truth_csv):
    # norm path
    evalutes_csv_list = [norm_path(path) for path in evalutes_csv_list]
    ground_truth_csv = norm_path(ground_truth_csv) if ground_truth_csv else None

    # read
    evalutes_list = list()
    for evalutes_csv in evalutes_csv_list:
        evalutes = list()
        with open(evalutes_csv, 'rt') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                cols = line.strip('\r\n').split(',')
                for idx in range(len(cols)):
                    if idx > 0:
                        cols[idx] = float(cols[idx])
                evalutes.append(cols)
        print('rows', len(evalutes), evalutes_csv)
        evalutes_list.append(evalutes)

    evalutes_list = [sorted(l, key=lambda x: x[0]) for l in evalutes_list]

    # sum softmax
    sum_evalutes = evalutes_list[0].copy()
    for idx_eval, evalutes in enumerate(evalutes_list[1:]):
        for idx_row, row in enumerate(evalutes):
            for idx_col, col in enumerate(row):
                if idx_col == 0:
                    if sum_evalutes[idx_row][0] != col:
                        print('error different file')
                        exit()
                else:
                    sum_evalutes[idx_row][idx_col] += col

    # read ground truth
    gt_dict = dict()
    with open(ground_truth_csv, 'rt') as fgt:
        while True:
            line = fgt.readline()
            if not line:
                break
            name, class_idx = line.strip('\r\n').split(',')
            gt_dict[name] = int(class_idx)

    cnt_evalute = 0
    cnt_right_evalute = 0
    correct_list = list()
    incorrect_list = list()
    for evalute in sum_evalutes:
        name = evalute[0]
        values = evalute[1:]
        class_idx = values.index(max(values))

        cnt_evalute += 1

        if gt_dict[name] == class_idx:
            cnt_right_evalute += 1
            correct_list.append(name)
        else:
            incorrect_list.append([name, class_idx])

    acc = (cnt_right_evalute / len(sum_evalutes)) * 100
    print(cnt_right_evalute)
    print(len(sum_evalutes))
    print(cnt_evalute)
    print('정확도', acc)

    return correct_list, incorrect_list


def evalute(evalutes_csv_list):
    # norm path
    evalutes_csv_list = [norm_path(path) for path in evalutes_csv_list]

    # read
    evalutes_list = list()
    for evalutes_csv in evalutes_csv_list:
        evalutes = list()
        with open(evalutes_csv, 'rt') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                cols = line.strip('\r\n').split(',')
                for idx in range(len(cols)):
                    if idx > 0:
                        cols[idx] = float(cols[idx])
                evalutes.append(cols)
        print('rows', len(evalutes), evalutes_csv)
        evalutes_list.append(evalutes)

    evalutes_list = [sorted(l, key=lambda x: x[0]) for l in evalutes_list]

    # sum softmax
    sum_evalutes = evalutes_list[0].copy()
    for idx_eval, evalutes in enumerate(evalutes_list[1:]):
        for idx_row, row in enumerate(evalutes):
            for idx_col, col in enumerate(row):
                if idx_col == 0:
                    if sum_evalutes[idx_row][0] != col:
                        print('error different file')
                        exit()
                else:
                    sum_evalutes[idx_row][idx_col] += col

    cnt_evalute = 0
    kidney_list = list()
    non_kidney_list = list()
    for evalute in sum_evalutes:
        name = evalute[0]
        values = evalute[1:]
        class_idx = values.index(max(values))

        cnt_evalute += 1

        if class_idx == 1:
            kidney_list.append(name)
        else:
            non_kidney_list.append(name)

    print('cnt kidney', len(kidney_list))
    print('cnt non-kidney', len(non_kidney_list))
    print(len(sum_evalutes))
    print(cnt_evalute)

    return kidney_list, non_kidney_list


'''
지정한 이미지들의 size를 출력

output: 기기, 폴더, 파일, 진단, 긴쪽, 짧은쪽, 폴더내 신장 순서

diagnosis_csv header: Date, File, RecoredPatientID, RealPatientID, AccNo, Diagnosis, Excluded
dicom_csv     header: File, Manufacturer, PhysicalUnitsXDirection, PhysicalDeltaX, PhysicalUnitsYDirection, PhysicalDeltaY
'''


def get_kidney_size(image_name_list, seg_path, dicom_csv):
    dicom_info = read_dicom_pixel_size_csv(dicom_csv)

    # gather seg files
    seg_dict = dict()
    for (root, dirs, files) in os.walk(seg_path):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext.lower() not in ['.png', '.jpg']:
                continue
            seg_dict[name] = os.path.join(root, file)

    kidney_size = dict()

    for file_name in image_name_list:
        name = file_name.split('#')[0]
        if file_name not in seg_dict.keys():
            print('not found seg', name)
            continue
        if name not in dicom_info.keys():
            print('not found dicom_info', name)
            continue

        # compute Long cm, Short cm
        seg_img = cv2.imread(seg_dict[file_name], cv2.IMREAD_GRAYSCALE)
        _, long_px, short_px = object_wh(seg_img)
        PhysicalDeltaX, PhysicalDeltaY = dicom_info[name]['PhysicalDeltaX'], dicom_info[name]['PhysicalDeltaY']
        PhysicalDeltaX = float(PhysicalDeltaX) if PhysicalDeltaX != 'None' else 0.0
        PhysicalDeltaY = float(PhysicalDeltaY) if PhysicalDeltaY != 'None' else 0.0
        LongCM, ShortCM = long_px * PhysicalDeltaX, short_px * PhysicalDeltaY

        kidney_size[file_name] = [LongCM, ShortCM]
        # print(file_name, LongCM, ShortCM)

    return kidney_size


def save_kidney_size(kidney_size, result_csv):
    result_csv = norm_path(result_csv)
    result_path = os.path.split(result_csv)[0]
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(result_csv, 'wt') as fsave:
        for key in kidney_size.keys():
            file = key
            size = [str(size) for size in kidney_size[key]]
            line = ','.join([file, size[0], size[1]])
            fsave.write(line + '\n')


def make_compare_size(compute_kidney_size_csv, gt_kidney_size_csv, result_csv):
    result_csv = norm_path(result_csv)
    result_path = os.path.split(result_csv)[0]
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # File LongCM ShortCM
    # Folder File Diagnosis Pos Size

    compute_kidney = dict()
    with open(compute_kidney_size_csv) as f:
        while True:
            line = f.readline()
            if not line:
                break
            cols = line.strip('\r\n').split(',')
            compute_kidney[cols[0]] = float(cols[1])
    print('len compute_kidney', len(compute_kidney))

    gt_kidney = dict()
    with open(gt_kidney_size_csv) as f:
        f.readline()  # pass header
        while True:
            line = f.readline()
            if not line:
                break
            cols = line.strip('\r\n').split(',')
            if cols[4] == '':
                continue
            name = os.path.splitext(cols[1])[0]
            gt_kidney[name] = float(cols[4])
    print('len gt_kidney', len(gt_kidney))

    # make result csv
    valid_kidney_list = list()
    with open(result_csv, 'wt') as fsave:
        for key in compute_kidney.keys():
            name, size_order = key.split('#')
            compute_size = compute_kidney[key]

            # 최대 크기의 신장만 사용
            if size_order != '0':
                continue

            # 픽셀 사이즈가 있어서 실제 크기 계산된것만 사용
            if compute_size == 0:
                continue

            if name not in gt_kidney.keys():
                continue

            gt_size = gt_kidney[name]
            line = ','.join([name, str(gt_size), str(compute_size), str(abs(gt_size - compute_size))])
            fsave.write(line + '\n')

            valid_kidney_list.append([key, gt_size, compute_size])

    return valid_kidney_list


def make_compare_image(valid_kidney_list, result_path, ori_path, mask_path):
    result_path = norm_path(result_path)
    ori_path = norm_path(ori_path)
    mask_path = norm_path(mask_path)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    ori_dict = name_dict(ori_path)
    mask_dict = name_dict(mask_path)

    for key, gt_size, compute_size in valid_kidney_list:
        name = key.split('#')[0]
        mask_name = key
        img_ori_path = ori_dict[name]
        img_mask_path = mask_dict[mask_name]

        img = cv2.imread(img_ori_path, cv2.IMREAD_COLOR)
        overlay = cv2.imread(img_mask_path, cv2.IMREAD_COLOR)
        overlay[:, :, 0] = 0
        overlay[:, :, 1] = 0

        img_overlay = cv2.addWeighted(img, 1.0, overlay, 0.15, 0)

        diff_size = abs(gt_size - compute_size)
        size_group = '{:.1f}'.format((diff_size // 0.3) * 0.3)
        dst_path = os.path.join(result_path, '' + size_group)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        dst_file = os.path.join(dst_path, key + '.png')

        img_result = np.hstack((img_overlay, img))
        cv2.imwrite(dst_file, img_result)


def copy_image(name_list, result_path, ori_path, mask_path):
    result_path = norm_path(result_path)
    ori_path = norm_path(ori_path)
    mask_path = norm_path(mask_path)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    ori_dict = name_dict(ori_path)
    mask_dict = name_dict(mask_path)

    for key, class_idx in name_list:
        name = key.split('#')[0]
        mask_name = key
        img_ori_path = ori_dict[name]
        img_mask_path = mask_dict[mask_name]

        img = cv2.imread(img_ori_path, cv2.IMREAD_COLOR)
        overlay = cv2.imread(img_mask_path, cv2.IMREAD_COLOR)
        overlay[:, :, 0] = 0
        overlay[:, :, 1] = 0

        img_overlay = cv2.addWeighted(img, 1.0, overlay, 0.15, 0)

        dst_path = os.path.join(result_path, "kidney" if class_idx == 1 else "non-kidney")
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        dst_file = os.path.join(dst_path, key + '.png')

        img_result = np.hstack((img_overlay, img))
        cv2.imwrite(dst_file, img_result)


if __name__ == '__main__':
    # classification kidney
    evalutes_csv_list = ['/home/bong07/lib/robin_yonsei2/results_mrcnn/Seg_aug2_20181219T111320/M2_result/result_CropKidney/evaluate.csv',
                         '/home/bong07/lib/robin_yonsei2/results_mrcnn/Seg_aug2_20181219T111320/M2_result/result_CropKidneyShape/evaluate.csv',
                         '/home/bong07/lib/robin_yonsei2/results_mrcnn/Seg_aug2_20181219T111320/M2_result/result_CropKidneyShapeWithColor/evaluate.csv']
    ground_truth_csv = '/home/bong07/lib/robin_yonsei/result_KorNK/gt.csv'

    do_evalute = True

    if do_evalute:
        kidney_list, non_kidney_list = evalute(evalutes_csv_list)

        # copy kidney seg
        seg_path = norm_path('/home/bong07/lib/robin_yonsei2/results_mrcnn/Seg_aug2_20181219T111320/SegKidney_MRCNN')
        result_path = norm_path('/home/bong07/lib/robin_yonsei2/results_mrcnn/Seg_aug2_20181219T111320/M2.5_result')

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        for (root, dirs, files) in os.walk(seg_path):
            for file in files:
                name, ext = os.path.splitext(file)
                if ext.lower() not in ['.jpg', '.png']:
                    continue
                if name not in kidney_list:
                    continue
                print(name)

                dst = os.path.join(result_path, file.split('#')[0] + '.png')
                shutil.copy(os.path.join(root, file), dst)
    else:
        corrent_list, incorrect_list = validation(evalutes_csv_list, ground_truth_csv)

        # compute kidney size
        seg_path = '~/data/yonsei2/eval/m1_result/SegKidney_MRCNN'
        dicom_info_path = '~/data/yonsei2/doc/Dicom정보/dicom_info_100+400.csv'
        kidney_size = get_kidney_size(corrent_list, seg_path, dicom_info_path)

        # save kidney size
        result_csv = '../result_KorNK/kidney_size.csv'
        save_kidney_size(kidney_size, result_csv)

        # kidney size compare csv
        compute_kidney_size_csv = '../result_KorNK/kidney_size.csv'
        gt_kidney_size_csv = '/home/bong07/data/yonsei/doc/400건_전수조사_데이터/400data_kidney_real_size.csv'
        result_csv = '../result_KorNK/kidney_size_compare.csv'
        valid_kidney_list = make_compare_size(compute_kidney_size_csv, gt_kidney_size_csv, result_csv)

        # kidney overlap image
        result_path = '../result_KorNK/overlap_image'
        ori_path = '~/data/KorNK/OriginalUS'
        mask_path = '~/lib/robin_yonsei/results_us3_mrcnn/kidney/20181106T091512/SegKidney_MRCNN'
        make_compare_image(valid_kidney_list, result_path, ori_path, mask_path)

        result_path = '../result_KorNK/overlap_image_non-kidney'
        copy_image(incorrect_list, result_path, ori_path, mask_path)
