import os
from glob import glob
import shutil
import json

main_dir_path = "/workspaces/ComputerVisionPlayGround/Dataset/shapenetcore_partanno_segmentation_benchmark_v0_normal/"
value = "test"

f = open("/workspaces/ComputerVisionPlayGround/Dataset/shapenetcore_partanno_segmentation_benchmark_v0_normal/train_test_split/shuffled_test_file_list.json", 'r')
file_name_list = json.load(f)
f.close()

for item in file_name_list:

    item = item.split("/")[1:]
    dir_name, file_name = item[0], item[1]

    os.makedirs(os.path.join(main_dir_path, f"{value}", dir_name), exist_ok=True)

    src = os.path.join(main_dir_path, dir_name, f"{file_name}.txt")
    dst = os.path.join(main_dir_path, f"{value}", dir_name, f"{file_name}.txt")
    shutil.copy(src, dst)

    src = os.path.join(main_dir_path, dir_name, f"{file_name}_8x8.npz")
    dst = os.path.join(main_dir_path, f"{value}", dir_name, f"{file_name}_8x8.npz")
    shutil.copy(src, dst)


# for dir_n in os.listdir(main_dir_path):

#     npz_list = 