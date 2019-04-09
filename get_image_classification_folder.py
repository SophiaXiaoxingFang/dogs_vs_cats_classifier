import os
import csv
import shutil

FILE_DIR_DOGS = ".\\result\\dogs"
FILE_DIR_CATS = ".\\result\\cats"

csvFile = open("submission_file.csv", "r")
reader = csv.reader(csvFile)

if os.path.exists(FILE_DIR_DOGS):
    shutil.rmtree(FILE_DIR_DOGS)

if os.path.exists(FILE_DIR_CATS):
    shutil.rmtree(FILE_DIR_CATS)

os.makedirs(FILE_DIR_DOGS)
os.makedirs(FILE_DIR_CATS)

for item in reader:
    if reader.line_num == 1:
        continue
    if item[1] == '1':
        shutil.copy('.\\data\\test\\{}.jpg'.format(item[0]), FILE_DIR_DOGS)
    else:
        shutil.copy('.\\data\\test\\{}.jpg'.format(item[0]), FILE_DIR_CATS)

print("The testing images have been put in corresponding folders.")