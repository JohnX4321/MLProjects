from os import makedirs,listdir
from shutil import copyfile
from random import *

dataset_home='dataset_dogs_vs_cats/'
subdirs=['train/','test/']
for subdir in subdirs:
    labeldirs=['dogs/','cats/']
    for labeldir in labeldirs:
        newdir=dataset_home+subdir+labeldir
        makedirs(newdir,exist_ok=True)

seed(1)

val_ratio=0.25
src_dir='train/'
for file in listdir(src_dir):
    src=src_dir+'/'+file
    dst_dir='test/'
    if file.startswith('cat'):
        dst=dataset_home+dst_dir+'cats/'+file
        copyfile(src,dst)
    elif file.startswith('dog'):
        dst=dataset_home+dst_dir+'dogs/'+file
        copyfile(src,dst)