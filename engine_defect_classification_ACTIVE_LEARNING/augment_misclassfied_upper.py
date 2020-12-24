import numpy as np 
import os
import cv2
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
import cv2
import glob
import os
import xml.etree.cElementTree as et
import pandas as pd
from imutils import paths
import random
import shutil
from tqdm import tqdm



def show(img):
    plt.imshow(img.astype(np.uint8))
    plt.show()




def get_shift_values(fraction_shift,height,width):
    x_shift_1_right=fraction_shift*width
    x_shift_1_left=-1*fraction_shift*width
    y_shift_1_right=fraction_shift*height
    y_shift_1_left=-1*fraction_shift*height
    return int(x_shift_1_right),int(x_shift_1_left),int(y_shift_1_right),int(y_shift_1_left)


def translation(image,fraction_shift):
    height,width,shape=image.shape
    x_r,x_l,y_r,y_l=get_shift_values(fraction_shift,height,width)
    img_copy_r=image.copy()
    shift=x_r
    for i in range(0, shift, +1):
        img_copy_r= np.roll(img_copy_r, -1, axis=1)
        img_copy_r[:, -1] = 255
    img_copy_l=image.copy()
    for i in range(img_copy_l.shape[1] -1, img_copy_l.shape[1] - shift, -1):
        img_copy_l= np.roll(img_copy_l, +1, axis=1)
        img_copy_l[:, -1] = 255
        
    ###ALONG Y
    shift=y_r
    img_copy_u=image.copy()
    for i in range(0, shift, +1):
        img_copy_u = np.roll(img_copy_u, +1, axis=0)
        img_copy_u[-1, :] = 255
        
    img_copy_d=image.copy()
    for i in range(img_copy_d.shape[0] -1, img_copy_d.shape[0] - shift, -1):
        img_copy_d = np.roll(img_copy_d, -1, axis=0)
        img_copy_d[-1, :] = 255
        
    
    return img_copy_r,img_copy_l,img_copy_u,img_copy_d


def gaussian_noise(image,mean,var):
    row,col,ch= image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
#     print(gauss.shape)
    gauss = gauss.reshape(row,col,ch)
    noise_img = image + gauss
    return noise_img

def Copy(current_folder,destination_folder):
    files = os.listdir(current_folder)
    for file in tqdm(files):
        if(file== ".ipynb_checkpoints"):
            continue
        if(file== "kpoints"):
            continue
        if(file== "out"):
            continue
        shutil.copy(os.path.join(current_folder,file), destination_folder) 

def image_augment(image,path,output_path):
        img = image.copy()
        filename=path.split('/')[-1]
        a,b,c,d=translation(img,.2)
        e,f,g,h=translation(img,.1)
        p=gaussian_noise(image,.5,.8)
        
        name_int = filename[:len(filename)-4]
#         print("name_int",name_int)
        cv2.imwrite(output_path+str(name_int)+'.bmp',img)
        cv2.imwrite(output_path+str(name_int)+'r_shift_2.bmp',a)
        cv2.imwrite(output_path+str(name_int)+'l_shift_2.bmp',b)
        cv2.imwrite(output_path+str(name_int)+'up_shift_2.bmp',c)
        cv2.imwrite(output_path+str(name_int)+'down_shift_2.bmp',d)
        cv2.imwrite(output_path+str(name_int)+'gaussian_noise.bmp',p)
        cv2.imwrite(output_path+str(name_int)+'r_shift_1.bmp',e)
        cv2.imwrite(output_path+str(name_int)+'l_shift_1.bmp',f)
        cv2.imwrite(output_path+str(name_int)+'up_shift_1.bmp',g)
        cv2.imwrite(output_path+str(name_int)+'down_shift_1.bmp',h)

        
def Diff(A,B):
    return (list(set(A) - set(B)))


def train_test_split(input_img,  train_img, test_img):
    files_bmp = os.listdir(input_img)
    train_bmp = random.sample(files_bmp,int(0.75*len(files_bmp)))
    test_bmp = Diff(files_bmp,train_bmp)
    
    for file in tqdm(train_bmp):
        if(file== ".ipynb_checkpoints"):
            continue
        if(file== "kpoints"):
            continue
        if(file== "out"):
            continue
            
        shutil.copy(os.path.join(input_img,file), train_img)
#         name = file.split('.')[0]
#         name = ".".join((name,"xml"))
#         shutil.copy(os.path.join(input_xml,name),train_xml)
        
    for file in tqdm(test_bmp):
        if(file== ".ipynb_checkpoints"):
            continue
        if(file== "kpoints"):
            continue

        shutil.copy(os.path.join(input_img,file), test_img)
#         name = file.split('.')[0]
#         name = ".".join((name,"xml"))
#         shutil.copy(os.path.join(input_xml,name),test_xml)



def train_test_split_majority(input_img,  train_img, test_img):
    files_bmp = os.listdir(input_img)
#     files_bmp=random.sample(files_bmp,int(0.75*len(files_bmp)))
    train_bmp = random.sample(files_bmp,int(0.75*len(files_bmp)))
    test_bmp = Diff(files_bmp,train_bmp)
    
    for file in tqdm(train_bmp):
        if(file== ".ipynb_checkpoints"):
            continue
        if(file== "kpoints"):
            continue
        if(file== "out"):
            continue
            
        shutil.copy(os.path.join(input_img,file), train_img)
#         name = file.split('.')[0]
#         name = ".".join((name,"xml"))
#         shutil.copy(os.path.join(input_xml,name),train_xml)
        
    for file in tqdm(test_bmp):
        if(file== ".ipynb_checkpoints"):
            continue
        if(file== "kpoints"):
            continue

        shutil.copy(os.path.join(input_img,file), test_img)
#         name = file.split('.')[0]
#         name = ".".join((name,"xml"))
#         shutil.copy(os.path.join(input_xml,name),test_xml)


       
# def run_augment_old(new_folder_path):
#     complete_def_upper=new_folder_path+'/1'
#     # complete_def_upper=new_folder_path+'/1"
#     complete_non_def_upper=new_folder_path+'/0'
#     # complete_non_def_upper=new_folder_path+'/0"


#     train_def_upper=new_folder_path+'/train/1'
#     # train_def_upper=new_folder_path+'/train/1"
#     train_non_def_upper=new_folder_path+'/train/0'
#     # train_non_def_upper=new_folder_path+'/train/0"

#     test_def_upper=new_folder_path+'/test/1'
#     # test_def_upper=new_folder_path+'/test/1"
#     test_non_def_upper=new_folder_path+'/test/0'
#     # test_non_def_upper=new_folder_path+'/test/0"
    
#     ########Create these folders if not exist else
    
#     folder=new_folder_path
#     list_train_test=["train","test","train_new_augmented"]
#     for i in list_train_test:
#         dirpath = os.path.join(folder, str(i))
#         if os.path.exists(dirpath) and os.path.isdir(dirpath):
#             shutil.rmtree(dirpath)
#         os.makedirs(dirpath)
        
    
    
#     ####upper
#     folder=new_folder_path+'/train'
#     for i in range(2):
#         dirpath = os.path.join(folder, str(i))
#         if os.path.exists(dirpath) and os.path.isdir(dirpath):
#             shutil.rmtree(dirpath)
#         os.makedirs(dirpath)

#     folder=new_folder_path+'/test'
#     for i in range(2):
#         dirpath = os.path.join(folder, str(i))
#         if os.path.exists(dirpath) and os.path.isdir(dirpath):
#             shutil.rmtree(dirpath)
#         os.makedirs(dirpath)
        
#     folder=new_folder_path+'/train_new_augmented'
#     for i in range(2):
#         dirpath = os.path.join(folder, str(i))
#         if os.path.exists(dirpath) and os.path.isdir(dirpath):
#             shutil.rmtree(dirpath)
#         os.makedirs(dirpath)
    
#     ####DEFECTYIVE CLASSES
#     train_test_split(complete_def_upper,  train_def_upper, test_def_upper)
#     # train_test_split(complete_def_upper,  train_def_upper, test_def_upper)
    
    
#     ###NONDEFECTIVE CLASSES
#     # train_test_split_majority(complete_non_def_upper,  train_non_def_upper, test_non_def_upper)
#     train_test_split_majority(complete_non_def_upper,  train_non_def_upper, test_non_def_upper)
    
#     #########AUGMENTING DEFECTIVE ONLY
    
#     file_dir=new_folder_path+'/train/1'
#     output_path=new_folder_path+'/train_new_augmented/1/'
    
    
#     for root, _, files in os.walk(file_dir):
# #         print(root)
#         for file in files:
#             if(file== ".ipynb_checkpoints"):
#                 continue
# #             print("file",file)
#             file_path=os.path.join(root,file)
#             print("file_path",file_path)
#             a=cv2.imread(file_path,-1)
# #             print(output_path)
#             image_augment(a,file_path,output_path)
    
#     ### Copying the non defective images to train_new_augmented
#     Copy(new_folder_path+'/train/0',new_folder_path+'/train_new_augmented/0')
#     # Copy(new_folder_path+'/train/0","new_folder_path+'/train_new_augmented/0")
    

def run_augment(new_folder_path):
    complete_def_upper=new_folder_path+'/1'
    complete_non_def_upper=new_folder_path+'/0'

    train_def_upper=new_folder_path+'/train/1'
    train_non_def_upper=new_folder_path+'/train/0'

    test_def_upper=new_folder_path+'/test/1'
    test_non_def_upper=new_folder_path+'/test/0'
    
    ########Create these folders if not exist else
    
    folder=new_folder_path
    list_train_test=["train","test","train_new_augmented"]
    for i in list_train_test:
        dirpath = os.path.join(folder, str(i))
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        os.makedirs(dirpath)
        
    ####upper
    folder=new_folder_path+'/train'
    for i in range(2):
        dirpath = os.path.join(folder, str(i))
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        os.makedirs(dirpath)

    folder=new_folder_path+'/test'
    for i in range(2):
        dirpath = os.path.join(folder, str(i))
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        os.makedirs(dirpath)
        
    folder=new_folder_path+'/train_new_augmented'
    for i in range(2):
        dirpath = os.path.join(folder, str(i))
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        os.makedirs(dirpath)
    
    ####DEFECTYIVE CLASSES
    train_test_split(complete_def_upper,  train_def_upper, test_def_upper)
    
    ###NONDEFECTIVE CLASSES
    train_test_split_majority(complete_non_def_upper,  train_non_def_upper, test_non_def_upper)
    
    number_of_def_train = len(os.listdir(train_def_upper))
    number_of_non_def_train = len(os.listdir(train_non_def_upper))
    
    number_of_non_def_delete = 0
    
    if(number_of_non_def_train>0):
        ratio = number_of_def_train/number_of_non_def_train
    
    if(ratio < 0.1):
        number_of_non_def_delete = number_of_non_def_train - 10*number_of_def_train
    
    #########AUGMENTING DEFECTIVE ONLY
    
    file_dir=new_folder_path+'/train/1'
    output_path=new_folder_path+'/train_new_augmented/1/'
    
    
    for root, _, files in os.walk(file_dir):
#         print(root)
        for file in files:
            if(file== ".ipynb_checkpoints"):
                continue
#             print("file",file)
            file_path=os.path.join(root,file)
#             print("file_path",file_path)
            a=cv2.imread(file_path,-1)
#             print(output_path)
            image_augment(a,file_path,output_path)
    
    ### Copying the non defective images to train_new_augmented
    Copy(new_folder_path+'/train/0',new_folder_path+'/train_new_augmented/0')
    # Copy(new_folder_path+'/train/0","new_folder_path+'/train_new_augmented/0")
    return number_of_non_def_delete