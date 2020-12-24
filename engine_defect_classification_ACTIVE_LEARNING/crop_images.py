## Get essential Libraries..
import cv2
import glob
import os
import xml.etree.cElementTree as et
import pandas as pd
from imutils import paths
import random
import shutil
from tqdm import tqdm
import random

## Location of images & corresponding xml files
## Input folders
# defective_path = "SAMPLE_rr/defective"
# non_defective_path = "SAMPLE_rr/nondefective"

# # ## Output folders
# final_output_lower_data = "SAMPLE_rr/Lower"
# final_output_upper_data = "SAMPLE_rr/Upper"

## Functions to crop images using XML files
def getvalueofnode(node):
    """ return node text or None """
    return node.text if node is not None else None

def Make_Class_Dir(target_folder):
    classes = ["Class_1_Left","Class_1_Right","Class_2_Left","Class_2_Right","Class_null","Defective"]
    for c in classes:
        dir_path = os.path.join(target_folder,c)
        if(os.path.isdir(dir_path)==True):
            continue
        else:
            os.makedirs(dir_path)
            
def XML_crop(xml_folder,img_folder,target_folder):
    Make_Class_Dir(target_folder)
    xml_files = xml_folder + "*"
    img_files = img_folder + "*"
    newName=""
    countImages=0
    COUNT = 0
    for iname in tqdm(glob.glob(img_files)):
        for fname in glob.glob(xml_files):
            if(iname.split('/')[-1].split('.')[0]==fname.split('/')[-1].split('.')[0]):
                try:
                    parsed_xml = et.parse(fname)
                    newName=""
                    countImages=countImages+1
                    name = iname.split('/')[-1]
                    im = cv2.imread(iname)
                    height, width, channels = im.shape
                    input_image_size = height
                    
                    root = parsed_xml.getroot()
                    objects = root.findall('object')
                    for obj in objects:
                        cn = obj.find('name')
                        class_name = getvalueofnode(cn)
                        bb = obj.find('bndbox')
                        if bb is not None:
                            xminn=bb.find('xmin')
                            vminx=getvalueofnode(xminn)
                            xmaxx=bb.find('xmax')
                            vmaxx=getvalueofnode(xmaxx)
                            yminn=bb.find('ymin')
                            vminy=getvalueofnode(yminn)
                            ymaxx=bb.find('ymax')
                            vmaxy=getvalueofnode(ymaxx)
                            cropped = im[int(vminy):int(vmaxy), int(vminx):int(vmaxx), :]
                            cropped_1= im[int(vminy):int(vmaxy), int(vminx)-15:int(vmaxx)-15, :]
                            cropped_2= im[int(vminy):int(vmaxy), int(vminx)+15:int(vmaxx)+15, :]
                            cropped_3= im[int(vminy)-15:int(vmaxy)-15, int(vminx):int(vmaxx), :]
                            cropped_4= im[int(vminy)+15:int(vmaxy)+15, int(vminx):int(vmaxx), :]
                            cropped_5= im[int(vminy)-15:int(vmaxy)+15, int(vminx):int(vmaxx), :]
                            cropped_6= im[int(vminy):int(vmaxy), int(vminx)-15:int(vmaxx)+15, :]
                            
                            class_name = class_name.replace(" ","_")
#                             print(class_name)
                            dir_path = os.path.join(target_folder,class_name)
                            img_name = name[:-4] + "_" + vminx + name[-4:]
                            img_name_1 = name[:-4] + "_1" + vminx + name[-4:]
                            img_name_2= name[:-4] + "_2" + vminx + name[-4:]
                            img_name_3= name[:-4] + "_3" + vminx + name[-4:]
                            img_name_4= name[:-4] + "_4" + vminx + name[-4:]
                            img_name_5= name[:-4] + "_5" + vminx + name[-4:]
                            img_name_6= name[:-4] + "_6" + vminx + name[-4:]
                            
                            img_path = os.path.join(dir_path,img_name)
                            img_path_1= os.path.join(dir_path,img_name_1)
                            img_path_2= os.path.join(dir_path,img_name_2)
                            img_path_3= os.path.join(dir_path,img_name_3)
                            img_path_4= os.path.join(dir_path,img_name_4)
                            img_path_5= os.path.join(dir_path,img_name_5)
                            img_path_6= os.path.join(dir_path,img_name_6)
                            
                            
                            cv2.imwrite(img_path,cropped)
                            cv2.imwrite(img_path_1,cropped_1)
                            cv2.imwrite(img_path_2,cropped_2)
                            cv2.imwrite(img_path_3,cropped_3)
                            cv2.imwrite(img_path_4,cropped_4)
                            cv2.imwrite(img_path_5,cropped_5)
                            cv2.imwrite(img_path_6,cropped_6)
                    break
                except:
                    pass
            else:
                continue
                
                
## Functions to get cropped images and to separate into lower and upper parts
def create_folder(path, x):
    rel_path = str(x)
    my_path = os.path.join(path,rel_path)
    if not os.path.isdir(my_path):
        os.makedirs(my_path)
    return my_path

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
        
#     shutil.rmtree(current_folder)

def crop(path):
    img_folder = path + "/img/"
    xml_folder = path + "/xml/"
    target_path = create_folder(path, "cropped")
    XML_crop(xml_folder,img_folder,target_path)
    # Location of cropped images
    upper_img_folder_1 = target_path + "/Class_1_Left"
    upper_img_folder_2 = target_path + "/Class_1_Right"

    lower_img_folder_1 = target_path + "/Class_2_Left"
    lower_img_folder_2 = target_path + "/Class_2_Right"
    
    #### Location to save final lower $ upper images

    output_lower = create_folder(path, "lower")
    output_upper = create_folder(path, "upper")
    
    ### Copying all defective upper [C1L and C1R] to one folder

    Copy(upper_img_folder_1,output_upper)
    Copy(upper_img_folder_2,output_upper)

    ### Copying all defective lower [C2L and C2R] to one folder

    Copy(lower_img_folder_1,output_lower)
    Copy(lower_img_folder_2,output_lower)
    
    ### remove extra folders once copying is done
    shutil.rmtree(target_path)
            
    return (output_lower,output_upper)

## Functions to get final lower and upper dataset
def create_final_data_folder(path):
    defective = path + "/1/"
    non_defective = path + "/0/"
    classes = [defective, non_defective]
    for c in classes:
        dir_path = c
        if(os.path.isdir(dir_path)==True):
            continue
        else:
            os.makedirs(dir_path)
    return (non_defective, defective)

def get_data_upper(input_path_def, input_path_non_def, output_path_upper):
    ## create final folder to get upper data
    non_def_dest_u ,def_dest_u = create_final_data_folder(output_path_upper)
    
    ## get all lower and upper data
    output_def_lower, output_def_upper = crop(input_path_def)
    output_non_def_lower, output_non_def_upper = crop(input_path_non_def)
    
    ## copy upper data in its respective folders
    Copy(output_non_def_upper,non_def_dest_u)
    Copy(output_def_upper,def_dest_u)
    
    ## remove extra folders once copying is done..
    classes = [output_def_upper, output_non_def_upper]
    for c in classes:
        dir_path = c
        shutil.rmtree(dir_path)
        
def get_data_lower(input_path_def, input_path_non_def, output_path_lower):
    ## create final folder to get upper data
    non_def_dest_l ,def_dest_l = create_final_data_folder(output_path_lower)
    
    ## get all lower and upper data
    output_def_lower, output_def_upper = crop(input_path_def)
    output_non_def_lower, output_non_def_upper = crop(input_path_non_def)
    
    ## copy upper data in its respective folders
    Copy(output_non_def_lower,non_def_dest_l)
    Copy(output_def_lower,def_dest_l)
    
    ## remove extra folders once copying is done..
    classes = [output_def_lower, output_non_def_lower]
    for c in classes:
        dir_path = c
        shutil.rmtree(dir_path)


## FInally call to get things done....        
# get_data(defective_path, non_defective_path, final_output_lower_data, final_output_upper_data)