  
#!/bin/bash

# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

# ************************************
# MODIFY THESE OPTIONS
# Modify this: which gpu (according to nvidia-smi) do you want to use for training
# this can be a single number, or a list. E.g "3" or "0,1" "0,2,3"
# the training script will use all gpus you list
GPU="0"

# location of original data
old_data_folder="/root/preprocess-3/root/Anshit/OLD_DATA/old_lower"
# location of new data for retraining
# new_data_folder="/root/preprocess-3/root/Anshit/testing_scripts_training/new_data/lower"
new_data_folder="/root/preprocess-3/root/Anshit/NEW_DATA"

# location of output_checkpoints to be saved after retraining (also model checkpoint path)
output_folder="/root/preprocess-3/root/Anshit/NEW_DATA/lower_retrain_ckpts"

# location of old trained model
old_model_path="/root/preprocess-3/root/Anshit/OLD_DATA/old_lower/old_trained_model/saved-model-augmented-gpu-100.h5"
# location to save final best model
final_model_path="testing_scripts_training/final_compare_weights/lower/"

# # what are the input lmdb databases called
# train_lmdb_file="train-mnist.lmdb"
# test_lmdb_file="test-mnist.lmdb"

# # what learning rate should the network use
# learning_rate=1e-4
# number_classes=2
# balance_classes=1
# use_augmentation=1 # {0, 1}
# MODIFY THESE OPTIONS
# ************************************

# DO NOT MODIFY ANYTHING BELOW

# limit the script to only the GPUs you selected above
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export CUDA_VISIBLE_DEVICES=${GPU}

mkdir -p ${output_folder}

# launch training script with required options
echo "Launching Training Script"
python3 Train_Lower.py  --old_data_path=${old_data_folder} --new_data_path=${new_data_folder} --output_dir=${output_folder} --old_model_path=${old_model_path} --final_model_path=${final_model_path}
echo "Job completed"