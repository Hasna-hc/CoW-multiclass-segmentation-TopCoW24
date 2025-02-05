import os
import time
import torch
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk

from tqdm import tqdm
from natsort import natsorted
from scipy.ndimage import label
from nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnUNet.nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from post_processings import *


'''
cmd:
export PYTHONPATH=/home/hasna/miccai24_challenges/topcow_challenge_final
export nnUNet_raw='/home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/dataset/nnUNet_raw'
export nnUNet_preprocessed='/home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/dataset/preprocessed'
export nnUNet_results='/home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/datasetnnUNet_trained_models'


>>> CTA:  5 folds with SkeletonRecall + BinDice + NoMirror + With/Without PostProcessing
python /home/hasna/miccai24_challenges/topcow_challenge/src/run_ablation_final.py --save_folder /home/hasna/miccai24_challenges/topcow_challenge/evals/final_CTA_bin809_mul806_skr-bindice_nomir_5folds_val-best --num_folds 5 --min_vol 20

>>> MRA:  5 folds with SkeletonRecall + BinDice + NoMirror + With/Without PostProcessing
python /home/hasna/miccai24_challenges/topcow_challenge/src/run_ablation_final.py --save_folder /home/hasna/miccai24_challenges/topcow_challenge/evals/final_MRA_bin802_mul808_skr-bindice_nomir_5folds_val-best --num_folds 5 --min_vol 20

'''

''' --------------------------------------------------------------- '''
parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--save_folder', type=str, default='', help='Specify the dir to al the trained models')
parser.add_argument('--min_dist', type=float, default=15, help='Specify the minimum distance to link')
parser.add_argument('--min_vol', type=int, default=20, help='Specify the minimum volume to remove')
parser.add_argument('--num_folds', type=int, default=5, help='Specify the number of folds to be used')
parser.add_argument('--interm_save', type=bool, default=False, help='Specify if it saves the intermediate post-processing results')
''' --------------------------------------------------------------- '''



def main(args):
        
    ### Iterate over each input image for inference
    for fold in natsorted(os.listdir(os.path.join(args.save_folder, 'without_pp'))):
        os.makedirs(os.path.join(args.save_folder, 'steps', fold), exist_ok=True)
        for file in tqdm([file for file in natsorted(os.listdir(os.path.join(args.save_folder, 'without_pp', fold))) if file.endswith('.nii.gz')]):
            img = nib.load(os.path.join(args.save_folder, 'without_pp', fold, file))
            pred_array_mul_0 = img.get_fdata()
            pred_array_bin = nib.load(os.path.join(args.save_folder, 'bin', fold, file)).get_fdata()

            ## --- Postprocessing -- - :
            ## Step 1: Replace the BG voxels in multiclass pred & Delete the small disconnected volumes (<20 voxels)
            pred_array_mul = replace_background_with_nearest_label(pred_array_mul_0, pred_array_bin)
            nib.save(nib.Nifti1Image(pred_array_mul, img.affine), os.path.join(args.save_folder, 'steps', fold, 'step1_'+file))
            
            cleaned_arr = clean_small_components(pred_array_mul, args.min_vol)
            cleaned_arr_0 = clean_small_components(pred_array_mul_0, args.min_vol)
            nib.save(nib.Nifti1Image(cleaned_arr, img.affine), os.path.join(args.save_folder, 'steps', fold, 'step2_'+file))
            nib.save(nib.Nifti1Image(cleaned_arr_0, img.affine), os.path.join(args.save_folder, 'steps', fold, 'step2-0_'+file))
            
            pred_array_5 = final_check(cleaned_arr)  # Final checking: if a certain label has disconnected components, the smallest is either deleted (if it's not connected to anything) or takes the label of its neighbour (to ensure continuity)..
            pred_array_5[pred_array_5 == 13] = 15  # Last one just in case..
            nib.save(nib.Nifti1Image(pred_array_5, img.affine), os.path.join(args.save_folder, 'steps', fold, 'step5_'+file))



            ## Step 2: Connect the disconnected components (linkin parts)
            new_arr = np.zeros(cleaned_arr.shape)  # New array to store the modifications (bridges)
            for i in np.unique(cleaned_arr):  # Iterating over each class
                binary_image = 1*(cleaned_arr==i)  # Dealing with 1 label class per time
                _, nums = label(binary_image)

                if nums > 1:  # If for that specific label, there are more than 1 component (thus, disconnexions)
                    dilated = link_components(binary_image)  # Link between the two closest points of the two closest components by dilating the line connecting them and multiply it by their class label
                    dilated_corrected = 1*( (dilated - 1*(cleaned_arr>0)) > 0)  # Get only the extra voxels to not interfere with what was originally there..
                    dilated_corrected = i*( (dilated_corrected - 1*(new_arr>0)) > 0)  # Get only the extra voxels to not interfere with what was previously dilated from other accumulated labels...
                    new_arr += (dilated_corrected.astype(np.uint8))

            corrected = new_arr*((1*(new_arr>0) - 1*(cleaned_arr>0)) > 0)  # Getting the final extra voxels (from bridges) to not interfer with the original + multiplying by the new_arr to get the classes in the binary (subtraction)
            cleaned_arr[corrected != 0] = 0  # Replace only where there is no interferance with the other labels !!
            cleaned_arr += (corrected.astype(np.uint8))  # Adding the final extra voxel to the original, after previously replacing the original with 0 where it should..
            nib.save(nib.Nifti1Image(cleaned_arr, img.affine), os.path.join(args.save_folder, 'steps', fold, 'step3_'+file))


            ## Step 2-0: ---------------------------
            tmp_cleaned_arr_0 = pred_array_mul_0.copy()
            new_arr_0 = np.zeros(tmp_cleaned_arr_0.shape)
            for i in np.unique(tmp_cleaned_arr_0):
                binary_image = 1*(tmp_cleaned_arr_0==i)
                _, nums = label(binary_image)
                if nums > 1:  # If for that specific label, there are more than 1 component (thus, disconnexions)
                    dilated = link_components(binary_image)  # Link between the two closest points of the two closest components by dilating the line connecting them and multiply it by their class label
                    dilated_corrected = 1*( (dilated - 1*(tmp_cleaned_arr_0>0)) > 0)  # Get only the extra voxels to not interfere with what was originally there..
                    dilated_corrected = i*( (dilated_corrected - 1*(new_arr_0>0)) > 0)  # Get only the extra voxels to not interfere with what was previously dilated from other accumulated labels...
                    new_arr_0 += (dilated_corrected.astype(np.uint8))

            corrected = new_arr_0*((1*(new_arr_0>0) - 1*(tmp_cleaned_arr_0>0)) > 0)  # Getting the final extra voxels (from bridges) to not interfer with the original + multiplying by the new_arr to get the classes in the binary (subtraction)
            tmp_cleaned_arr_0[corrected != 0] = 0  # Replace only where there is no interferance with the other labels !!
            tmp_cleaned_arr_0 += (corrected.astype(np.uint8))
            nib.save(nib.Nifti1Image(tmp_cleaned_arr_0, img.affine), os.path.join(args.save_folder, 'steps', fold, 'step3-0_'+file))


            pred_array = final_check(cleaned_arr)  # Final checking: if a certain label has disconnected components, the smallest is either deleted (if it's not connected to anything) or takes the label of its neighbour (to ensure continuity)..
            pred_array[pred_array == 13] = 15  # Last one just in case..
            nib.save(nib.Nifti1Image(pred_array, img.affine), os.path.join(args.save_folder, 'steps', fold, 'step4_'+file))

            pred_array_0 = final_check(pred_array_mul_0)
            pred_array_0[pred_array_0 == 13] = 15
            nib.save(nib.Nifti1Image(pred_array_0, img.affine), os.path.join(args.save_folder, 'steps', fold, 'step4-0_'+file))

            # img = nib.load(os.path.join(args.save_folder, 'steps', fold, 'step2_'+file))
            # pred_array_mul_0 = img.get_fdata()
            # pred_array = final_check(pred_array_mul_0)  # Final checking: if a certain label has disconnected components, the smallest is either deleted (if it's not connected to anything) or takes the label of its neighbour (to ensure continuity)..
            # pred_array[pred_array == 13] = 15  # Last one just in case..
            # nib.save(nib.Nifti1Image(pred_array, img.affine), os.path.join(args.save_folder, 'steps', fold, 'step5__'+file))

# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
