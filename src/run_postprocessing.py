''' CODE FOR POST-PROCESSING ON ABLATION RESULTS'''
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
export PYTHONPATH=/home/hasna/miccai24_challenges/topcow_challenge
export nnUNet_raw='/home/hasna/miccai24_challenges/topcow_challenge/nnunet_dir/dataset/nnUNet_raw'
export nnUNet_preprocessed='/home/hasna/miccai24_challenges/topcow_challenge/nnunet_dir/dataset/preprocessed'
export nnUNet_results='/home/hasna/miccai24_challenges/topcow_challenge/nnunet_dir/datasetnnUNet_trained_models'

python /home/hasna/miccai24_challenges/topcow_challenge/src/run_postprocessing.py --preds_mul_folder /home/hasna/miccai24_challenges/topcow_challenge/nnunet_dir/datasetnnUNet_trained_models/Dataset815_TopCoWSegCTA/nnUNetTrainerSkeletonRecallBinDiceNoMirroring__nnUNetPlans__3d_fullres_1000epochs --preds_bin_folder /home/hasna/miccai24_challenges/topcow_challenge/nnunet_dir/datasetnnUNet_trained_models/Dataset809_TopCoWSegBinCTA/nnUNetTrainerSkeletonRecall__nnUNetPlans__3d_fullres_1000epochs --save_folder /home/hasna/miccai24_challenges/topcow_challenge/evals/CTA_D815-all_D809-all_1000ep --num_folds 5 --min_vol 20 --interm_save True --mod ct
python /home/hasna/miccai24_challenges/topcow_challenge/src/run_postprocessing.py --preds_mul_folder /home/hasna/miccai24_challenges/topcow_challenge/nnunet_dir/datasetnnUNet_trained_models/Dataset806_TopCoWSegCTAMRA/nnUNetTrainerSkeletonRecallBinDiceNoMirroring__nnUNetPlans__3d_fullres_ps --preds_bin_folder /home/hasna/miccai24_challenges/topcow_challenge/nnunet_dir/datasetnnUNet_trained_models/Dataset809_TopCoWSegBinCTA/nnUNetTrainerSkeletonRecall__nnUNetPlans__3d_fullres_1000epochs --save_folder /home/hasna/miccai24_challenges/topcow_challenge/evals/CTA_D806-all_D809-all_1000ep --num_folds 5 --min_vol 20 --interm_save True --mod ct
python /home/hasna/miccai24_challenges/topcow_challenge/src/run_postprocessing.py --preds_mul_folder /home/hasna/miccai24_challenges/topcow_challenge/evals/806_cta_skr_bindice_nomir_5folds_val-best/without_pp --preds_bin_folder /home/hasna/miccai24_challenges/topcow_challenge/nnunet_dir/datasetnnUNet_trained_models/Dataset809_TopCoWSegBinCTA/nnUNetTrainerSkeletonRecall__nnUNetPlans__3d_fullres_1000epochs --save_folder /home/hasna/miccai24_challenges/topcow_challenge/evals/CTA_D806-all-best_D809-all_1000ep --num_folds 5 --min_vol 20 --interm_save True --mod ct

'''

''' --------------------------------------------------------------- '''
parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--save_folder', type=str, default='', help='Specify the dir to al the trained models')
parser.add_argument('--preds_mul_folder', type=str, default='', help='Specify the dir to the multiclass preds')
parser.add_argument('--preds_bin_folder', type=str, default='', help='Specify the dir to the binary preds')
parser.add_argument('--min_dist', type=float, default=15, help='Specify the minimum distance to link')
parser.add_argument('--min_vol', type=int, default=20, help='Specify the minimum volume to remove')
parser.add_argument('--num_folds', type=int, default=5, help='Specify the number of folds to be used')
parser.add_argument('--mod', type=str, default='mr', help='Specify the modality to be used')
parser.add_argument('--interm_save', type=bool, default=False, help='Specify if it saves the intermediate post-processing results')
''' --------------------------------------------------------------- '''



def main(args):
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if args.interm_save: os.makedirs(os.path.join(args.save_folder, 'steps'), exist_ok=True)
        
    ### Iterate over each input image for inference
    for fold in range(args.num_folds): #natsorted(os.listdir(os.path.join(args.preds_mul_folder))):
        os.makedirs(os.path.join(args.save_folder, f'with_pp_{args.min_vol}', f'fold_{fold}'), exist_ok=True)
        for file in tqdm([file for file in natsorted(os.listdir(os.path.join(args.preds_mul_folder, f'fold_{fold}'))) if file.startswith(f'topcow_{args.mod}')]):   #file.endswith('.nii.gz')
            img = nib.load(os.path.join(args.preds_mul_folder, f'fold_{fold}', file))
            pred_array_mul = img.get_fdata()
            # pred_array_bin = nib.load(os.path.join(args.preds_bin_folder, f'fold_{fold}', 'validation', file)).get_fdata()  #FIXME: For ablation (5 folds)
            pred_array_bin = nib.load(os.path.join(args.preds_bin_folder, 'fold_all/validation', file)).get_fdata()  #FIXME: For final one (all)

            ## --- Postprocessing --- :
            ## Step 1: Replace the BG voxels in multiclass pred & Delete the small disconnected volumes (<20 voxels)
            pred_array_mul = replace_background_with_nearest_label(pred_array_mul, pred_array_bin)
            if args.interm_save: nib.save(nib.Nifti1Image(pred_array_mul, img.affine), os.path.join(args.save_folder, 'steps', 'step1_'+file))
            
            cleaned_arr = clean_small_components(pred_array_mul, args.min_vol)
            if args.interm_save: nib.save(nib.Nifti1Image(cleaned_arr, img.affine), os.path.join(args.save_folder, 'steps', 'step2_'+file))

            ## Step 2: Connect the disconnected components (linkin parts)
            new_arr = np.zeros(cleaned_arr.shape)  # New array to store the modifications (bridges)
            for i in np.unique(cleaned_arr):  # Iterating over each class
                binary_image = 1*(cleaned_arr==i)  # Dealing with 1 label class per time
                _, nums = label(binary_image)

                if nums > 1:  # If for that specific label, there are more than 1 component (thus, disconnexions)
                    dilated = link_components(binary_image, cleaned_arr)  # Link between the two closest points of the two closest components by dilating the line connecting them and multiply it by their class label
                    dilated_corrected = 1*( (dilated - 1*(cleaned_arr>0)) > 0)  # Get only the extra voxels to not interfere with what was originally there..
                    dilated_corrected = i*( (dilated_corrected - 1*(new_arr>0)) > 0)  # Get only the extra voxels to not interfere with what was previously dilated from other accumulated labels...            
                    new_arr += (dilated_corrected.astype(np.uint8))

            corrected = new_arr*((1*(new_arr>0) - 1*(cleaned_arr>0)) > 0)  # Getting the final extra voxels (from bridges) to not interfer with the original + multiplying by the new_arr to get the classes in the binary (subtraction)
            cleaned_arr[corrected != 0] = 0  # Replace only where there is no interferance with the other labels !!
            cleaned_arr += (corrected.astype(np.uint8))  # Adding the final extra voxel to the original, after previously replacing the original with 0 where it should..
            if args.interm_save: nib.save(nib.Nifti1Image(cleaned_arr, img.affine), os.path.join(args.save_folder, 'steps', 'step3_'+file))

            pred_array = final_check(cleaned_arr)  # Final checking: if a certain label has disconnected components, the smallest is either deleted (if it's not connected to anything) or takes the label of its neighbour (to ensure continuity)..
            pred_array[pred_array == 13] = 15  # Last one just in case..
            nib.save(nib.Nifti1Image(pred_array, img.affine), os.path.join(args.save_folder, f'with_pp_{args.min_vol}', f'fold_{fold}', file))


# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)










''' ORIGINAL CODE HERE '''
# import os
# import time
# import torch
# import argparse
# import numpy as np
# import nibabel as nib
# import SimpleITK as sitk

# from tqdm import tqdm
# from natsort import natsorted
# from scipy.ndimage import label
# from nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
# from nnUNet.nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
# from post_processings import *


# '''
# cmd:
# export PYTHONPATH=/home/hasna/miccai24_challenges/topcow_challenge_final
# export nnUNet_raw='/home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/dataset/nnUNet_raw'
# export nnUNet_preprocessed='/home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/dataset/preprocessed'
# export nnUNet_results='/home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/datasetnnUNet_trained_models'


# #FIXME: MRA-CROWN2023   5 folds with SkeletonRecall + BinDice + NoMirror + With/Without PostProcessing
# python /home/hasna/miccai24_challenges/topcow_challenge_final/src/inference_postprocess.py --model_bin_path /home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/datasetnnUNet_trained_models/Dataset802_TopCoWSegBinMRA/nnUNetTrainerSkeletonRecall__nnUNetPlans__3d_fullres --model_mul_path /home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/datasetnnUNet_trained_models/Dataset808_TopCoWSegMRA/nnUNetTrainerSkeletonRecallBinDiceNoMirroring__nnUNetPlans__3d_fullres --input_images /home/hasna/datasets/TopCoW2024_Data_Release/CROWN23/imagesTr --save_folder /home/hasna/miccai24_challenges/topcow_challenge_final/evals/mra_crown23_skr_bindice_nomir_5folds --num_folds 5 --gpu 3

# #FIXME: MRA-CROWN2023   4 folds with SkeletonRecall + BinDice + NoMirror + With/Without PostProcessing (4 folds by removing the one with 0 in label 13)
# python /home/hasna/miccai24_challenges/topcow_challenge_final/src/inference_postprocess.py --model_bin_path /home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/datasetnnUNet_trained_models/Dataset802_TopCoWSegBinMRA/nnUNetTrainerSkeletonRecall__nnUNetPlans__3d_fullres --model_mul_path /home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/datasetnnUNet_trained_models/Dataset808_TopCoWSegMRA/nnUNetTrainerSkeletonRecallBinDiceNoMirroring__nnUNetPlans__3d_fullres --input_images /home/hasna/datasets/TopCoW2024_Data_Release/CROWN23/imagesTr --save_folder /home/hasna/miccai24_challenges/topcow_challenge_final/evals/mra_crown23_skr_bindice_nomir_4folds --num_folds 4 --gpu 3

# #FIXME: MRA-Val   5 folds with SkeletonRecall + BinDice + NoMirror + With/Without PostProcessing 
# python /home/hasna/miccai24_challenges/topcow_challenge_final/src/inference_postprocess.py --model_bin_path /home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/datasetnnUNet_trained_models/Dataset802_TopCoWSegBinMRA/nnUNetTrainerSkeletonRecall__nnUNetPlans__3d_fullres --model_mul_path /home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/datasetnnUNet_trained_models/Dataset808_TopCoWSegMRA/nnUNetTrainerSkeletonRecallBinDiceNoMirroring__nnUNetPlans__3d_fullres --input_images /home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/dataset/nnUNet_raw/imagesVal --save_folder /home/hasna/miccai24_challenges/topcow_challenge_final/evals/mra_val_skr_bindice_nomir_5folds --num_folds 5 --gpu 3
# python /home/hasna/miccai24_challenges/topcow_challenge_final/src/inference_postprocess.py --model_bin_path /home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/datasetnnUNet_trained_models/Dataset802_TopCoWSegBinMRA/nnUNetTrainerSkeletonRecall__nnUNetPlans__3d_fullres --model_mul_path /home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/datasetnnUNet_trained_models/Dataset808_TopCoWSegMRA/nnUNetTrainerSkeletonRecallBinDiceNoMirroring__nnUNetPlans__3d_fullres --input_images /home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/dataset/nnUNet_raw/imagesVal --save_folder /home/hasna/miccai24_challenges/topcow_challenge_final/evals/mra_val_skr_bindice_nomir_5folds_intermediates --num_folds 5 --gpu 3 --interm_save True



# #FIXME: MRA   5 folds with SkeletonRecall + BinDice + NoMirror + With/Without PostProcessing  + with different min_vol values (30, 40, 50) 
# python /home/hasna/miccai24_challenges/topcow_challenge_final/src/run_postprocessing.py --save_folder /home/hasna/miccai24_challenges/topcow_challenge_final/evals/mra_skr_bindice_nomir_5folds --num_folds 5 --min_vol 30

# '''

# ''' --------------------------------------------------------------- '''
# parser = argparse.ArgumentParser(description='Get all command line arguments.')
# parser.add_argument('--save_folder', type=str, default='', help='Specify the dir to al the trained models')
# parser.add_argument('--min_dist', type=float, default=15, help='Specify the minimum distance to link')
# parser.add_argument('--min_vol', type=int, default=20, help='Specify the minimum volume to remove')
# parser.add_argument('--num_folds', type=int, default=5, help='Specify the number of folds to be used')
# parser.add_argument('--interm_save', type=bool, default=False, help='Specify if it saves the intermediate post-processing results')
# ''' --------------------------------------------------------------- '''



# def main(args):
#     # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
#     # if args.interm_save: os.makedirs(os.path.join(args.save_folder, 'steps'), exist_ok=True)
        
#     ### Iterate over each input image for inference
#     for fold in natsorted(os.listdir(os.path.join(args.save_folder, 'without_pp'))):
#         os.makedirs(os.path.join(args.save_folder, f'with_pp_{args.min_vol}', fold), exist_ok=True)
#         for file in tqdm([file for file in natsorted(os.listdir(os.path.join(args.save_folder, 'without_pp', fold))) if file.endswith('.nii.gz')]):
#             img = nib.load(os.path.join(args.save_folder, 'without_pp', fold, file))
#             pred_array_mul = img.get_fdata()
#             pred_array_bin = nib.load(os.path.join(args.save_folder, 'bin', fold, file)).get_fdata()

#             ## --- Postprocessing --- :
#             ## Step 1: Replace the BG voxels in multiclass pred & Delete the small disconnected volumes (<20 voxels)
#             pred_array_mul = replace_background_with_nearest_label(pred_array_mul, pred_array_bin)
#             # if args.interm_save: nib.save(nib.Nifti1Image(pred_array_mul, img.affine), os.path.join(args.save_folder, 'steps', 'step1_'+file))
            
#             cleaned_arr = clean_small_components(pred_array_mul, args.min_vol)
#             # if args.interm_save: nib.save(nib.Nifti1Image(cleaned_arr, img.affine), os.path.join(args.save_folder, 'steps', 'step2_'+file))

#             ## Step 2: Connect the disconnected components (linkin parts)
#             new_arr = np.zeros(cleaned_arr.shape)  # New array to store the modifications (bridges)
#             for i in np.unique(cleaned_arr):  # Iterating over each class
#                 binary_image = 1*(cleaned_arr==i)  # Dealing with 1 label class per time
#                 _, nums = label(binary_image)

#                 if nums > 1:  # If for that specific label, there are more than 1 component (thus, disconnexions)
#                     dilated = link_components(binary_image, cleaned_arr)  # Link between the two closest points of the two closest components by dilating the line connecting them and multiply it by their class label
#                     dilated_corrected = 1*( (dilated - 1*(cleaned_arr>0)) > 0)  # Get only the extra voxels to not interfere with what was originally there..
#                     dilated_corrected = i*( (dilated_corrected - 1*(new_arr>0)) > 0)  # Get only the extra voxels to not interfere with what was previously dilated from other accumulated labels...
                    
#                     _, nums_2 = label(1*(dilated_corrected>0))  #FIXME:
#                     if nums_2 > 1:  #FIXME:  meaning that the link got disrupted by subtracting the original.. so we keep the link instead
#                         dilated_corrected = i*(dilated > 0)  #FIXME:
#                     new_arr += (dilated_corrected.astype(np.uint8))

#             # corrected = new_arr*((1*(new_arr>0) - 1*(cleaned_arr>0)) > 0)  # Getting the final extra voxels (from bridges) to not interfer with the original + multiplying by the new_arr to get the classes in the binary (subtraction)
#             corrected = new_arr  #FIXME: the one above is the original
#             cleaned_arr[corrected != 0] = 0  # Replace only where there is no interferance with the other labels !!
#             cleaned_arr += (corrected.astype(np.uint8))  # Adding the final extra voxel to the original, after previously replacing the original with 0 where it should..
#             # if args.interm_save: nib.save(nib.Nifti1Image(cleaned_arr, img.affine), os.path.join(args.save_folder, 'steps', 'step3_'+file))

#             pred_array = final_check(cleaned_arr)  # Final checking: if a certain label has disconnected components, the smallest is either deleted (if it's not connected to anything) or takes the label of its neighbour (to ensure continuity)..
#             pred_array[pred_array == 13] = 15  # Last one just in case..
#             nib.save(nib.Nifti1Image(pred_array, img.affine), os.path.join(args.save_folder, f'with_pp_{args.min_vol}', fold, file))


# # %%
# if __name__ == "__main__":
#     args = parser.parse_args()
#     main(args)
