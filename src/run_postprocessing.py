
import os
import argparse
import numpy as np
import nibabel as nib

from tqdm import tqdm
from natsort import natsorted
from scipy.ndimage import label, generate_binary_structure
from skimage.morphology import skeletonize

from post_processings import *


'''
cmd:
python ./src/run_postprocessing.py --preds_mul_folder ./nnUNetTrainerSkeletonRecallBinDiceNoMirroring__nnUNetPlans__3d_fullres --preds_bin_folder ./nnUNetTrainerSkeletonRecall__nnUNetPlans__3d_fullres --save_folder ./evals

'''

''' --------------------------------------------------------------- '''
parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--save_folder', type=str, default='', help='Specify the dir to al the trained models')
parser.add_argument('--preds_mul_folder', type=str, default='', help='Specify the dir to the multiclass preds')
parser.add_argument('--preds_bin_folder', type=str, default='', help='Specify the dir to the binary preds')
parser.add_argument('--min_dist', type=float, default=15, help='Specify the minimum distance to link')
parser.add_argument('--min_vol', type=int, default=20, help='Specify the minimum volume to remove')
''' --------------------------------------------------------------- '''



def main(args):
    os.makedirs(args.save_folder, exist_ok=True)

    for file in tqdm([file for file in natsorted(os.listdir(args.preds_mul_folder)) if file.endswith('.nii.gz')]):
        img = nib.load(os.path.join(args.preds_mul_folder, file))
        pred_array_mul = img.get_fdata()
        pred_array_bin = nib.load(os.path.join(args.preds_bin_folder, file)).get_fdata()

        ## --- Postprocessing --- :
        ## Step 1: Background filling with binary mask
        pred_array_mul = fill_missing_voxels(pred_array_mul, pred_array_bin, 3)
        

        ## Step 2: Small components removal
        structure = generate_binary_structure(rank=3, connectivity=3)  # 3D, 26-connectivity
        pred_array_mul = clean_small_components(pred_array_mul, args.min_vol, structure=structure)


        ## Step 3: Union of disconnected components
        new_arr = np.zeros(pred_array_mul.shape)  # New array to store the modifications (bridges)
        lab_list = np.unique(pred_array_mul[pred_array_mul != 0]) if args.mode == 'general' else [8, 9, 10, 13, 15]
        for lab in lab_list:
            binary_image = (pred_array_mul==lab).astype(int)  # Dealing with 1 label class per time
            skel = skeletonize(binary_image)
            _, n_labels = label(binary_image, structure=structure)

            if n_labels > 1:  # If for that specific label, there are more than 1 component (thus, disconnexions)
                print(f'Label {lab} has {n_labels} components')
                labeled_image, num_features = label(binary_image, structure=structure)
                dilated_image = binary_image.copy()

                while num_features > 1:  # If one class have disconnected components, we will try to link them when possible..
                    # Find the two components:
                    component_coords = [np.argwhere(labeled_image == i + 1) for i in range(num_features)]
                    comp1, comp2, distance, point1, point2 = get_closest_points(component_coords)
                    prev_comp1, prev_comp2, prev_dist, prev_point1, prev_point2 = comp1, comp2, distance, point1, point2
                    print(f'Label {lab}. Closest components {comp1} and {comp2} have a distance of {distance}.')
                    if distance < args.min_dist:  # If the distance between the two found closest components is not that big, then we link them.. otherwise we skip
                        skel1 = skel * (labeled_image == comp1)
                        skel2 = skel * (labeled_image == comp2)

                        #NOTE: In some cases, the skeletonization can return an empty array, so to bypass the issue we add the clodest point from earlier just to have something..                            
                        if skel1.sum() == 0:
                            skel1[tuple(point1)] = 1
                        if skel2.sum() == 0:
                            skel2[tuple(point2)] = 1

                        ## Fit a Curve to the skeleton
                        curve_pts, p1, p2 = fit_curve_all(skel1, skel2, point1, point2, log_file)
        
                        # Find the indices of the skeleton endpoints (because the curve_pts include the neighboring points of p1 and p2.. which we don't want as they exist already..)
                        index_array_1 = np.where(np.all(curve_pts == np.array(p1), axis=1))[0] if len(np.where(np.all(curve_pts == np.array(p1), axis=1))[0]) != 0 else np.array(0) #NOTE: Using this trick in case the point p1 or p2 are not in the curve_pts, in which case replace by the first index or last...
                        index_array_2 = np.where(np.all(curve_pts == np.array(p2), axis=1))[0] if len(np.where(np.all(curve_pts == np.array(p2), axis=1))[0]) != 0 else np.array(len(curve_pts)) #NOTE: Using this trick in case the point p1 or p2 are not in the curve_pts, in which case replace by the first index or last...
                        
                        filtered_curve_pts = curve_pts[index_array_1.item():index_array_2.item()+1]  ## Keeping only the curve points between the two skeleton endpoints
                        new_skeleton = add_curve_to_mask(np.zeros(labeled_image.shape), filtered_curve_pts)
        

                        ## Dilate the curve
                        new_mask = dilate_curve(new_skeleton, binary_image.copy(), filtered_curve_pts[0], filtered_curve_pts[-1])
                        _, tmp_num_features = label(new_mask, structure=structure)
                        if tmp_num_features > 1:  #NOTE: In the case where even with the dilation, parts are disconnected, we increase the size of the struct_elem to try to connect them still..
                            print('Need to increase the size of the struct_elem to connect the two components')
                            new_mask = dilate_curve(new_skeleton, binary_image.copy(), filtered_curve_pts[0], filtered_curve_pts[-1], larger=True)
                        dilated_image[new_mask>0] = 1

                        # Check again the number of components, and if they are still more than 1 then stay in the loop and connect the next ones..
                        # Check again the number of components, if they are still more than 1 but it's stuck in a loop, exit and go to another label..
                        labeled_image, num_features = label(dilated_image, structure=structure)
                        if num_features > 1:
                            component_coords = [np.argwhere(labeled_image == i + 1) for i in range(num_features)]
                            comp1, comp2, distance, point1, point2 = get_closest_points(component_coords)
                            if (comp1 == prev_comp1 and
                                comp2 == prev_comp2 and
                                distance == prev_dist and
                                np.array_equal(point1, prev_point1) and
                                np.array_equal(point2, prev_point2)):
                                print('Stuck in a loop. Moving on to the next label')
                                break
                    else:
                        break

                dilated_corrected = 1*( (labeled_image - 1*(pred_array_mul>0)) > 0)  # Get only the extra voxels to not interfere with what was originally there..
                dilated_corrected = lab*( (dilated_corrected - 1*(new_arr>0)) > 0)  # Get only the extra voxels to not interfere with what was previously dilated from other accumulated labels...
                new_arr += (dilated_corrected.astype(np.uint8))
                print(f'Label {lab}. Added dilated_corrected volume is: {dilated_corrected.astype(np.uint8).sum()}.')

        corrected = new_arr*((1*(new_arr>0) - 1*(pred_array_mul>0)) > 0)  # Getting the final extra voxels (from bridges) to not interfer with the original + multiplying by the new_arr to get the classes in the binary (subtraction)
        pred_array_mul[corrected != 0] = 0  # Replace only where there is no interferance with the other labels !!
        pred_array_mul += (corrected.astype(np.uint8))  # Adding the final extra voxel to the original, after previously replacing the original with 0 where it should..


        ## Step 4: Final continuity check
        structure_26 = generate_binary_structure(rank=3, connectivity=3)  # 3D, 26-connectivity
        pred_array_mul = final_check(pred_array_mul, structure_26, min_dist=15)
        pred_array_mul[pred_array_mul == 13] = 15  # Last one to adapt to the TopCoW dataset labels, just in case..


        nib.save(nib.Nifti1Image(pred_array_mul, img.affine), os.path.join(args.save_folder, file))


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
