import os
import time
import torch
import argparse
import numpy as np
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
'''

''' --------------------------------------------------------------- '''
parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--model_bin_path', type=str, required=True, help='Specify the path to the binary predictions')
parser.add_argument('--model_mul_path', type=str, required=True, help='Specify the path to the multiclass model')
parser.add_argument('--input_images', type=str, default='', help='Specify the dir to al the input images')
parser.add_argument('--save_folder', type=str, default='', help='Specify the dir to al the trained models')
parser.add_argument('--min_dist', type=float, default=15, help='Specify the minimum distance to link')
parser.add_argument('--min_vol', type=float, default=20, help='Specify the minimum volume to remove')
parser.add_argument('--num_folds', type=int, default=5, help='Specify the number of folds to be used')
parser.add_argument('--interm_save', type=bool, default=False, help='Specify if it saves the intermediate post-processing results')
parser.add_argument('--gpu', type=int, default=0, help='Specify the gpu to be used')
''' --------------------------------------------------------------- '''




def write_array_as_image_file(array, input_path, output_path):
    input_img = sitk.ReadImage(input_path)

    ## Reorder array from (x,y,z) to (z,y,x) before using sitk.GetImageFromArray
    array = array.transpose((2, 1, 0)).astype(np.uint8)

    ## Converting prediction array back to SimpleITK copying the metadata from the original image
    seg_mask = sitk.GetImageFromArray(array.astype(np.uint8))

    ## Copies the Origin, Spacing, and Direction from the source image
    seg_mask.CopyInformation(input_img)

    sitk.WriteImage(
        seg_mask,
        output_path,
        useCompression=True,
    )



def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    ### Define the binary predictor and initialize its weights        
    predictor_bin = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )    
    predictor_bin.initialize_from_trained_model_folder(
        args.model_bin_path,
        use_folds=('all'),
        checkpoint_name='checkpoint_best.pth',
    )

    ### Define the multiclass predictor and initialize its weights
    predictor_mul = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,  #FIXME: disabling inference mirroring, as it was also trained with NoMirroring trainer option
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )
    predictor_mul.initialize_from_trained_model_folder(
        args.model_mul_path,
        use_folds=(0, 1, 2, 3, 4,),
        checkpoint_name='checkpoint_best.pth',
    )

    os.makedirs(os.path.join(args.save_folder, 'without_pp'), exist_ok=True)
    os.makedirs(os.path.join(args.save_folder, 'with_pp'), exist_ok=True)
    os.makedirs(os.path.join(args.save_folder, 'bin'), exist_ok=True)
    if args.interm_save: os.makedirs(os.path.join(args.save_folder, 'steps'), exist_ok=True)
        
    ### Iterate over each input image for inference
    for file in tqdm([file for file in natsorted(os.listdir(args.input_images)) if file.endswith('.nii.gz')]):
        input_path = os.path.join(args.input_images, file)
        input_array, input_props = SimpleITKIO().read_images([input_path])  # Read input image with its properties

        pred_array_bin = predictor_bin.predict_single_npy_array(input_array, input_props)
        pred_array_bin = pred_array_bin.transpose((2, 1, 0)).astype(np.uint8)
        write_array_as_image_file(pred_array_bin, input_path, os.path.join(args.save_folder, 'bin', file))

        pred_array_mul = predictor_mul.predict_single_npy_array(input_array, input_props)
        pred_array_mul[pred_array_mul == 13] = 15
        pred_array_mul = pred_array_mul.transpose((2, 1, 0)).astype(np.uint8)

        write_array_as_image_file(pred_array_mul, input_path, os.path.join(args.save_folder, 'without_pp', file))


        ## --- Postprocessing --- :
        ## Step 1: Replace the BG voxels in multiclass pred & Delete the small disconnected volumes (<20 voxels)
        pred_array_mul = replace_background_with_nearest_label(pred_array_mul, pred_array_bin)
        if args.interm_save: write_array_as_image_file(pred_array_mul, input_path, os.path.join(args.save_folder, 'steps', 'step1_'+file))
        
        cleaned_arr = clean_small_components(pred_array_mul)
        if args.interm_save: write_array_as_image_file(cleaned_arr, input_path, os.path.join(args.save_folder, 'steps', 'step2_'+file))

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
        if args.interm_save: write_array_as_image_file(cleaned_arr, input_path, os.path.join(args.save_folder, 'steps', 'step3_'+file))

        pred_array = final_check(cleaned_arr)  # Final checking: if a certain label has disconnected components, the smallest is either deleted (if it's not connected to anything) or takes the label of its neighbour (to ensure continuity)..
        pred_array[pred_array == 13] = 15  # Last one just in case..
        write_array_as_image_file(pred_array, input_path, os.path.join(args.save_folder, 'with_pp', file))


# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
