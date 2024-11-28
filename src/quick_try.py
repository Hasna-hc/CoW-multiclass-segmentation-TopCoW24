import os
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
export PYTHONPATH=/home/hasna/miccai24_challenges/topcow_challenge_final
export nnUNet_raw='/home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/dataset/nnUNet_raw'
export nnUNet_preprocessed='/home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/dataset/preprocessed'
export nnUNet_results='/home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/datasetnnUNet_trained_models'


#TODO:  CTA   5 folds with SkeletonRecall + BinDice + NoMirror + With/Without PostProcessing
python /home/hasna/miccai24_challenges/topcow_challenge_final/src/crossval_postprocess.py --model_bin_path /home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/datasetnnUNet_trained_models/Dataset807_TopCoWSegBinCTAMRA/nnUNetTrainerSkeletonRecall__nnUNetPlans__3d_fullres_ps --model_mul_path /home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/datasetnnUNet_trained_models/Dataset806_TopCoWSegCTAMRA/nnUNetTrainerSkeletonRecallBinDiceNoMirroring__nnUNetPlans__3d_fullres_ps --input_images /home/hasna/miccai24_challenges/topcow_challenge/nnunet_dir/dataset/nnUNet_raw/Dataset703_TopCoWDetCTACropExtendedMulSegMask/imagesTr --preds_folder /home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/datasetnnUNet_trained_models/Dataset806_TopCoWSegCTAMRA/nnUNetTrainerSkeletonRecallBinDiceNoMirroring__nnUNetPlans__3d_fullres_ps --save_folder /home/hasna/miccai24_challenges/topcow_challenge_final/evals/cta_skr_bindice_nomir_5folds --num_folds 5 --mod 'ct' --gpu 3


#TODO:  MRA   5 folds with SkeletonRecall + BinDice + NoMirror + With/Without PostProcessing
python /home/hasna/miccai24_challenges/topcow_challenge_final/src/crossval_postprocess.py --model_bin_path /home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/datasetnnUNet_trained_models/Dataset802_TopCoWSegBinMRA/nnUNetTrainerSkeletonRecall__nnUNetPlans__3d_fullres --model_mul_path /home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/datasetnnUNet_trained_models/Dataset808_TopCoWSegMRA/nnUNetTrainerSkeletonRecallBinDiceNoMirroring__nnUNetPlans__3d_fullres --input_images /home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/dataset/nnUNet_raw/Dataset808_TopCoWSegMRA/imagesTr --preds_folder /home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/datasetnnUNet_trained_models/Dataset808_TopCoWSegMRA/nnUNetTrainerSkeletonRecallBinDiceNoMirroring__nnUNetPlans__3d_fullres --save_folder /home/hasna/miccai24_challenges/topcow_challenge_final/evals/mra_skr_bindice_nomir_5folds --num_folds 5 --mod 'mr' --gpu 3


python /home/hasna/miccai24_challenges/topcow_challenge_final/src/quick_try.py --model_bin_path /home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/datasetnnUNet_trained_models/Dataset802_TopCoWSegBinMRA/nnUNetTrainerSkeletonRecall__nnUNetPlans__3d_fullres --model_mul_path /home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/datasetnnUNet_trained_models/Dataset808_TopCoWSegMRA/nnUNetTrainerSkeletonRecallBinDiceNoMirroring__nnUNetPlans__3d_fullres --input_images /home/hasna/miccai24_challenges/topcow_challenge_final/to_delete --preds_folder /home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/datasetnnUNet_trained_models/Dataset808_TopCoWSegMRA/nnUNetTrainerSkeletonRecallBinDiceNoMirroring__nnUNetPlans__3d_fullres --save_folder /home/hasna/miccai24_challenges/topcow_challenge_final/evals/mra_skr_bindice_nomir_5folds_binary --num_folds 5 --mod 'mr' --gpu 0
python /home/hasna/miccai24_challenges/topcow_challenge_final/src/quick_try.py --model_bin_path /home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/datasetnnUNet_trained_models/Dataset807_TopCoWSegBinCTAMRA/nnUNetTrainerSkeletonRecall__nnUNetPlans__3d_fullres_ps --input_images /home/hasna/miccai24_challenges/topcow_challenge_final/tmp_todelete --save_folder /home/hasna/miccai24_challenges/topcow_challenge_final/tmp_cta_bin_todelete --mod 'ct' --gpu 0


python /home/hasna/miccai24_challenges/topcow_challenge_final/src/quick_try.py --model_bin_path /home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/datasetnnUNet_trained_models/Dataset807_TopCoWSegBinCTAMRA/nnUNetTrainerSkeletonRecall__nnUNetPlans__3d_fullres_ps --input_images /home/hasna/miccai24_challenges/topcow_challenge_final/nnunet_dir/dataset/nnUNet_raw/Dataset807_TopCoWSegBinCTAMRA/imagesTr --save_folder /home/hasna/miccai24_challenges/topcow_challenge_final/evals/cta_skr_bindice_nomir_5folds --mod 'ct' --gpu 0

'''

''' --------------------------------------------------------------- '''
parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--model_bin_path', type=str, required=True, help='Specify the path to the binary predictions')
# parser.add_argument('--model_mul_path', type=str, required=True, help='Specify the path to the multiclass model')
parser.add_argument('--input_images', type=str, default='', help='Specify the dir to al the input images')
# parser.add_argument('--preds_folder', type=str, default='', help='Specify the dir to al the preds per fold')
parser.add_argument('--save_folder', type=str, default='', help='Specify the dir to al the trained models')
parser.add_argument('--min_dist', type=float, default=15, help='Specify the minimum distance to link')
parser.add_argument('--min_vol', type=float, default=20, help='Specify the minimum volume to remove')
# parser.add_argument('--num_folds', type=int, default=5, help='Specify the number of folds to be used')
parser.add_argument('--mod', type=str, default='', help='Specify the modality to be used')
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

    ### Define the multiclass predictor and initialize its weights
    model_bin_path = args.model_bin_path #'/home/hasna/miccai24_challenges/topcow_challenge/nnunet_dir/datasetnnUNet_trained_models/Dataset802_TopCoWSegBinMRA/nnUNetTrainerSkeletonRecall__nnUNetPlans__3d_fullres'

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
        model_bin_path,
        use_folds=('all'),
        checkpoint_name='checkpoint_best.pth',
    )

    os.makedirs(os.path.join(args.save_folder, "bin"), exist_ok=True)
    for file in tqdm(natsorted(os.listdir(os.path.join(args.input_images)))):
        if file.startswith(f'topcow_{args.mod}_'):
            input_path = os.path.join(args.input_images, file)
            input_array, input_props = SimpleITKIO().read_images([input_path])  # Read input image with its properties

            pred_array_bin = predictor_bin.predict_single_npy_array(input_array, input_props)
            pred_array_bin = pred_array_bin.transpose((2, 1, 0)).astype(np.uint8)
            write_array_as_image_file(pred_array_bin, input_path, os.path.join(args.save_folder, "bin", file))

# %%
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
