"""
The following is a simple example algorithm for task 1: CoW multi-class segmentation.

It can be run locally (good for setting up and debugging your algorithm) 
or in a docker container (required for submission to grand-challenge.org).

If you use this template you can simply replace the `your_segmentation_algorithm` function with your own algorithm. 
The suggested inputs are np.arrays of the MR and CT images respectively, the output is your segmentation prediction array.

To run your algorithm, execute the `inference.py` script (inference.py is the entry point for the docker container). 
NOTE: You don't need to change anything in the inference.py script!

The relevant paths are as follows:
    input_path: contains the input images inside the folders /images/head-mr-angio or /images/head-ct-angio
        Docker: /input
        Local: ./test/input
    output_path: output predictions are stored inside the folder /images/cow-multiclass-segmentation
        Docker: /output
        Local: ./test/output
    resource_path: any additional resources needed for the algorithm during inference can be stored here (optional)
        Docker: resources
        Local: ./resources

Before submitting to grand-challenge.org, you must ensure that your algorithm runs in the docker container. To do this, run 
  ./test_run.sh
This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:
  ./save.sh
"""
import os
import torch
import numpy as np
from scipy.ndimage import label

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from post_processings import *

#######################################################################################
# TODO: 
# Choose your TRACK. Track is either 'MR' or 'CT'.
TRACK = 'MR' # or 'CT'
# END OF TODO
#######################################################################################

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'


# ------------------------------------------------------------------

def your_segmentation_algorithm(*, mr_input_array: np.array, mra_props, ct_input_array: np.array, cta_props) -> np.array:
    """
    This is an example of a prediction algorithm.
    It is a dummy algorithm that returns an array of the correct shape with randomly assigned integers between 0 and 12.
    args:
        mr_input_array: np.array - input image for MR track
        ct_input_array: np.array - input image for CT track
    returns:
        np.array - prediction
    """

    #######################################################################################
    # TODO: place your own prediction algorithm here.
    # You are free to remove everything! Just return to us an npy in (x,y,z).
    # You can use the input_head_mr_angiography and/or input_head_ct_angiography
    # to make your prediction.

    # NOTE: the prediction array must have the same shape as the input image of the chosen track!

    # NOTE: If you extract the array from SimpleITK, note that
    #              SimpleITK npy array axis order is (z,y,x).
    #              Then you might have to transpose this to (x,y,z)

    #######################################################################################

    # load and initialize your model here
    # model = ...
    # device = ...

    # You can also place and load additional files in the resources folder
    # with open(resources / "some_resource.txt", "r") as f:
    #     print(f.read())

    # For now, let us set make bogus predictions
    # output_shape = tuple()
    if TRACK == 'CT':
        input_array = ct_input_array
        input_props = cta_props
    elif TRACK == 'MR':
        input_array = mr_input_array
        input_props = mra_props
    else:
        raise ValueError("Invalid TRACK chosen. Choose either 'MR' or 'CT'.")
    
    # # Set the seed for reproducibility
    # np.random.seed(42)
    # # Randomly assign values between 0 and 12 to the array
    # pred_array = np.random.randint(0, 13, output_shape)

    ## Initialization of the Multiclass predictor
    predictor_mul = nnUNetPredictor(
            tile_step_size=0.5,  #FIXME: was 0.5
            use_gaussian=True,
            use_mirroring=False,  #FIXME: was True
            perform_everything_on_device=True,
            device=torch.device('cuda', 0),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False
        )

    model_mul_path = 'nn_unet/nnUNet_results/Dataset801_TopCoWSegMRA/nnUNetTrainerSkeletonRecallBinDice__nnUNetPlans__3d_fullres'
    predictor_mul.initialize_from_trained_model_folder(
        model_mul_path,
        use_folds=(0, 1),
        checkpoint_name='checkpoint_best.pth',
    )

    ## Multiclass prediction
    pred_array_mul = predictor_mul.predict_single_npy_array(input_array, input_props)
    pred_array_mul[pred_array_mul == 13] = 15
    pred_array_mul = pred_array_mul.transpose((2, 1, 0)).astype(np.uint8)

    
    ## -----------------------------------------------------------------------------------------------------------
    ##                                              post-processing
    ## -----------------------------------------------------------------------------------------------------------
    # patch_size = (160, 192, 64)
    # num_sw = calculate_sliding_windows(patch_size, pred_array_mul.shape)  # Check the number of patches needed for the sliwing-window
    # if num_sw <= 150:  # In this case, the image is not too big and the full post-processing can be applied (including the binary part..)
    ## Initialization of the Binary predictor (1 single model)
    model_bin_path = 'nn_unet/nnUNet_results/Dataset802_TopCoWSegBinMRA/nnUNetTrainerSkeletonRecall__nnUNetPlans__3d_fullres'
    predictor_bin = nnUNetPredictor(
        tile_step_size=0.5,  #FIXME: was originally 0.5  TODO: was 0.6
        use_gaussian=True,
        use_mirroring=True,  #FIXME: was originally True
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
    pred_array_bin = predictor_bin.predict_single_npy_array(input_array, input_props)
    pred_array_bin = pred_array_bin.transpose((2, 1, 0)).astype(np.uint8)
    ## Replace the BG voxels in multiclass pred, that are FG in binary pred, with the closest multiclass label..
    pred_array_mul = replace_background_with_nearest_label(pred_array_mul, pred_array_bin)

    cleaned_arr = clean_small_components(pred_array_mul)  # Cleaning the small disconnected volumes
    
    new_arr = np.zeros(cleaned_arr.shape)  # New array to store the modifications (bridges)
    for i in np.unique(cleaned_arr):
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

    pred_array = final_check(cleaned_arr)  # Final checking: if a certain label has disconnected components, the smallest is either deleted (if it's not connected to anything) or takes the label of its neighbour (to ensure continuity)..
    pred_array[pred_array == 13] = 15  # Last one just in case..
    
    return pred_array