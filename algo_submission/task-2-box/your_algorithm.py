"""
The following is a simple example algorithm for task 2: CoW detection/localization.

It can be run locally (good for setting up and debugging your algorithm) 
or in a docker container (required for submission to grand-challenge.org).

If you use this template you can simply replace the `your_detection_algorithm` function with your own algorithm. 
The suggested inputs are np.arrays of the MR and CT images respectively, the output should be a dictionary containing 
the predicted bounding box in the form
    {
    "size": [x, y, z],
    "location": [x, y, z]
    }

To run your algorithm, execute the `inference.py` script (inference.py is the entry point for the docker container). 
NOTE: You don't need to change anything in the inference.py script!

The relevant paths are as follows:
    input_path: contains the input images inside the folders /images/head-mr-angio or /images/head-ct-angio
        Docker: /input
        Local: ./test/input
    output_path: output predictions are stored as cow-roi.json
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
# import numpy as np
import os
import sys
import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.spatial.distance import cdist
from scipy.ndimage import find_objects, label
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


repo_path = os.getcwd()  # /opt/app/
os.chdir(repo_path) # /opt/app/
sys.path.insert(0,str(repo_path)) if str(repo_path) not in sys.path else None



def clean_small_components(segmentation, volume_threshold=20):
    for i in np.unique(segmentation):
        # Label the connected components
        labeled_array, num_features = label(segmentation==i)

        # Iterate over the labeled components
        for component_label in range(1, num_features + 1):  # Skip background (label 0)
            component_mask = (labeled_array == component_label)
            
            # Calculate the volume of the current component
            volume = np.sum(component_mask)
            # If the component volume is smaller than the threshold, remove it
            if volume < volume_threshold:
                segmentation[component_mask] = 0  # Remove the component by setting it to 0

        # Resulting segmentation with small components removed
        cleaned_segmentation = segmentation  #useless pass, I can directly return segmentation..
    return cleaned_segmentation


def link_far_components(binary_image, min_distance=80):
    labeled_image, num_features = label(binary_image)
    dilated_image = binary_image.copy()

    while num_features > 1:
        component_coords = [np.argwhere(labeled_image == i + 1) for i in range(num_features)]
        component_volumes = [len(coords) for coords in component_coords]  # Calculate volumes of components

        # Initialize a list to hold distances to the closest point of other components
        closest_distances = []

        for i in range(num_features):
            distances = []
            for j in range(num_features):
                if i != j:
                    # Compute distances between all points in component i and all points in component j
                    dist_matrix = cdist(component_coords[i], component_coords[j])
                    min_dist_to_other = np.min(dist_matrix)  # Find minimum distance to any point in component j
                    distances.append(min_dist_to_other)

            # Get the closest distance to any other component
            closest_distances.append(np.min(distances) if distances else np.inf)

        # Create a list of tuples (index, volume, closest_distance)
        component_info = [
            (i + 1, component_volumes[i], closest_distances[i])  # 1-based index
            for i in range(num_features)
        ]

        # Sort components by closest distance and volume
        component_info.sort(key=lambda x: (x[2], x[1]))  # Sort by closest distance first, then by volume

        # Check if all remaining components are close enough
        if all(comp[2] <= min_distance for comp in component_info):
            # print("All remaining components are close enough. Exiting.")
            break

        # Identify the smallest component with a maximum distance greater than min_distance
        component_to_remove = None
        for comp in component_info:
            if comp[2] > min_distance:
                component_to_remove = comp[0]  # Get the index of the component to remove
                break

        # If a component to remove was found, delete it
        if component_to_remove is not None:
            dilated_image[labeled_image == component_to_remove] = 0  # Set to background
            # print(f'Removed component: {component_to_remove}')

            # Recalculate labeled_image and num_features after removal
            labeled_image, num_features = label(dilated_image)

    return dilated_image


def get_bounding_box_3d(mask):
    """
    Given a 3D binary mask, returns the bounding box size and location in the format:
    (size_x, size_y, size_z), (min_x, min_y, min_z)
    """
    slices = find_objects(mask)
    if slices and slices[0] is not None:
        min_x, min_y, min_z = slices[0][0].start, slices[0][1].start, slices[0][2].start
        max_x, max_y, max_z = slices[0][0].stop, slices[0][1].stop, slices[0][2].stop
        size = (max_x - min_x, max_y - min_y, max_z - min_z)
        location = (min_x, min_y, min_z)
        return size, location
    else:
        return None, None


def your_detection_algorithm(*, mr_input_array: np.array, mra_props, ct_input_array: np.array, cta_props) -> dict:
    """
    This is an example of a prediction algorithm.
    It is a dummy algorithm that returns a bounding box in the center of the image
    args:
        mr_input_array: np.array - input image for MR track
        ct_input_array: np.array - input image for CT track
    returns:
        dict - bounding box prediction in the form {"size": [x, y, z], "location": [x, y, z]}
    """
    #######################################################################################
    # TODO: place your own prediction algorithm here.
    # You are free to remove everything! Just return to us a dictionary containing 
    # the bounding box prediction in the form {"size": [x, y, z], "location": [x, y, z]}.
    # You can use the input_head_mr_angiography and/or input_head_ct_angiography
    # to make your prediction.

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
    
    # model_bin_path = os.path.join(repo_path, 'nn_unet/nnUNet_results/Dataset703_TopCoWDetCTACropExtendedMulSegMask/nnUNetTrainerSkeletonRecall__nnUNetPlans__3d_fullres')  #FIXME: For CTA
    model_bin_path = os.path.join(repo_path, 'nn_unet/nnUNet_results/Dataset705_TopCoWDetCTAextendedMaskCTAMRA/nnUNetTrainerSkeletonRecall__nnUNetPlans__3d_fullres_spacing_ps')  #FIXME: For CTA
    
    # model_bin_path = os.path.join(repo_path, 'nn_unet/nnUNet_results/Dataset704_TopCoWDetMRAextendedMask/nnUNetTrainerSkeletonRecall__nnUNetPlans__3d_fullres')  #FIXME: For MRA

    predictor_bin.initialize_from_trained_model_folder(
        model_bin_path,
        use_folds=(0, 1, 2, 3, 4),  #FIXME: for CTA
        # use_folds=(0, 1, 2),  #FIXME: for MRA
        checkpoint_name='checkpoint_best.pth',
    )
    
    
    pred_array_bin = predictor_bin.predict_single_npy_array(ct_input_array, cta_props)  #FIXME: CTA
    # pred_array_bin = predictor_bin.predict_single_npy_array(mr_input_array, mra_props)  #FIXME: MRA

    # Ensure the mask is binary (0s and 1s)
    mask = pred_array_bin.transpose((2, 1, 0)).astype(np.uint8)  #FIXME: ??

    ## Post-processing
    cleaned_arr = clean_small_components(mask, 80)  # Removing small disconnected volumes (smaller than 80..)
    mask = link_far_components(cleaned_arr, 80)  # Deleting disconnected components that are far away (farther than 80..)

    # Get the bounding box size and location
    size, location = get_bounding_box_3d(mask.astype(np.uint8))

    # For now, let us set make bogus predictions for the MR image
    # output_shape = mr_input_array.shape
    
    pred_dict = {
        "size": [s for s in size],
        "location": [l for l in location]
    }

    return pred_dict

