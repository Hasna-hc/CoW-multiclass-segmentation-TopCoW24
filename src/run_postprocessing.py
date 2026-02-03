
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




