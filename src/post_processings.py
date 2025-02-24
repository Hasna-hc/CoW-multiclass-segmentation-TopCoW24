

import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import label, distance_transform_edt, binary_dilation, generate_binary_structure



def calculate_sliding_windows(patch_size, image_shape, overlap=0.5):
    """
    Calculate the number of sliding window patches that can be extracted from an image.
    Parameters:
    - patch_size: Tuple of integers representing the size of the patch (depth, height, width).
    - image_shape: Tuple of integers representing the shape of the image (depth, height, width).
    - overlap: Fraction representing the overlap between patches (default is 0.5).
    Returns:
    - total_patches: Integer representing the total number of sliding window patches.
    """
    total_patches = 1  # Start with 1 for multiplication

    for p_size, img_size in zip(patch_size, image_shape):
        # Calculate the number of steps in this dimension
        num_steps = img_size / p_size
        # Adjust for overlap
        adjusted_patches = num_steps / overlap
        # Round down to get the number of patches that fit
        total_patches *= int(adjusted_patches)

    return total_patches


def clean_small_components(segmentation, volume_threshold=20):
    """ 
    This code aims to remove all small components (<threshold) for each label.. 
    Parameters:
    - segmentation: 3D numpy array representing the multiclass segmentation mask.
    - volume_threshold: Integer representing the volume threshold (components should be removed if their volume is less than that value).
    Returns:
    - segmentation: 3D numpy array representing the cleaned multiclass segmentation mask.
    """

    for i in np.unique(segmentation):  # Iterate over each label to check their respective disconnected components
        # Label the connected components
        labeled_array, num_features = label(segmentation==i)

        # Iterate over the labeled components
        for component_label in range(1, num_features + 1):  # Skip background (label 0)
            component_mask = (labeled_array == component_label)
            
            # Calculate the volume of the current component
            volume = np.sum(component_mask)
            # If the component volume is smaller than the threshold, remove it by setting it to 0
            if volume < volume_threshold:
                segmentation[component_mask] = 0

    return segmentation


def replace_background_with_nearest_label(seg, gt):
    """
    Replace background voxels in 'seg' that are foreground in 'gt' with the nearest class label from 'seg'.
    Parameters:
    - seg: 3D numpy array representing the multiclass segmentation mask.
    - gt: 3D numpy array representing the binary ground truth mask.
    Returns:
    - seg_filled: 3D numpy array with the background voxels filled with the nearest class label.
    """
    # Find background voxels in 'seg' that are foreground in 'gt'
    coords_to_fill = np.where((seg == 0) & (gt > 0))

    if len(coords_to_fill[0]) == 0:  # Early exit if no coordinates to fill
        return seg
    
    # Create a mask for foreground classes in 'seg'
    foreground_mask = seg > 0
    
    # Compute the distance transform for the non-background regions (distances from the foreground to the background)
    distances, nearest_indices = distance_transform_edt(foreground_mask == 0, return_indices=True)
    
    # Create a copy of 'seg' to modify it
    seg_filled = seg.copy()
    
    # Replace the background voxels that are foreground in 'gt' with the nearest class label from 'seg'
    seg_filled[coords_to_fill] = seg[tuple(nearest_indices[:, coords_to_fill[0], coords_to_fill[1], coords_to_fill[2]])]
    
    return seg_filled


def draw_line(image, point1, point2):
    """
    Draws a straight line between two points in 3D.
    Parameters:
    - image: 3D numpy array representing the binary mask of a certain label.
    - point1: #TODO: 
    - point2: #TODO:
    Returns:
    - line_mask: 3D numpy array representing the line connecting the two points (same shape as the input image).
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    # Number of points along the line (based on the largest difference between coordinates)
    num_points = np.max(np.abs(point2 - point1)) + 1
    
    # Interpolate the coordinates between point1 and point2
    line_coords = np.stack([np.linspace(p1, p2, num_points) for p1, p2 in zip(point1, point2)], axis=-1).round().astype(int)
    
    # Create a binary mask for the line
    line_mask = np.zeros_like(image, dtype=bool)
    line_mask[tuple(line_coords.T)] = True
    
    return line_mask


def dilate_line(image, line_mask, dilation_size=1):
    """
    Dilates the 3D line by a given size and applies it back to the original image.
    Parameters:
    - image: 3D numpy array representing the binary mask of a certain label.
    - line_mask: 3D numpy array representing the binary mask of the line drawn between the two closest points.
    - dilation_size: Integer representing the number of iterations for the dilation operation.
    Returns:
    - image: 3D numpy array representing the binary mask of the prediction with the dilated line connecting the two closest points.
    """
    structuring_element = generate_binary_structure(3, 1)
    
    # Apply dilation to the line mask only
    dilated_line = binary_dilation(line_mask, structure=structuring_element, iterations=dilation_size)
    
    # Add the dilated line in the original image (link the components)
    image[dilated_line] = 1
    
    return image


def link_components(binary_image, min_distance=15):
    """ 
    This code is used inside a loop that iterates over each class to link disconnected components.
    Parameters:
    - binary_image: 3D numpy array representing the binary mask of a certain label.
    - min_distance: Integer representing the minimum distance between two components to be linked.
    Returns:
    - dilated_image: 3D numpy array representing the resulting mask with the linked components.
    """

    labeled_image, num_features = label(binary_image)
    dilated_image = binary_image.copy()

    while num_features > 1:  # If one class have disconnected components, we will try to link them when possible..
        component_coords = [np.argwhere(labeled_image == i + 1) for i in range(num_features)]
        # Find closest pairs of components
        closest_pairs = []
        for i in range(len(component_coords)):
            for j in range(i + 1, len(component_coords)):
                # Compute distances between all points in component i and component j
                dist_matrix = cdist(component_coords[i], component_coords[j])
                min_dist = np.min(dist_matrix)  # Find the minimum distance
                min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
                
                # Add the result to closest_pairs as (component i, component j, minimum distance, point1, point2)
                closest_pairs.append((i + 1, j + 1, min_dist, component_coords[i][min_idx[0]], component_coords[j][min_idx[1]]))

        # Sort closest pairs by minimum distance
        closest_pairs.sort(key=lambda x: x[2])

        # Output the closest pairs: here we only keep the first two closest points, so we can start with them, then iterate again and link them to others if needed..
        for pair in closest_pairs:
            comp1, comp2, distance, point1, point2 = pair
            break

        if distance < 15: #min_distance: # If the distance between the two found closest components is not that big, then we link them.. otherwise we skip
            # Draw the line and get the binary mask of the line
            line_mask = draw_line(dilated_image, point1, point2)

            # Dilate the line in the binary mask
            dilated_image = dilate_line(dilated_image, line_mask, dilation_size=2)

            # Check again the number of components, and if they are still more than 1 then stay in the loop and connect the next ones..
            labeled_image, num_features = label(dilated_image)
        else:
            break
    return dilated_image



def final_check(multiclass_image):
    """
    This function is used as a final check, to remove disconnected components that were not previously removed (>volume_threshold), or relabel them as the structure that is connected to them (to ensure continuity..hopefully!).
    Parameters:
    - multiclass_image: 3D numpy array representing the multiclass prediction.
    Returns:
    - new_mask: 3D numpy array representing the corrected multiclass prediction.
    """

    new_mask = multiclass_image.copy()

    for i in np.unique(multiclass_image): # Iterate over each class to check for the disconnected components
        binary_image = 1*(new_mask==i)  # Dealing with 1 class at a time
        labeled_image, num_features = label(binary_image)

        while num_features > 1:  # In case even after trying to link disconnected components, there are stil some remaining, then we either remove them (if they are not connected to anything) or relabel them (if they have neighboring component, they take that label)
            # Identify the smallest component and its volume
            component_volumes = [np.sum(labeled_image == i) for i in range(1, num_features + 1)]
            smallest_component_idx = np.argmin(component_volumes) + 1  # Adjust for label index (1-based)
            # Get coordinates of the smallest component
            smallest_component_coords = np.argwhere(labeled_image == smallest_component_idx)

            # Check connectivity to other components using the multiclass_image mask
            if smallest_component_coords.size > 0:
                # Get neighboring voxel values using a dilated mask
                struct = generate_binary_structure(3, 1)
                dilated_mask = binary_dilation(labeled_image == smallest_component_idx, structure=struct)
                
                neighboring_indices = np.argwhere(dilated_mask & (labeled_image != smallest_component_idx))
                neighboring_labels = new_mask[neighboring_indices[:, 0], neighboring_indices[:, 1], neighboring_indices[:, 2]]  # Get the neighboring labels (intersection between multiclass mask and the dilated one..)
                
                unique_labels = np.unique(neighboring_labels[neighboring_labels > 0])
                if len(unique_labels) > 0:  # A neighboring component exists..
                    # Assign the label of the first neighbor component to the current one
                    new_label = unique_labels[0]
                    new_mask[labeled_image == smallest_component_idx] = new_label
                else:  # No neighboring components..
                    # Delete that component
                    new_mask[labeled_image == smallest_component_idx] = 0
                
                labeled_image, num_features = label(1*(new_mask==i))  # Recompute the connected components to check if there are still some disconnexions (in which case, stay in the loop..).

    return new_mask

