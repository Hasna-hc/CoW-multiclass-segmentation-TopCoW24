
import numpy as np
from skimage.segmentation import expand_labels

from scipy.ndimage import label

from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev
from skimage.morphology import ball
from scipy.ndimage import distance_transform_edt

from scipy.ndimage import binary_dilation

## -------------------------- Step 1 --------------------------
def get_bounding_box(binary, margin=5):
    ''' Get the bounding box to focus only on the non-zero regions. 
        margin is set to 5 to get enough space to expand labels with a distance_th of 3 voxels.. 
    '''
    coords = np.argwhere(binary)
    min_coords = np.maximum(coords.min(axis=0) - margin, 0)
    max_coords = np.minimum(coords.max(axis=0) + margin + 1, binary.shape)
    slices = tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords))
    return slices


def fast_fill_with_expand_cropped(seg, binary, distance_th=3):
    ''' This function crops both the binary and multiclass masks within the bbox to focus on the non-zero regions.
        Then, it expands the labels of the multiclass mask by 3 voxels, and add those labels within the targetted regions (where seg==0 and bin==1).
        The result is added back to the multiclass mask within the bbox region.    
    '''
    seg_filled = seg.copy()

    # Only process where binary mask is nonzero
    bbox = get_bounding_box(binary)

    cropped_seg = seg[bbox]
    cropped_binary = binary[bbox]

    background_mask = (cropped_seg == 0) & (cropped_binary == 1)
    expanded_cropped = expand_labels(cropped_seg, distance_th)

    cropped_result = cropped_seg.copy()
    cropped_result[background_mask] = expanded_cropped[background_mask]

    seg_filled[bbox] = cropped_result
    return seg_filled


def fill_missing_voxels(mul_mask, bin_mask, distance_th=3):
    ''' This function loops by filling the multiclass mask within the targetted regions by a distance of 3 voxels until there is no new change.
        The last step (optional) adds the remaining parts from the binary mask that are beyond 3 voxels away.    
    '''
    new_seg = mul_mask.copy()
    status = True

    while status:
        prev = new_seg.copy()
        new_seg = fast_fill_with_expand_cropped(new_seg, bin_mask, distance_th)
        status = not np.array_equal(new_seg, prev)

    final = new_seg
    return final



## -------------------------- Step 2 --------------------------
def clean_small_components(segmentation, volume_threshold, structure):
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
        labeled_array, num_features = label(segmentation==i, structure=structure)

        # Iterate over the labeled components
        for component_label in range(1, num_features + 1):  # Skip background (label 0)
            component_mask = (labeled_array == component_label)
            
            # Calculate the volume of the current component
            volume = np.sum(component_mask)
            # If the component volume is smaller than the threshold, remove it by setting it to 0
            if volume < volume_threshold:
                print(f"Label {i}. Deleting component of volume: {volume}.")
                segmentation[component_mask] = 0

    return segmentation



## -------------------------- Step 3 --------------------------
def get_closest_points(component_coords):
    ''' Find closest pairs of components '''
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

    return comp1, comp2, distance, point1, point2


def find_closest_points(coords1, coords2):
    """Find the closest pair of points between two point sets."""
    dists = cdist(coords1, coords2)
    idx = np.unravel_index(np.argmin(dists), dists.shape)
    return coords1[idx[0]], coords2[idx[1]]


def get_local_points(skeleton, center, radius=5):
    ''' Find the points of the skeleton within a neighborhood range of 5 voxels '''
    coords = np.argwhere(skeleton)
    deltas = coords - center
    dists = np.linalg.norm(deltas, axis=1)
    return coords[dists <= radius]


def draw_line(point1, point2):
    """
    Draws a straight line between two points in 3D.
    Parameters:
    - image: 3D numpy array representing the binary mask of a certain label.
    - point1:
    - point2:
    Returns:
    - line_mask: 3D numpy array representing the line connecting the two points (same shape as the input image).
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    # Number of points along the line (based on the largest difference between coordinates)
    num_points = np.max(np.abs(point2 - point1)) + 1
    
    # Interpolate the coordinates between point1 and point2
    line_coords = np.stack([np.linspace(p1, p2, num_points) for p1, p2 in zip(point1, point2)], axis=-1).round().astype(int)    
    
    return line_coords


def remove_duplicates_ordered(points):
    ''' This function removes duplicated points '''
    seen = set()
    unique_pts = []
    for p in points:
        if p not in seen:
            seen.add(p)
            unique_pts.append(p)
    return unique_pts


def fit_smooth_connection_all(p1, p2, pts1, pts2, point1, point2, n_points=50):
    ''' This function fits a 3D spline on the provided points (from the skeletons: pts1 and pts2 + the two closest endpoints of the components: point1 and point2), sorted by the vector p2-p1. 
        All duplicated points are removed.
    '''

    # Removing points to avoid repetition (otherwise splprep wouldn't work..)
    control_pts = np.vstack([pts1[~np.all(pts1 == point1, axis=1)], pts2[~np.all(pts2 == point2, axis=1)], point1, point2])

    # Optionally sort control points along the vector p1 → p2
    vec = p2 - p1
    control_pts = control_pts[np.argsort((control_pts - p1) @ vec)]

    # Fit 3D spline
    tck, _ = splprep(control_pts.T, s=1.0)
    u = np.linspace(0, 1, n_points)
    spline_pts = np.vstack(splev(u, tck)).T
    spline_pts = np.round(spline_pts).astype(np.int32)
    
    return remove_duplicates_ordered([tuple(p) for p in spline_pts])


def fit_curve_all(skel1, skel2, pnt1, pnt2):
    ''' This function finds the skeleton points of each of the pair component, including the two closest points p1 and p2,
        and the 5-voxels neighbors from those two points within the skeleton points. If the number of these points are > 4, we fit a 3D spline, otherwise we use a straight line.
        This fitted line include all the aforementioned points + the two closest points (pnt1 and pnt2) within the components themselves (not the skeleton).
    '''
    skel1_coords = np.argwhere(skel1 > 0)
    skel2_coords = np.argwhere(skel2 > 0)
    
    p1, p2 = find_closest_points(skel1_coords, skel2_coords)
    pts1 = get_local_points(skel1, p1)  # Get neighboring points around p1
    pts2 = get_local_points(skel2, p2)  # Get neighboring points around p2
    if len(pts1) + len(pts2) <4:
        print(f'Not enough points to fit a curve (len(pts1): {len(pts1)}, len(pts2): {len(pts2)}). Drawing a line instead.')
        curve_pts = draw_line(p1, p2)
    else:
        curve_pts = fit_smooth_connection_all(p1, p2, pts1, pts2, pnt1, pnt2)  # Fit a smooth curve between p1 and p2
    return curve_pts, p1, p2


def add_curve_to_mask(mask, curve_pts):
    """ Draw curve points into a 3D mask. """
    for pt in curve_pts:
        pt = np.array(pt)
        if np.all((0 <= pt) & (pt < mask.shape)):  # Making sure it doesn't go beyong the image shape (out of bound)
            mask[tuple(pt)] = 1
    return mask


def dilate_curve(new_skeleton, bin_img, p1, p2, larger=False):
    """ Dilate the curve by inserting in each point of the 
        curve a ball of radius adapted to the distance transform between the two end points. 
        In case where the results still doesn't form one component, the radius is enlarged with 1 voxel, wihch corresponds to 'larger=True' (probably doesn't happen..).
    """
    num_points = new_skeleton.sum().astype(int)
    dist_transform = distance_transform_edt(bin_img.copy())
    radii = np.linspace(dist_transform[tuple(p1)], dist_transform[tuple(p2)], num_points)
    curve_coords = np.argwhere(new_skeleton > 0)

    new_mask = np.zeros(new_skeleton.shape, dtype=np.uint8)
    for coord, r in zip(curve_coords, radii):  # radii is a list of per-point radii
        struct_elem = ball(r) if not larger else ball(r+1)
        se_shape = np.array(struct_elem.shape)
        offset = se_shape // 2

        # Compute bounding box in new_mask
        min_pt = np.array(coord) - offset
        max_pt = min_pt + se_shape

        # Skip if out of bounds
        if np.any(min_pt < 0) or np.any(max_pt > new_mask.shape):
            continue  # or handle edge cropping if needed

        # Insert structuring element
        slices_mask = tuple(slice(p1, p2) for p1, p2 in zip(min_pt, max_pt))
        new_mask[slices_mask] |= struct_elem
    return new_mask



## -------------------------- Step 4 --------------------------
def find_distance(comp1, comp2):
    """Find the closest pair of points between two point sets."""
    coords1 = np.argwhere(comp1)
    coords2 = np.argwhere(comp2)
    dists = cdist(coords1, coords2)
    return np.min(dists)


def final_check(multiclass_image, structure_n26, min_dist=15):
    """
    This function is used as a final check, to remove disconnected components that were not previously removed (>volume_threshold), or relabel them as the structure that is connected to them (to ensure continuity..hopefully!).
    Parameters:
    - multiclass_image: 3D numpy array representing the multiclass prediction.
    Returns:
    - new_mask: 3D numpy array representing the corrected multiclass prediction.
    """

    new_mask = multiclass_image.copy()
    skipped_coords = np.zeros_like(new_mask, dtype=bool)

    # for lab in np.unique(multiclass_image): # Iterate over each class to check for the disconnected components
    for lab in np.unique(multiclass_image[multiclass_image != 0]): # Iterate over each class to check for the disconnected components         
        binary_image = 1*(new_mask==lab)  # Dealing with 1 class at a time
        labeled_image, num_features = label(binary_image, structure_n26)
        # comp_track = []
        # skipped_components = set()

        while num_features > 1:  # In case even after trying to link disconnected components, there are stil some remaining, then we either remove them (if they are not connected to anything) or relabel them (if they have neighboring component, they take that label)
            # Identify the smallest component and its volume
            component_volumes = [np.sum(labeled_image == i) for i in range(1, num_features + 1)]
            smallest_component_idx = np.argmin(component_volumes) + 1
            largest_component_idx = np.argmax(component_volumes) + 1

            current_component_mask = (labeled_image == smallest_component_idx)
            if np.any(skipped_coords & current_component_mask):
                print(f"Label {lab}, component_volumes: {component_volumes}. This component was previously skipped, skipping again.")
                # Remove it from the current relabeling round
                labeled_image[current_component_mask] = 0
                num_features -= 1
                continue
            
            dilated_mask = binary_dilation(current_component_mask, structure=structure_n26)
            neighboring_indices = np.argwhere(dilated_mask & (labeled_image != smallest_component_idx))
            neighboring_labels = new_mask[neighboring_indices[:, 0], neighboring_indices[:, 1], neighboring_indices[:, 2]]
            unique_labels = np.unique(neighboring_labels[neighboring_labels > 0])

            if len(unique_labels) > 0:
                closest_dist = find_distance(current_component_mask, labeled_image == largest_component_idx)
                print(f"Label {lab}, component_volumes: {component_volumes}. Closest distance to largest component: {closest_dist}")

                if closest_dist < min_dist:
                    print(f"Label {lab}, component_volumes: {component_volumes}. Component is close enough (<= {min_dist}), skipping.")
                    skipped_coords |= current_component_mask  # Track voxels directly
                else:
                    new_label = unique_labels[0]
                    print(f"Label {lab}, component_volumes: {component_volumes}. Relabeling to label {new_label}")
                    new_mask[current_component_mask] = new_label
            else:
                print(f"Label {lab}, component_volumes: {component_volumes}. No neighbors found. Deleting the component..")
                new_mask[current_component_mask] = 0

            # Re-label the remaining voxels of this class
            labeled_image, num_features = label((new_mask == lab) & (~skipped_coords), structure=structure_n26)

    return new_mask


