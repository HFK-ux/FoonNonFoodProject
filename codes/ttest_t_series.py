import nibabel as nib
import numpy as np
from scipy.stats import ttest_1samp

#Set the paths
main_directory = "Users/harunfurkankanik/Desktop/FoodNoFood-main"
BOLD_folder = f"{main_directory}/volume"
output_folder = f"{main_directory}/TimeSeries_food_NonFood"

#Specify the subjects
subject_id = "sub-01"

#Load BOLD data
nii_data = nib.load (f"{BOLD_folder}/sub-01.nii")
data_4d = nii_data.get_fdata()
data_3d = np.mean(data_4d, axis=3, keepdims=True)

#subtract 4d to 3d
final_image = data_4d - data_3d
xdim, ydim, zdim, _ = final_image.shape

time_points_food = [
    [0, 48, 90, 137, 187, 234, 279, 321],  # First volume across blocks
    [1, 49, 91, 138, 188, 235, 280, 322],  # Second volume across blocks
    [2, 50, 92, 139, 189, 236, 281, 323],  # Third volume, and so on...
    [3, 51, 93, 140, 190, 237, 282, 324],
    [4, 52, 94, 141, 191, 238, 283, 325],
    [5, 53, 95, 142, 192, 239, 284, 326],
    [6, 54, 96, 143, 193, 240, 285, 327],
    [7, 55, 97, 144, 194, 241, 286, 328],
    [8, 56, 98, 145, 195, 242, 287, 329],
    [9, 57, 99, 146, 196, 243, 288, 330],
    [10, 58, 100, 147, 197, 244, 289, 331],
    [11, 59, 101, 148, 198, 245, 290, 332],
    [12, 60, 102, 149, 199, 246, 291, 333],
    [13, 61, 103, 150, 200, 247, 292, 334],
    [14, 62, 104, 151, 201, 248, 293, 335],
    [15, 63, 105, 152, 202, 249, 294, 336],
]

time_points_NonFood = [
    [25, 70, 112, 157, 209, 257, 299, 346],  
    [26, 71, 113, 158, 210, 258, 300, 347],  
    [27, 72, 114, 159, 211, 259, 301, 348],  
    [28, 73, 115, 160, 212, 260, 302, 349],  
    [29, 74, 116, 161, 213, 261, 303, 350],  
    [30, 75, 117, 162, 214, 262, 304, 351],  
    [31, 76, 118, 163, 215, 263, 305, 352],  
    [32, 77, 119, 164, 216, 264, 306, 353],  
    [33, 78, 120, 165, 217, 265, 307, 354],  
    [34, 79, 121, 166, 218, 266, 308, 355],  
    [35, 80, 122, 167, 219, 267, 309, 356],  
    [36, 81, 123, 168, 220, 268, 310, 357],  
    [37, 82, 124, 169, 221, 269, 311, 358],  
    [38, 83, 125, 170, 222, 270, 312, 359],  
    [39, 84, 126, 171, 223, 271, 313, 360],  
    [40, 85, 127, 172, 224, 272, 314, 361],  
]

# Function to compute t-values and save the time series
def compute_t_series(time_points, condition_name):
    t_series = []  # Store 3D t-value maps for all volumes

    for volume_idx, block_time_points in enumerate(time_points):
        # Extract time series for the current volume
        volume_data = final_image[:, :, :, block_time_points]

        # Initialize t-values array
        t_values = np.zeros((xdim, ydim, zdim))

        # Compute t-test for each voxel
        for i in range(xdim):
            for j in range(ydim):
                for k in range(zdim):
                    voxel_data = volume_data[i, j, k, :]
                    if np.any(voxel_data):  # Ensure the voxel has valid data
                        t, _ = ttest_1samp(voxel_data, 0, nan_policy="omit")
                        t_values[i, j, k] = t

        t_series.append(t_values)  # Add 3D t-map to the time series

    # Stack all 3D t-value maps into a 4D image
    t_series_4d = np.stack(t_series, axis=3)

    # Save the 4D time series to a NIfTI file
    output_filename = f"{output_folder}/{subject_id}_t_series_{condition_name}.nii.gz"
    output_image = nib.Nifti1Image(t_series_4d, affine=nii_data.affine)
    output_image.set_data_dtype(np.float32)
    nib.save(output_image, output_filename)
    print(f"Saved {condition_name} time series to: {output_filename}")

# Compute and save time series for Food and NonFood conditions
compute_t_series(time_points_food, "food")
compute_t_series(time_points_NonFood, "nonfood")
