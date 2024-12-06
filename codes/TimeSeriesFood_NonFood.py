import nibabel as nib
import numpy as np

#Set the paths
main_directory = "Users/harunfurkankanik/Desktop/FoodNoFood-main"
BOLD_folder = f"{main_directory}/volume"
output_folder = f"{main_directory}/TimeSeries"

#Specify the subjects
subject_id = "sub-01"

#Set experiment paremeter
num_blocks = 8
rest_block_duration = 16
stimils_block_duration = 16
Total_block_duration = 32
TR = 1.6
start_index = 1
end_index = 32
duration = end_index - start_index + 1

#Load BOLD data
nii_data = nib.load (f"{BOLD_folder}/sub-01.nii")
data_4d = nii_data.get_fdata()
data_3d = np.mean(data_4d, axis=3, keepdims=True)

#subtract 4d to 3d
final_image = data_4d - data_3d
xdim, ydim, zdim, _ = final_image.shape

time_points_food = [
    list(range(0, 16)),
    list(range(48, 64)),
    list(range(90, 106)),
    list(range(137, 153)),
    list(range(187, 203)),
    list(range(234, 250)),
    list(range(279, 295)),
    list(range(321, 337))
]
time_points_NonFood =[
    list(range(25, 41)),
    list(range(70, 86)),
    list(range(112, 128)),
    list(range(157, 173)),
    list(range(209, 225)),
    list(range(257, 273)),
    list(range(299, 315)),
    list(range(346,362))
]

for block_idx, block_time_points in enumerate (time_points_food):
    #extract time series for food
    time_series_food = final_image[:, :, :, block_time_points]

    # Save the time series
    output_filename = output_folder + '/' + f'{subject_id}_time_series_food_{block_idx + 1}.nii'
    output_image = nib.Nifti1Image(time_series_food, affine = nii_data.affine)
    output_image.set_data_dtype(np.float32)
    nib.save(output_image, output_filename)

for block_idx_NonFood, block_time_points_NonFood in enumerate (time_points_NonFood):
    #extract time series for NonFood
    time_series_NonFood = final_image[:, :, :, block_time_points_NonFood]

    #Save the time series for NonFood
    output_filename_1 = output_folder + '/' + f'{subject_id}_time_series_NonFood_{block_idx_NonFood + 1}.nii'
    output_image_1 = nib.Nifti1Image(time_series_NonFood, affine = nii_data.affine)
    output_image_1.set_data_dtype(np.float32)
    nib.save(output_image_1, output_filename_1)
