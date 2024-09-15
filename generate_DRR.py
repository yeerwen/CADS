import numpy as np
from skimage.transform import radon, iradon
import SimpleITK as sitk
import os
import shutil
import tqdm
import sys
import cv2 as cv
import itk
import math
import nibabel as nib
def NiiDataRead(path, as_type=np.float32):
    img = sitk.ReadImage(path)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()
    img_it = sitk.GetArrayFromImage(img).astype(as_type)
    return img_it, spacing, origin, direction

def NiiDataWrite(path, prediction_final, spacing, origin, direction):
    # prediction_final = prediction_final.astype(as_type)
    img = sitk.GetImageFromArray(prediction_final)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(direction)
    sitk.WriteImage(img, path)

def DigitallyReconstructedRadiograph(
        input_path,
        output_path,
        ray_source_distance=20000,
        camera_tx= 0.,
        camera_ty= 0.,
        camera_tz= 0.,
        rotation_x=90.,
        rotation_y=0.,
        rotation_z= 0.,
        projection_normal_p_x=0.,
        projection_normal_p_y=0.,
        rotation_center_rt_volume_center_x=0.,
        rotation_center_rt_volume_center_y=0.,
        rotation_center_rt_volume_center_z=0.,
        threshold=-1024.,
):
    """
    Parameters description:

    ray_source_distance = 400                              # <-sid float>            Distance of ray source (focal point) focal point 400mm
    camera_translation_parameter = [0., 0., 0.]            # <-t float float float>  Translation parameter of the camera

    rotation_around_xyz = [0., 0., 0.]                     # <-rx float>             Rotation around x,y,z axis in degrees
    projection_normal_position = [0, 0]                    # <-normal float float>   The 2D projection normal position [default: 0x0mm]
    rotation_center_relative_to_volume_center = [0, 0, 0]  # <-cor float float float> The centre of rotation relative to centre of volume
    threshold = 10                                          # <-threshold float>      Threshold [default: 0]
    """
    input_name = input_path
    volume_lung = itk.imread(input_name, itk.ctype('float'))

    output_image_pixel_spacing = [1., 1., 1.]
    output_image_size = list(volume_lung.GetBufferedRegion().GetSize())
    output_image_size = [output_image_size[0], output_image_size[2], output_image_size[1]]
    output_image_size[-1] = 1

    InputImageType = type(volume_lung)
    FilterType = itk.ResampleImageFilter[InputImageType, InputImageType]
    filter = FilterType.New()
    filter.SetInput(volume_lung)
    filter.SetDefaultPixelValue(0)
    filter.SetSize(output_image_size)
    filter.SetOutputSpacing(output_image_pixel_spacing)

    TransformType = itk.CenteredEuler3DTransform[itk.D]
    transform = TransformType.New()
    transform.SetComputeZYX(True)

    InterpolatorType = itk.RayCastInterpolateImageFunction[InputImageType, itk.D]
    interpolator = InterpolatorType.New()

    dgree_to_radius_coef = 1. / 180. * math.pi
    camera_translation_parameter = [camera_tx, camera_ty, camera_tz]
    rotation_around_xyz = [rotation_x * dgree_to_radius_coef, rotation_y * dgree_to_radius_coef,
                           rotation_z * dgree_to_radius_coef]
    projection_normal_position = [projection_normal_p_x, projection_normal_p_y]
    rotation_center_relative_to_volume_center = [
        rotation_center_rt_volume_center_x,
        rotation_center_rt_volume_center_y,
        rotation_center_rt_volume_center_z
    ]

    imageOrigin = volume_lung.GetOrigin()
    imageSpacing = volume_lung.GetSpacing()
    imageRegion = volume_lung.GetBufferedRegion()
    imageSize = imageRegion.GetSize()
    imageCenter = [imageOrigin[i] + imageSpacing[i] * imageSize[i] / 2.0 for i in range(3)]

    transform.SetTranslation(camera_translation_parameter)
    transform.SetRotation(rotation_around_xyz[0], rotation_around_xyz[1], rotation_around_xyz[2])

    center = [c + imageCenter[i] for i, c in enumerate(rotation_center_relative_to_volume_center)]
    transform.SetCenter(center)

    interpolator.SetTransform(transform)
    interpolator.SetThreshold(threshold)
    focalPoint = [imageCenter[0], imageCenter[1], imageCenter[2] - ray_source_distance / 2.0]
    interpolator.SetFocalPoint(focalPoint)

    filter.SetInterpolator(interpolator)
    filter.SetTransform(transform)

    origin = [
        imageCenter[0] + projection_normal_position[0] - output_image_pixel_spacing[0] * (
                    output_image_size[0] - 1) / 2.,
        imageCenter[1] + projection_normal_position[1] - output_image_pixel_spacing[1] * (
                    output_image_size[1] - 1) / 2.,
        imageCenter[2] + imageSpacing[2] * imageSize[2]
    ]

    filter.SetOutputOrigin(origin)
    filter.Update()

    image = np.asarray(filter.GetOutput())[0]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    np.save(output_path, (image - 0.5)*2)

def pre_process_image_DeepLesion(path, out_path):
    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    os.makedirs(out_path, exist_ok=True)

    kits19_training_dir = os.listdir(path)
    for file_name in tqdm.tqdm(kits19_training_dir):

        # print(file_name)
        mid_1_path = os.path.join(path, file_name)
        if os.path.exists(os.path.join(out_path, file_name.replace("nii.gz", "npy"))):
            continue

        DigitallyReconstructedRadiograph(
            input_path=mid_1_path,
            output_path=os.path.join(out_path, file_name.replace("nii.gz", "npy")),
            rotation_x=90.,
            rotation_y=0.,
            rotation_z=90.)



if __name__ == '__main__':
    pre_process_image_DeepLesion(path="/data/userdisk0/ywye/Dataset/DeepLesion/Images_nifti_spacing/", out_path="/data/userdisk0/ywye/Dataset/DeepLesion/DRR/")


