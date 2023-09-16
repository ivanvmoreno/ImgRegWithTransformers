import cv2
import numpy as np
import SimpleITK as sitk
import torch


def check_image_values(image):
    print("Max value: ", np.max(image))
    print("Min value: ", np.min(image))
    print("Mean value: ", np.mean(image))
    print("Contains NaNs: ", np.isnan(image).any())
    print("Contains Infs: ", np.isinf(image).any())


def register(source, target, source_mask, target_mask, device="cuda"):
    source = source.cpu().numpy()
    target = target.cpu().numpy()
    source_mask = source_mask.cpu().numpy()
    target_mask = target_mask.cpu().numpy()

    # Inverting intensity values
    target = 255 - target
    source = 255 - source

    # Background Removal
    target_mask = np.array(target_mask != 0, dtype=np.uint8)
    source_mask = np.array(source_mask != 0, dtype=np.uint8)
    target = cv2.bitwise_and(target, target, mask=target_mask)
    source = cv2.bitwise_and(source, source, mask=source_mask)

    # check_image_values(source)
    # check_image_values(target)

    # Getting SimpleITK Images from numpy arrays
    source_image_inv_sitk = sitk.GetImageFromArray(source)
    target_image_inv_sitk = sitk.GetImageFromArray(target)

    # Explicitly set the spacing to a non-zero value
    source_image_inv_sitk.SetSpacing([1.0, 1.0])
    target_image_inv_sitk.SetSpacing([1.0, 1.0])

    # Ensure the images have the same size and spacing
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(source_image_inv_sitk.GetSize())
    resampler.SetOutputSpacing(source_image_inv_sitk.GetSpacing())
    target_image_inv_sitk = resampler.Execute(target_image_inv_sitk)

    source_image_inv_sitk = sitk.Cast(source_image_inv_sitk, sitk.sitkFloat32)
    target_image_inv_sitk = sitk.Cast(target_image_inv_sitk, sitk.sitkFloat32)

    if (
        min(source_image_inv_sitk.GetSize()) < 2
        or min(target_image_inv_sitk.GetSize()) < 2
    ):
        raise ValueError("Images have degenerate dimensions.")

    # Define the transform
    transformDomainMeshSize = [4] * source_image_inv_sitk.GetDimension()
    tx = sitk.BSplineTransformInitializer(
        source_image_inv_sitk, transformDomainMeshSize
    )

    R = sitk.ImageRegistrationMethod()
    R.SetInitialTransformAsBSpline(tx, inPlace=True, scaleFactors=[1, 2, 5])
    R.SetMetricAsMattesMutualInformation(50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.2)

    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([4, 2, 1])
    R.SetOptimizerAsGradientDescentLineSearch(
        0.5, 100, convergenceMinimumValue=1e-4, convergenceWindowSize=5
    )
    R.SetInterpolator(sitk.sitkLinear)

    outTx = R.Execute(source_image_inv_sitk, target_image_inv_sitk)
    displacement_field_image = sitk.TransformToDisplacementField(
        outTx,
        sitk.sitkVectorFloat64,
        source_image_inv_sitk.GetSize(),
        source_image_inv_sitk.GetOrigin(),
        source_image_inv_sitk.GetSpacing(),
        source_image_inv_sitk.GetDirection(),
    )
    # Convert the displacement field image to a numpy array
    displacement_field_array = sitk.GetArrayFromImage(displacement_field_image)

    # Convert the numpy array to a PyTorch tensor and move to GPU
    return torch.from_numpy(displacement_field_array).to(device)
