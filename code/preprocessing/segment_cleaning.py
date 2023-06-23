import glob
import os
import slicerio
import shutil
import SimpleITK as sitk

root_dir = r"L:\basic\divi\jstoker\slicer_pdac\Master Students SS 23\Matej\keras_data_new"


def check_all_segmentations(root_dir):
    """
    Read all seg.nrrd files in subdirectories of a directory and return a list of
    file paths for which the pancreas is not labeled with "1".

    """
    problematic_files = []

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".nrrd"):
                seg_file = os.path.join(subdir, file)
                segmentation_info = slicerio.read_segmentation_info(seg_file)

                segment_names = slicerio.segment_names(segmentation_info)
                if "pancreas" in segment_names:
                    pancreas_segment = slicerio.segment_from_name(segmentation_info, "pancreas")
                    if pancreas_segment["labelValue"] != 1:
                        problematic_files.append(seg_file)

    if len(problematic_files) != 0:
        raise ValueError(
            f"The labelValue of the pancreas mask is different from 1 in the following files: {problematic_files}. "
            f"Reorder the segmentations before proceeding")


def write_pancreas_masks(root_dir):
    """
    Write a new nrrd file in each directory which will contain only the masks of the pancreas (labelValue 1).

    """
    # Get a list of all directories containing nrrd files
    patient_dirs = [f.path for f in os.scandir(root_dir) if f.is_dir()]

    # Loop over each directory
    for patient_dir in patient_dirs:

        seg_file = os.path.join(patient_dir, "seg.nrrd")
        new_seg_file = os.path.join(patient_dir, "pancreas_seg.nrrd")

        # Read the segmentation file and get the pancreas segment
        segmentation_info = slicerio.read_segmentation_info(seg_file)
        pancreas_segment = slicerio.segment_from_name(segmentation_info, "pancreas")
        pancreas_label_value = pancreas_segment["labelValue"]

        # Create a new segmentation file containing only the pancreas segment
        image = sitk.ReadImage(seg_file)
        label_values = sitk.GetArrayFromImage(image)
        pancreas_mask = (label_values == pancreas_label_value).astype(int)
        new_image = sitk.GetImageFromArray(pancreas_mask)
        new_image.CopyInformation(image)

        try:

            sitk.WriteImage(new_image, new_seg_file)

        except:

            raise ValueError(f"Error: could not write new segmentation mask for {seg_file}.")


def folder_cleaning(root_dir):
    """
    Remove the original "seg.nrrd" file and rename the newly created "pancreas_seg.nrrd" file to "seg.nrrd"
    for each directory in the given root directory that contains both files.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        if "seg.nrrd" in filenames and "pancreas_seg.nrrd" in filenames:
            os.remove(os.path.join(dirpath, "seg.nrrd"))
            os.rename(os.path.join(dirpath, "pancreas_seg.nrrd"), os.path.join(dirpath, "seg.nrrd"))


def main():
    check_all_segmentations(root_dir)
    write_pancreas_masks(root_dir)
    folder_cleaning(root_dir)

if __name__=='__main__':
    main()
