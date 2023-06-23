# This code assumes that there is only patient scan in the T1 folder.

import os
import re
import shutil

root_dir = r"L:\basic\divi\jstoker\slicer_pdac\Master Students SS 23\Matej\Ready_TO_USE_IPMN"
out_dir = r"L:\basic\divi\jstoker\slicer_pdac\Master Students SS 23\Matej\keras_data_new" # Make output folder

patients = []


def make_directories(root_dir, out_dir):
    """
    Check whether patient has an MRI/T1 folder and create a subdirectory for every patient from root_dir in out_dir.

    Parameters:
        :param: root_dir (str): Root directory containing patient directories.
        :param: out_dir (str): Output directory where patient subdirectories will be created.

    :return: None

    """
    pattern = r"Panc_\d{4}"
    for subdir, subdirs, files in os.walk(root_dir):
        if "MRI" in subdirs and os.path.isdir(os.path.join(subdir, "MRI", "T1")):
            match = re.search(pattern, subdir)
            if match:
                patient = match.group().replace("_", "-")
                patient_dir = os.path.join(out_dir, patient)
                if not os.path.exists(patient_dir):
                    os.makedirs(patient_dir)


def move_files(root_dir, out_dir):
    """
    Move files from the root directory into the output directory, renaming them as "im.nii.gz" and "seg.nrrd".

    Parameters:
        :param: root_dir (str): Root directory containing the files to be moved.
        :param: out_dir (str): Output directory where the files will be moved.

    :return: None

    """
    pattern = r"Panc_\d{4}"

    for subdir, dirs, files in os.walk(root_dir):
        if "MRI" in dirs and os.path.isdir(os.path.join(subdir, "MRI", "T1")):
            t1_dir = os.path.join(subdir, "MRI", "T1")
            match = re.search(pattern, subdir)
            if match:
                patient = match.group().replace("_", "-")
                patient_dir = os.path.join(out_dir, patient)

            for file in os.listdir(t1_dir):
                if file.endswith(".seg.nrrd"):
                    seg = os.path.join(t1_dir, file)
                    shutil.move(seg, patient_dir)
                elif file.endswith(".nii.gz") or file.endswith(".nii") or file.endswith(".nrrd") or file.endswith(".nhdr"):
                    img = os.path.join(t1_dir, file)
                    shutil.move(img, patient_dir)


def rename_files(sorted_dir):
    """
    Rename files in the sorted output directory to im.nrrd and seg.nrrd.

    :param: sorted_dir (str): Directory containing the sorted files.
    :return: None

    """
    for subdir in os.listdir(sorted_dir):
        for file in os.listdir(os.path.join(sorted_dir, subdir)):
            file_dir = os.path.join(sorted_dir, subdir, file)

            if file_dir.endswith(".seg.nrrd"):
                os.rename(file_dir, os.path.join(sorted_dir, subdir, "seg.nrrd"))

            elif file_dir.endswith(".nii"):
                os.rename(file_dir, os.path.join(sorted_dir, subdir, "im.nii"))

            elif file_dir.endswith(".nii.gz"):
                os.rename(file_dir, os.path.join(sorted_dir, subdir, "im.nii.gz"))

            elif file_dir.endswith(".nrrd"):
                os.rename(file_dir, os.path.join(sorted_dir, subdir, "im.nrrd"))

            elif file_dir.endswith(".nhdr"):
                os.rename(file_dir, os.path.join(sorted_dir, subdir, "im.nhdr"))


def file_check(out_dir):
    """
    Check if each subdirectory in out_dir contains exactly two files starting with 'im.' and 'seg.' prefixes.

    :param: out_dir (str): Directory containing the subdirectories to be checked.
    :return: None
    :raise: ValueError: If any subdirectory does not contain exactly two files starting with 'im.' and 'seg.' prefixes.

    """
    for subdir in os.listdir(out_dir):
        sub_path = os.path.join(out_dir, subdir)
        if os.path.isdir(sub_path):
            sub_files = os.listdir(sub_path)
            if len(sub_files) == 2 and any(f.startswith("im.") for f in sub_files) and any(f.startswith("seg.") for f in sub_files):
                pass
            else:
                raise ValueError(f"Error: {subdir} does not contain exactly two files starting with 'im.' and 'seg.'")


def main():
    make_directories(root_dir, out_dir)
    move_files(root_dir, out_dir)
    rename_files(out_dir)
    file_check(out_dir)


if __name__=='__main__':
    main()

