import os
import pydicom
import numpy as np
import pandas as pd

dct = {
    'SliceThickness': (0x0018, 0x0050),
    'Rows': (0x0028, 0x0010),
    'Columns': (0x0028, 0x0011),
    'Magnetic Field Strength': (0x0018, 0x0087),
    'Flip Angle': (0x0018, 0x1314),
    'PixelSpacing': (0x0028, 0x0030),
    'SpacingBetweenSlices': (0x0018, 0x0088),
    'ImagePositionPatient': (0x0020, 0x0032)
 }


def get_tag_value(dcm, tagno, tagname):
    """
    Returns the dicom tag value based on e a string name (tagname)
    or number (tagno: (0x0001,0x0001))
    """
    try:
        out = dcm[tagno].value
    except:
        try:
            out = dcm[tagname].value
        except:
            out = np.NaN
    return out


def all_tag_values(dcm, mdct, stringconvert=False):
    out = []
    for k, v in mdct.items():
        tag = get_tag_value(dcm, k, v)
        if stringconvert:
            tag = str(tag)
        out.append(tag)
    return out


if __name__ == "__main__":
    group = "Master Students SS 23"
    dicom_details = {}
    root_directory = 'L:\\basic\\divi\\jstoker\\slicer_pdac\\' + group + '_CT_data'
    if group == 'Ewout': root_directory = 'L:\\basic\\divi\\jstoker\\slicer_pdac\\Ewout\\DICOM files model design cohort'
    if group == 'PREOPANC2': root_directory += '\\AI-PANC-CT_data\\'
    if group == 'Master Students SS 23': root_directory = 'L:\\basic\\divi\\jstoker\\slicer_pdac\\Master Students SS 23\\Danial\\scans without segmentations'
    for dcm_dir in os.listdir(root_directory):
        phase_dir = os.path.join(root_directory, dcm_dir, os.listdir(root_directory + '\\' + dcm_dir)[0])
        print(phase_dir)
        if 'LAPC' in group or 'PREOPANC' in group:
            phase_dir = os.path.join(phase_dir, os.listdir(phase_dir)[0])
            phase_dir = os.path.join(phase_dir, os.listdir(phase_dir)[len(os.listdir(phase_dir))-1])
        elif 'Ewout' in group or 'Master Students SS 23' in group:
            phase_dir = os.path.join(phase_dir, os.listdir(phase_dir)[1])
        file_dir = os.path.join(phase_dir, os.listdir(phase_dir)[0])
        print(file_dir)
        dcm = pydicom.dcmread(file_dir, stop_before_pixels=True, force=True)
        print(dcm)
        if group == 'Ewout':
            dicom_details[file_dir.split('cohort\\')[len(file_dir.split('cohort\\')) - 1].split('\\')[0]] = all_tag_values(dcm, dct)
        elif group == 'PREOPANC2':
            dicom_details[file_dir.split('CT_data\\')[len(file_dir.split('CT_data\\')) - 1].split('\\')[0]] = all_tag_values(dcm,dct)
        else:
            dicom_details[file_dir.split('segmentations\\')[len(file_dir.split('segmentations\\')) - 1].split('\\')[0]] = all_tag_values(dcm,dct)

    df = pd.DataFrame.from_dict(dicom_details, orient='index').transpose()
    df.insert(0, 'Parameter', list(dct.keys()))
    print(df.max(axis = 1))
    #df.insert(1, 'Maximum', df.max(axis = 1))
    #df.insert(1, 'Minimum', df.min(axis = 1))
    # df.to_excel('L:\\basic\\divi\\jstoker\\slicer_pdac\\Master Students SS 23\\Matej\\MRI_modalities.xlsx', index=False)
