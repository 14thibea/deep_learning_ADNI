from torch.utils.data import Dataset
import torch
import pandas as pd
from os import path
import nibabel as nib
import numpy as np
from nilearn import plotting


class MriBrainDataset(Dataset):
    """Dataset of subjects of CLINICA (baseline only)"""

    def __init__(self, subjects_df_path, caps_dir, transform=None, classes=2, mni=False):
        """

        :param subjects_df_path: Path to a TSV file with the list of the subjects in the dataset
        :param caps_dir: The BIDS directory where the images are stored
        :param transform: Optional transform to be applied to a sample
        :param classes: Number of classes to consider for classification
            if 2 --> ['CN', 'AD']
            if 3 --> ['CN', 'MCI', 'AD']
        """
        if type(subjects_df_path) is str:
            self.subjects_df = pd.read_csv(subjects_df_path, sep='\t')
        elif type(subjects_df_path) is pd.DataFrame:
            self.subjects_df = subjects_df_path
        else:
            raise ValueError('Please enter a path or a Dataframe as first argument')

        self.caps_dir = caps_dir
        self.transform = transform

        if classes == 2:
            self.diagnosis_code = {'CN': 0, 'AD': 1}
        elif classes == 3:
            self.diagnosis_code = {'CN': 0, 'MCI': 1, 'AD': 2}

        if mni:
            self.extension = '_ses-M00_T1w_segm-graymatter_space-Ixi549Space_modulated-off_probability.nii.gz'
            self.last_folder = 'normalized_space'
        else:
            self.extension = '_ses-M00_T1w_segm-graymatter_dartelinput.nii.gz'
            self.last_folder = 'dartel_input'

    def __len__(self):
        return len(self.subjects_df)

    def __getitem__(self, subj_idx):
        subj_name = self.subjects_df.loc[subj_idx, 'participant_id']
        diagnosis = self.subjects_df.loc[subj_idx, 'diagnosis']
        cohort = self.subjects_df.loc[subj_idx, 'cohort']
        img_name = subj_name + self.extension
        caps_path = path.join(self.caps_dir, 'CAPS_' + cohort, 'subjects')
        img_path = path.join(caps_path, subj_name, 'ses-M00', 't1', 'spm', 'segmentation', self.last_folder, img_name)

        reading_image = nib.load(img_path)
        image = reading_image.get_data()

        # Convert diagnosis to int
        if type(diagnosis) is str:
            diagnosis = self.diagnosis_code[diagnosis]

        sample = {'image': image, 'diagnosis': diagnosis}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def subjects_list(self):
        return self.subjects_df['participant_id'].values.tolist()

    def diagnosis_list(self):
        diagnosis_list = self.subjects_df['diagnosis'].values.tolist()
        diagnosis_code = [self.diagnosis_code[diagnosis] for diagnosis in diagnosis_list]
        return diagnosis_code

    def imsave(self, subj_idx, output_path, cut_coords=None, use_transforms=True):
        """
        Creates a png file with frontal, axial and lateral cuts of the brain.

        :param subj_idx: The index of the subject in the dataset
        :param output_path: The path to the created image
        :param cut_coords: Coordinates to define the cuts (optional)
        :return: None
        """
        subj_name = self.subjects_df.loc[subj_idx, 'participant_id']
        diagnosis = self.subjects_df.loc[subj_idx, 'diagnosis']
        cohort = self.subjects_df.loc[subj_idx, 'cohort']
        img_name = subj_name + self.extension
        caps_path = path.join(self.caps_dir, 'CAPS_' + cohort, 'subjects')
        img_path = path.join(caps_path, subj_name, 'ses-M00', 't1', 'spm', 'segmentation', self.last_folder, img_name)

        reading_image = nib.load(img_path)
        image = reading_image.get_data()

        sample = {'image': image, 'diagnosis': diagnosis}

        if use_transforms:
            sample = self.transform(sample)

        final_image = nib.Nifti1Image(sample['image'], affine=np.eye(4))
        anat = plotting.plot_anat(final_image, title='subject ' + subj_name, cut_coords=cut_coords)
        anat.savefig(output_path)
        anat.close()


class ToTensor(object):
    """Convert image type to Tensor and diagnosis to diagnosis code"""

    def __init__(self, gpu=False):
        self.gpu = gpu

    def __call__(self, sample):
        image, diagnosis = sample['image'], sample['diagnosis']
        np.nan_to_num(image, copy=False)

        if self.gpu:
            return {'image': torch.from_numpy(image[np.newaxis, :]).float(),
                    'diagnosis': torch.from_numpy(np.array(diagnosis))}
        else:
            return {'image': torch.from_numpy(image[np.newaxis, :]).float(),
                    'diagnosis': diagnosis}


class MeanNormalization(object):
    """Normalize images using a .nii file with the mean values of all the subjets"""

    def __init__(self, mean_path):
        assert path.isfile(mean_path)
        self.mean_path = mean_path

    def __call__(self, sample):
        reading_mean = nib.load(self.mean_path)
        mean_img = reading_mean.get_data()
        return {'image': sample['image'] - mean_img,
                'diagnosis': sample['diagnosis']}


class LeftHippocampusSegmentation(object):

    def __init__(self):
        self.x_min = 68
        self.x_max = 88
        self.y_min = 60
        self.y_max = 80
        self.z_min = 28
        self.z_max = 48

    def __call__(self, sample):
        image, diagnosis = sample['image'], sample['diagnosis']
        hippocampus = image[self.x_min:self.x_max:, self.y_min:self.y_max:, self.z_min:self.z_max:]
        return {'image': hippocampus,
                'diagnosis': sample['diagnosis']}
