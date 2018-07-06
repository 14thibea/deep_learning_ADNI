from torch.utils.data import Dataset
import torch
import pandas as pd
from os import path
from copy import copy
import nibabel as nib
import numpy as np
from nilearn import plotting
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter


bids_cohort_dict = {'ADNI': 'ADNI_BIDS_T1_PET',
                    'AIBL': 'AIBL_BIDS',
                    'OASIS': 'OASIS_BIDS_new'}
minimum_size = np.array([145, 230, 200])
maximum_size = np.array([235, 280, 280])


def crop(image):
    size = np.array(np.shape(image))
    crop_idx = np.rint((size - minimum_size) / 2).astype(int)
    first_crop = copy(crop_idx)
    second_crop = copy(crop_idx)
    for i in range(3):
        if minimum_size[i] + first_crop[i] * 2 != size[i]:
            first_crop[i] -= 1

    cropped_image = image[first_crop[0]:size[0]-second_crop[0],
                          first_crop[1]:size[1]-second_crop[1],
                          first_crop[2]:size[2]-second_crop[2]]

    return cropped_image


def pad(image):
    size = np.array(np.shape(image))
    pad_idx = np.rint((maximum_size - size) / 2).astype(int)
    first_pad = copy(pad_idx)
    second_pad = copy(pad_idx)
    for i in range(3):
        if size[i] + first_pad[i] * 2 != maximum_size[i]:
            first_pad[i] -= 1

    padded_image = np.pad(image, np.array([first_pad, second_pad]).T, mode='constant')

    return padded_image


def transform_bids_image(reading_img, transform='crop'):
    """
    Transformation of BIDS image: transposition of coordinates, flipping coordinages, rescaling voxel size,
    rescaling global size

    """

    header = reading_img.header
    img = reading_img.get_data()

    if len(np.shape(img)) == 4:
        img = img[:, :, :, 0]

    # Transposition
    loc_x = np.argmax(np.abs(header['srow_x'][:-1:]))
    loc_y = np.argmax(np.abs(header['srow_y'][:-1:]))
    loc_z = np.argmax(np.abs(header['srow_z'][:-1:]))
    transposed_image = img.transpose(loc_x, loc_y, loc_z)

    # Directions
    flips = [False, False, False]
    flips[0] = (np.sign(header['srow_x'][loc_x]) == -1)
    flips[1] = (np.sign(header['srow_y'][loc_y]) == -1)
    flips[2] = (np.sign(header['srow_z'][loc_z]) == -1)
    for coord, flip in enumerate(flips):
        if flip:
            transposed_image = np.flip(transposed_image, coord)

    # Resizing voxels
    coeff_x = np.max(np.abs(header['srow_x'][:-1:]))
    coeff_y = np.max(np.abs(header['srow_y'][:-1:]))
    coeff_z = np.max(np.abs(header['srow_z'][:-1:]))
    transposed_size = np.shape(transposed_image)
    transposed_image = transposed_image / np.max(transposed_image)
    new_size = np.rint(np.array(transposed_size) * np.array([coeff_x, coeff_y, coeff_z]))
    resized_image = resize(transposed_image, new_size, mode='constant')

    # Adaptation before rescale
    if transform == 'crop':
        image = crop(resized_image)
    elif transform == 'pad':
        image = pad(resized_image)
    else:
        raise ValueError("The transformations allowed are cropping (transform='crop') or padding (transform='pad')")

    # Final rescale
    rescale_image = resize(image, (121, 145, 121), mode='constant')

    return rescale_image


class BidsMriBrainDataset(Dataset):
    """Dataset of subjects of CLINICA (baseline only) from BIDS"""

    def __init__(self, subjects_df_path, caps_dir, transform=None, classes=2, rescale='crop'):
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
        elif classes == 4:
            self.diagnosis_code = {'CN': 0, 'sMCI': 1, 'pMCI': 2, 'AD': 3}

        self.extension = '_ses-M00_T1w.nii.gz'
        self.folder_path = path.join('ses-M00', 'anat')
        self.rescale = rescale

    def __len__(self):
        return len(self.subjects_df)

    def __getitem__(self, subj_idx):
        subj_name = self.subjects_df.loc[subj_idx, 'participant_id']
        diagnosis = self.subjects_df.loc[subj_idx, 'diagnosis']
        cohort = self.subjects_df.loc[subj_idx, 'cohort']
        img_name = subj_name + self.extension

        data_path = path.join(self.caps_dir, bids_cohort_dict[cohort])
        img_path = path.join(data_path, subj_name, self.folder_path, img_name)

        reading_image = nib.load(img_path)
        image = transform_bids_image(reading_image, self.rescale)

        # Convert diagnosis to int
        if type(diagnosis) is str:
            diagnosis = self.diagnosis_code[diagnosis]

        sample = {'image': image, 'diagnosis': diagnosis, 'name': subj_name}

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

        data_path = path.join(self.caps_dir, bids_cohort_dict[cohort])
        img_path = path.join(data_path, subj_name, self.folder_path, img_name)

        reading_image = nib.load(img_path)
        image = transform_bids_image(reading_image, self.rescale)

        sample = {'image': image, 'diagnosis': diagnosis, 'name': subj_name}

        if use_transforms and self.transform is not None:
            sample = self.transform(sample)

        final_image = nib.Nifti1Image(sample['image'], affine=np.eye(4))
        anat = plotting.plot_anat(final_image, title='subject ' + subj_name, cut_coords=cut_coords)
        anat.savefig(output_path)
        anat.close()


class MriBrainDataset(Dataset):
    """Dataset of subjects of CLINICA (baseline only) from CAPS"""

    def __init__(self, subjects_df_path, caps_dir, transform=None, classes=2, preprocessing='dartel', on_cluster=False):
        """

        :param subjects_df_path: Path to a TSV file with the list of the subjects in the dataset
        :param caps_dir: The CAPS directory where the images are stored
        :param transform: Optional transform to be applied to a sample
        :param classes: Number of classes to consider for classification
            if 2 --> ['CN', 'AD']
            if 3 --> ['CN', 'MCI', 'AD']
        :param processing:
        """
        if type(subjects_df_path) is str:
            self.subjects_df = pd.read_csv(subjects_df_path, sep='\t')
        elif type(subjects_df_path) is pd.DataFrame:
            self.subjects_df = subjects_df_path
        else:
            raise ValueError('Please enter a path or a Dataframe as first argument')

        self.caps_dir = caps_dir
        self.transform = transform
        self.on_cluster = on_cluster

        if classes == 2:
            self.diagnosis_code = {'CN': 0, 'AD': 1}
        elif classes == 3:
            self.diagnosis_code = {'CN': 0, 'MCI': 1, 'AD': 2}

        if preprocessing == 'mni':
            self.extension = '_ses-M00_T1w_segm-graymatter_space-Ixi549Space_modulated-off_probability.nii.gz'
            self.folder_path = path.join('ses-M00', 't1', 'spm', 'segmentation', 'normalized_space')
        elif preprocessing == 'dartel':
            self.extension = '_ses-M00_T1w_segm-graymatter_dartelinput.nii.gz'
            self.folder_path = path.join('ses-M00', 't1', 'spm', 'segmentation', 'dartel_input')
        else:
            ValueError('The directory is a CAPS folder and the preprocessing value entered is not valid.'
                       'Valid values are ["dartel", "mni"]')

    def __len__(self):
        return len(self.subjects_df)

    def __getitem__(self, subj_idx):
        subj_name = self.subjects_df.loc[subj_idx, 'participant_id']
        diagnosis = self.subjects_df.loc[subj_idx, 'diagnosis']
        cohort = self.subjects_df.loc[subj_idx, 'cohort']
        img_name = subj_name + self.extension

        if self.on_cluster:
            caps_name = 'CAPS_' + cohort + '_T1_SPM'
        else:
            caps_name = 'CAPS_' + cohort

        data_path = path.join(self.caps_dir, caps_name, 'subjects')
        img_path = path.join(data_path, subj_name, self.folder_path, img_name)

        reading_image = nib.load(img_path)
        image = reading_image.get_data()

        # Convert diagnosis to int
        if type(diagnosis) is str:
            diagnosis = self.diagnosis_code[diagnosis]

        sample = {'image': image, 'diagnosis': diagnosis, 'name': subj_name}

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

        if self.on_cluster:
            caps_name = 'CAPS_' + cohort + '_T1_SPM'
        else:
            caps_name = 'CAPS_' + cohort

        data_path = path.join(self.caps_dir, caps_name, 'subjects')
        img_path = path.join(data_path, subj_name, self.folder_path, img_name)

        reading_image = nib.load(img_path)
        image = reading_image.get_data()

        sample = {'image': image, 'diagnosis': diagnosis, 'name': subj_name}

        if use_transforms and self.transform is not None:
            sample = self.transform(sample)

        final_image = nib.Nifti1Image(sample['image'], affine=np.eye(4))
        anat = plotting.plot_anat(final_image, title='subject ' + subj_name, cut_coords=cut_coords)
        anat.savefig(output_path)
        anat.close()


class GaussianSmoothing(object):

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, sample):
        image = sample['image']
        np.nan_to_num(image, copy=False)
        smoothed_image = gaussian_filter(image, sigma=self.sigma)
        sample['image'] = smoothed_image

        return sample


class ToTensor(object):
    """Convert image type to Tensor and diagnosis to diagnosis code"""

    def __init__(self, gpu=False):
        self.gpu = gpu

    def __call__(self, sample):
        image, diagnosis, name = sample['image'], sample['diagnosis'], sample['name']
        np.nan_to_num(image, copy=False)

        if self.gpu:
            return {'image': torch.from_numpy(image[np.newaxis, :]).float(),
                    'diagnosis': torch.from_numpy(np.array(diagnosis)),
                    'name': name}
        else:
            return {'image': torch.from_numpy(image[np.newaxis, :]).float(),
                    'diagnosis': diagnosis,
                    'name': name}


class MeanNormalization(object):
    """Normalize images using a .nii file with the mean values of all the subjets"""

    def __init__(self, mean_path):
        assert path.isfile(mean_path)
        self.mean_path = mean_path

    def __call__(self, sample):
        reading_mean = nib.load(self.mean_path)
        mean_img = reading_mean.get_data()
        return {'image': sample['image'] - mean_img,
                'diagnosis': sample['diagnosis'],
                'name': sample['name']}


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
                'diagnosis': sample['diagnosis'],
                'name': sample['name']}


if __name__ == '__main__':
    import torchvision

    subjects_tsv_path = '/Volumes/aramis-projects/elina.thibeausutre/data/2-classes/dataset-ADNI+AIBL+corrOASIS.tsv'
    caps_path = '/Volumes/aramis-projects/CLINICA/CLINICA_datasets/BIDS'
    sigma = 0
    composed = torchvision.transforms.Compose([GaussianSmoothing(sigma),
                                               # ToTensor()
                                               ])

    dataset = BidsMriBrainDataset(subjects_tsv_path, caps_path, transform=composed)
    # lengths = []
    # for i in range(len(dataset)):
    #     image = dataset[i]['image']
    #     lengths.append(np.shape(image))
    #     if i % 100 == 99:
    #         print(i + 1, '/', len(dataset))
    #
    # lengths = np.unique(np.array(lengths), axis=0)
    # print(lengths)
    # length_df = pd.DataFrame(lengths)
    # length_df.to_csv('/Users/elina.thibeausutre/Documents/data/lengths_BIDS.tsv', sep='\t')
    idx = 0
    dataset.imsave(idx, '/Users/elina.thibeausutre/Desktop/smooth' + str(sigma) + '+cropped+doubleresized+normalized_figure' + str(idx))
