import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import param_manager
import data_preprocess as dp
import data_preprocess_tiff as dp_tiff

import utils
import Parameter
import paths



########################################################################################################################
#                                                       Settings                                                       #
########################################################################################################################

# Paths in paths.py and parameters in param_manager.py might also have to be changed

# image type
mrxs = False
tiff = True # this might work for other formats as well

# [Raw Data Pre-processing]
# In the pre-processing step, the raw data is first manually cropped to remove most of the background areas.
# Then the whole-slide image is segmented into image patches by local reading.
# The image patches are saved as RGBA (used for generation of final results) and grayscale (used for image registration) respectively.
# Paths: (1) grayscale image pairs: paths.grayscale_pair_path
#        (2) rgba image pairs: paths.rgba_pair_path

pre_processing = True

# levels to generate the patches, by type of image
grayscale_levels = [0]
rgba_levels = [0]

# max_patch_size =  Parameter.get_value("max_patch_size")
patch_formats = ['jpg'] # ['mha', 'jpg']
tissue_threshold = 0.03
stain_normalization = True
generate_pairs = True


# boundary list for the H&E-CD8 pairs from UKF_Dataset2 (CytoChroma-WSI)
manual_boundary_list = {
    # Each tuple (x, y, width, height) describes the valid area on the corresponding whole-slide image,
    # (x, y) giving the top left pixel in the level 8 reference frame, (width, height) giving the region size.
    # 'Test1_HE1': (63, 179, 210, 234),
    # 'Test1_CD8_2': (63, 179, 210, 234),
    # 'Test1_HE_3': (63, 179, 210, 234),
    # 'Test2_CD8_1': (109, 186, 210, 234),
    # 'Test2_HE_2': (109, 186, 210, 234),
    # '1_HE': (36, 187, 310, 391),
    # '1_CD8': (36, 187, 310, 391),
    '2_HE': (81, 149, 238, 459),
    '2_CD8': (81, 149, 238, 459),
    # '3_HE': (25, 160, 320, 415),
    # '3_CD8': (25, 160, 320, 415),
    # '4_HE': (20, 215, 318, 335),
    # '4_CD8': (20, 215, 318, 335),
    # '5_HE': (85, 275, 230, 340),
    # '5_CD8': (85, 275, 230, 340),
    # '6_HE': (55, 220, 295, 410),
    # '6_CD8': (55, 220, 295, 410),
    # '7_HE': (58, 270, 301, 263),
    # '7_CD8': (58, 270, 301, 263),
    '8_HE': (22, 277, 280, 286),
    '8_CD8': (22, 277, 280, 286),
    # '9_HE': (81, 224, 265, 399),
    # '9_CD8': (81, 224, 265, 399),
    # '10_HE': (45, 261, 316, 354),
    # '10_CD8': (45, 261, 316, 354),
    # '11_HE': (55, 205, 305, 335),
    # '11_CD8': (55, 205, 305, 335),
    # '12_HE': (18, 187, 337, 376),
    # '12_CD8': (18, 187, 337, 376),
    # '13_HE': (25, 155, 335, 460),
    # '13_CD8': (25, 155, 335, 460),
    # '14_HE': (15, 170, 305, 420),
    # '14_CD8': (15, 170, 305, 420),
    # '15_HE': (45, 194, 247, 286),
    # '15_CD8': (45, 194, 247, 286),
    # '16_HE': (36, 179, 301, 384),
    # '16_CD8': (36, 179, 301, 384),
    # '17_HE': (18, 231, 328, 317),
    # '17_CD8': (18, 231, 328, 317),
    # '18_HE': (20, 245, 350, 360),
    # '18_CD8': (20, 245, 350, 360),
}  # Image boundaries for the manual cropping

image_pair_list = [
    # Each tuple (x, y) describes a pair of images to register, where x is the source image and y is the target image.
    # ('Test1_CD8_2', 'Test1_HE1'),
    # ('Test2_HE_2', 'Test2_CD8_1'),
    # ('Test2_CD8_1', 'Test1_HE1'),
    # ('1_CD8', '1_HE'),
    ('2_CD8', '2_HE'),
    # ('3_CD8', '3_HE'),
    # ('4_CD8', '4_HE'),
    # ('5_CD8', '5_HE'),
    # ('6_CD8', '6_HE'),
    # ('7_CD8', '7_HE'),
    ('8_CD8', '8_HE'),
    # ('9_CD8', '9_HE'),
    # ('10_CD8', '10_HE'),
    # ('11_CD8', '11_HE'),
    # ('12_CD8', '12_HE'),
    # ('13_CD8', '13_HE'),
    # ('14_CD8', '14_HE'),
    # ('15_CD8', '15_HE'),
    # ('16_CD8', '16_HE'),
    # ('17_CD8', '17_HE'),
    # ('18_CD8', '18_HE'),
]  # The image pairs to be registered





########################################################################################################################
#                                                Algorithm starts here                                                 #
########################################################################################################################


if mrxs:

    if pre_processing:
        print("Start data pre-processing...")
        for filename in manual_boundary_list.keys():
            manual_boundary = manual_boundary_list[filename]
            dp.pre_processing(filename, grayscale_levels=grayscale_levels, rgba_levels=rgba_levels, manual_crop=True, manual_boundary=manual_boundary, format=patch_formats, show=False, tissue_threshold=tissue_threshold, stain_normalization=stain_normalization)

        if generate_pairs:
            dp.generate_image_pairs(image_pair_list)



elif tiff:

    if pre_processing:  # data pre-processing step
        print("Start data pre-processing...")
        for filename in os.listdir(paths.raw_data_path):
            dp_tiff.pre_processing(filename, grayscale_levels=grayscale_levels, rgba_levels=rgba_levels, manual_crop=False, manual_boundary=None, format=patch_formats, show=False, tissue_threshold=tissue_threshold, stain_normalization=stain_normalization)
            # dp_tiff.generate_image_pairs(image_pair_list)



