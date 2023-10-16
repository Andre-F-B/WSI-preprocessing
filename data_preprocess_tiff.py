import os
import time
from external import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.color as color
import openslide
import SimpleITK as sitk
from skimage.filters import threshold_otsu
from PIL import Image

import utils
import paths
import Parameter
from normalize_staining import normalize_staining


# General setting
# register_levels = [0]
max_patch_size = Parameter.get_value("max_patch_size")

def pre_processing(filename, grayscale_levels=[], rgba_levels=[], manual_crop=False, manual_boundary=None, format=[], show=False, threshold_rgb=True, stain_normalization=True, tissue_threshold=0.001):

    # all_levels = list(set(levels + register_levels))
    combined_levels_set = set(grayscale_levels) | set(rgba_levels)
    all_levels = list(combined_levels_set)

    for extern_file in [file for file in os.listdir(paths.raw_data_path) if file != ".DS_Store"]:
        # print(extern_file)
        # print('filename', filename)
        if (filename == extern_file) and ('.tiff' in extern_file):
            b_t = time.time()

            img_path = os.path.join(paths.raw_data_path, Path(extern_file))
            print("Current WSI: ", img_path)

            # open the whole slide image
            slide = openslide.open_slide(filename=img_path)
            thresholds = get_threshold(slide, threshold_rgb)

            # load the lowest resolution image
            curr_level = slide.level_count - 1
            curr_level_img = np.array(slide.read_region(location=(0, 0), level=curr_level, size=slide.level_dimensions[curr_level]))

            # detect boundaries
            if not manual_crop:
                x, y, w, h = utils.boundary_detect(curr_level_img)
                # print(f'x, y, w, h = {(x, y, w, h)}')

            # I think I can remove this, since the TIFF images don't have manual boundaries
            # else:
            #     assert manual_boundary is not None
            #     x, y, w, h = manual_boundary  # TO DEFINE

            downsample_factor = int(slide.level_downsamples[curr_level] / slide.level_downsamples[0])
            # print(f'downsample_factor = {downsample_factor}')
            # print(f'slide.level_downsamples = {slide.level_downsamples}')
            orig_x = downsample_factor * x  # boundary with the maximal area (level 0)
            orig_y = downsample_factor * y

            if show:  # observe the result (in the current level)
                cropped_img = np.array(slide.read_region(location=(orig_x, orig_y), level=curr_level, size=(w, h)))
                gray_img = 1 - color.rgb2gray(color.rgba2rgb(cropped_img))

                print(f"Image shape before cropping: {curr_level_img.shape[0]}, {curr_level_img.shape[1]}")
                print(f"Image shape after cropping: {cropped_img.shape[0]}, {cropped_img.shape[1]}")

                plt.figure()
                plt.suptitle('Process 1: cropping and grayscale')
                plt.subplot(1, 2, 1)
                plt.imshow(curr_level_img)
                plt.xlabel(f'level {curr_level} - before')
                plt.subplot(1, 2, 2)
                plt.imshow(gray_img, cmap="gray")
                plt.xlabel(f'level {curr_level} - after')
                plt.show()

            # process the given level images
            for level in all_levels:
                print(f"Current level: {level}")
                extracted_patches = 0

                if level in rgba_levels:
                    rgba_output_path = os.path.join(paths.rgba_data_path, filename, f"level_{level}")
                    if not os.path.exists(rgba_output_path):
                        os.makedirs(rgba_output_path)
                if level in grayscale_levels:
                    grayscale_output_path = os.path.join(paths.grayscale_data_path, filename, f"level_{level}")
                    if not os.path.exists(grayscale_output_path):
                        os.makedirs(grayscale_output_path)

                temp_downsample_factor = int(slide.level_downsamples[curr_level] / slide.level_downsamples[level])
                temp_w = temp_downsample_factor * w
                temp_h = temp_downsample_factor * h
                factor = np.power(2, level)


                if min(temp_w, temp_h) > max_patch_size:

                    # When min(height, width) > max_patch_size,
                    # split the image into several parts and save them respectively (prevent memory overflow)
                    n_row = int(np.ceil(temp_h / max_patch_size))
                    n_col = int(np.ceil(temp_w / max_patch_size))

                    step = 0
                    for row in range(n_row):
                        for col in range(n_col):
                            step += 1
                            if step % 10 == 0:
                                print(f"Step: {step} / {n_row * n_col}")

                            x_pos = orig_x + factor * (col * max_patch_size)
                            y_pos = orig_y + factor * (row * max_patch_size)

                            if row == n_row - 1 and col == n_col - 1:
                                curr_w = temp_w % max_patch_size if temp_w % max_patch_size != 0 else max_patch_size
                                curr_h = temp_h % max_patch_size if temp_h % max_patch_size != 0 else max_patch_size
                            elif row == n_row - 1:
                                curr_w = max_patch_size
                                curr_h = temp_h % max_patch_size if temp_h % max_patch_size != 0 else max_patch_size
                            elif col == n_col - 1:
                                curr_w = temp_w % max_patch_size if temp_w % max_patch_size != 0 else max_patch_size
                                curr_h = max_patch_size
                            else:
                                curr_w = max_patch_size
                                curr_h = max_patch_size

                            # load and save the RGBA image patches
                            temp_img_patch = np.array(slide.read_region(location=(x_pos, y_pos), level=0, size=(curr_w, curr_h)))

# ========================================================================================================

                            mask = np.ones((temp_img_patch.shape[0], temp_img_patch.shape[1]))
                            if len(thresholds) > 1:  #: Apply threshold on RGB channels
                                for i, img_channel in enumerate([temp_img_patch[:, :, 0], temp_img_patch[:, :, 1], temp_img_patch[:, :, 2]]):
                                    channel_mask = img_channel < thresholds[i]
                                    channel_mask = np.array(channel_mask, dtype=int)
                                    mask = mask * channel_mask
                            else:  #: Apply threshold on grayscale channel
                                gray_img = np.asarray(Image.fromarray(temp_img_patch).convert('L'))  #: Convert to grayscale
                                channel_mask = gray_img < thresholds[0]
                                mask = np.array(channel_mask, dtype=int)
                            tissue_ratio = np.sum(mask)/mask.size
                            if tissue_ratio > tissue_threshold:
                                if stain_normalization:
                                    # temp_img_patch is sliced because normalize staining needs an RGB input (read_region returns RGBA)
                                    temp_img_patch, _, _ = normalize_staining(temp_img_patch[:, :, :3], HEstain=False)
                                if temp_img_patch is None: # or temp_img_patch.shape != (224, 224, 3):
                                    continue

# ========================================================================================================
                                 # generate and save the rgb image patches
                                if level in rgba_levels:

                                    if 'mha' in format:
                                        to_save_mha = sitk.GetImageFromArray(temp_img_patch[:, :, :3])
                                        to_save_mha_path = os.path.join(rgba_output_path, f"{row}_{col}.mha")
                                        sitk.WriteImage(to_save_mha, to_save_mha_path)
                                    if 'jpg' in format:
                                        to_save_jpg = temp_img_patch[:, :, :3].astype(np.uint8)
                                        to_save_jpg_path = os.path.join(rgba_output_path, f"{row}_{col}.jpg")
                                        mpimg.imsave(to_save_jpg_path, to_save_jpg)


                                # generate and save the grayscale image patches
                                if level in grayscale_levels:
                                    temp_gray_patch = (1 - color.rgb2gray(temp_img_patch[:, :, :3])).astype(np.float32)
                                    if 'mha' in format:
                                        to_save_mha = sitk.GetImageFromArray(temp_gray_patch)
                                        to_save_mha_path = os.path.join(grayscale_output_path, f"{row}_{col}.mha")
                                        sitk.WriteImage(to_save_mha, to_save_mha_path)
                                    if 'jpg' in format:
                                        to_save_jpg = sitk.GetImageFromArray((temp_gray_patch * 255).astype(np.ubyte))
                                        to_save_jpg_path = os.path.join(grayscale_output_path, f"{row}_{col}.jpg")
                                        sitk.WriteImage(to_save_jpg, to_save_jpg_path)
                            
                            else:
                                extracted_patches += 1

                else:
                    temp_img = np.array(slide.read_region(location=(orig_x, orig_y), level=level, size=(temp_w, temp_h)))

                    # For the whole image, there is no need to check the tisuue ratio

                    if stain_normalization:
                        # temp_img_patch is sliced because normalize staining needs an RGB input (read_region returns RGBA)
                        temp_img, _, _ = normalize_staining(temp_img[:, :, :3], HEstain=False)

                    if level in rgba_levels:
                        if 'mha' in format:
                            to_save_mha = sitk.GetImageFromArray(temp_img[:, :, :3])
                            to_save_mha_path = os.path.join(rgba_output_path, "0_0.mha")
                            sitk.WriteImage(to_save_mha, to_save_mha_path)
                        if 'jpg' in format:
                            to_save_jpg = temp_img[:, :, :3].astype(np.uint8)
                            to_save_jpg_path = os.path.join(rgba_output_path, "0_0.jpg")
                            mpimg.imsave(to_save_jpg_path, to_save_jpg)

                    if level in grayscale_levels:
                        # temp_gray_img = (1 - color.rgb2gray(color.rgba2rgb(temp_img))).astype(np.float32)
                        temp_gray_img = (1 - color.rgb2gray(temp_img[:, :, :3])).astype(np.float32)
                        if 'mha' in format:
                            to_save_mha = sitk.GetImageFromArray(temp_gray_img)
                            to_save_mha_path = os.path.join(grayscale_output_path, "0_0.mha")
                            sitk.WriteImage(to_save_mha, to_save_mha_path)
                        if 'jpg' in format:
                            to_save_jpg = sitk.GetImageFromArray((temp_gray_img * 255).astype(np.ubyte))
                            to_save_jpg_path = os.path.join(grayscale_output_path, "0_0.jpg")
                            sitk.WriteImage(to_save_jpg, to_save_jpg_path)

            
            print(f'{extracted_patches} patches were below the tissue thresold of {tissue_threshold} and were ignored')
            e_t = time.time()
            print(f"Time: {e_t - b_t} s")
            print()



def get_threshold(slide, rgb):
    """Calculates the otsu threshold level of a WSI.

    Parameters
    ----------
    slide : OpenSlide
        Whole Slide Image.
    rgb : bool
        Whether the otsu threshold be calculated in RGB channels or in grayscale image.

    Returns
    -------
    thresholds : list
        List of thresholds in respective channels.

    """
    thumbnail = slide.get_thumbnail((slide.dimensions[0] / 256, slide.dimensions[1] / 256))  #: Thumbnail slide
    thresholds = []
    if rgb:
        thumbnail_arr = np.asarray(thumbnail)
        thresholds.extend([threshold_otsu(thumbnail_arr[:, :, 0]),   #: R
                           threshold_otsu(thumbnail_arr[:, :, 1]),   #: G
                           threshold_otsu(thumbnail_arr[:, :, 2])])  #: B
    else:
        thumbnail_arr = np.asarray(thumbnail.convert('L'))  #: Grayscale
        thresholds.append(threshold_otsu(thumbnail_arr))
    return thresholds


