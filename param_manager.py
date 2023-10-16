"""

"""
import os
from pathlib import Path

import Parameter
import paths

Parameter._init()  # Initialize the global parameter dict


### Raw data pre-processing params
Parameter.set_value("max_patch_size", 2048)  # Maximum patch size allowed for the local image reading (must be a power of 2)
Parameter.set_value("manual_boundary_list", {
    # Each tuple (x, y, width, height) describes the valid area on the corresponding whole-slide image,
    # (x, y) giving the top left pixel in the level 8 reference frame, (width, height) giving the region size.
    # 'Test1_HE1': (63, 179, 210, 234),
    # 'Test1_CD8_2': (63, 179, 210, 234),
    # 'Test1_HE_3': (63, 179, 210, 234),
    # 'Test2_CD8_1': (109, 186, 210, 234),
    # 'Test2_HE_2': (109, 186, 210, 234),
    '1_HE': (36, 187, 310, 391),
    '1_CD8': (36, 187, 310, 391),
    # '2_HE': (81, 149, 238, 459),
    # '2_CD8': (81, 149, 238, 459),
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
    # '8_HE': (22, 277, 280, 286),
    # '8_CD8': (22, 277, 280, 286),
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
    # '18_CD8': (20, 245, 350, 360)
})















