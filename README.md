# WSI-preprocessing

Breaks MRXS/TIFF images down into patches, saving them as RGB or grayscale images. Optionally, removes patches with too much background, applies stain normalization to the patches. For MRXS images, patches from corresponding images (e.g. 18_HE.mrxs and 18_CD8.mrxs) can be paired. 

### File Structure

- Unified_code
    - Python scripts
    - data
        - tiff_data
            - DATA_MRXS
            - DATA_RGBA
            - raw_data
        - mrxs_data
            - DATA_MRXS
            - DATA_RGBA
            - raw_data
    

### Parameters

- Set image type (MRXS or TIFF)
    - Set data_folder_path in paths.py accordingly (mrxs_data or tiff_data)
    - The TIFF pipeline might also work for other image formats
- pre_processing (boolean): performs patch extraction, background removal and stain normalization
    - Image levels to get the patches from (in grayscale and RGBA)
        - For TIFF images, the level must be 0 (no downscaling)
    - Maximum patch size (param_manager.py)
    - Image format to be used for the patches (MHA and/or JPG)
    - Tissue threshold: the minimum amount of tissue to be detected on a patch for it to be saved
    - Stain normalization (boolean): whether or not to apply stain normalization (Macenko method) to the patches
        - Macenko M, Niethammer M, Marron JS, Borland D, Woosley JT, Guan X, et al. A method for normalizing histology slides for quantitative analysis. In: 2009 IEEE International Symposium on Biomedical Imaging. Boston, MA (2009). p. 1107–10.
    - If the images are MRXS:
        - Decide whether to generate image pairs or not
        - Uncomment the manual boundaries of the images being processed
        - Uncomment the image pairs

### data_preprocess.py

- Breaks down the image into patches
- Filters out patches with tissue ratio lower than the tissue threshold
- Applies stain normalization using the Macenko method
    - TIFF images are normalized based on H-DAB
    - MRXS images are normalized based on either H&E or H-DAB (depending on the file’s name)
- Saves the final patches as RGBA and grayscale JPG images
- For MRXS images
    - Creates CD8-H&E image pairs
