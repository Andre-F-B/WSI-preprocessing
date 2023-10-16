from pathlib import Path

data_folder_path = Path("data/tiff_data")
output_folder_path = Path("data/output")

# Original data paths
raw_data_path = Path(data_folder_path, "raw_data")
rgba_data_path = Path(data_folder_path, "DATA_RGBA")
grayscale_data_path = Path(data_folder_path, "DATA_GRAYSCALE") # change

# Image pair paths
rgba_pair_path = Path(data_folder_path, "rgba_image_pair")
grayscale_pair_path = Path(data_folder_path, "grayscale_image_pair")


