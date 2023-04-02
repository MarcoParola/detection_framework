import fnmatch
import json
import hydra
import os
import shutil





def move_jpg_files_to_images_folder(src_folder, dest_folder):
    file_count = 0

    # Define the allowed file patterns
    file_patterns = ['*.jpg', '*.jpeg', '*.JPG', '*.png', '*.PNG']

    # Create the destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)

    # Iterate through the folder's content
    for entry in os.listdir(src_folder):
        full_path = os.path.join(src_folder, entry)
        # Check if the entry is a file (not a folder)
        if os.path.isfile(full_path):
            # Check if the file matches any of the allowed patterns
            if any(fnmatch.fnmatch(full_path, pattern) for pattern in file_patterns):
                # Move the file to the destination folder
                shutil.move(full_path, os.path.join(dest_folder, entry))
                file_count += 1

    print(f'Moved {file_count} .jpg and .JPG files to the "images" folder.')


# TODO commenta
def get_coco_image_filenames(coco_annotation_file):
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)

    # Create a set of all the "file_name" values in the COCO annotation file
    coco_image_filenames = {image['file_name'] for image in coco_data['images']}
    return coco_image_filenames


# TODO commenta
def remove_images_not_in_coco(images_folder, coco_image_filenames):
    removed_count = 0

    for entry in os.listdir(images_folder):
        full_path = os.path.join(images_folder, entry)

        if os.path.isfile(full_path) and entry not in coco_image_filenames:
            os.remove(full_path)
            removed_count += 1
    print(f'Removed {removed_count} images from the "images" folder that do not have a corresponding "file_name" in the COCO annotation file.')




def remove_missing_images_from_coco(coco_annotation_file, images_folder):
    # Load the COCO annotation file
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)

    # Filter out instances that do not have a corresponding image in the images folder
    filtered_images = []
    image_ids_to_keep = set()
    for image in coco_data['images']:
        if 'file_name' in image:
            file_path = os.path.join(images_folder, image['file_name'])
            if os.path.isfile(file_path):
                filtered_images.append(image)
                image_ids_to_keep.add(image['id'])

    # Filter out annotations that do not have a corresponding image
    filtered_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in image_ids_to_keep]

    # Update the 'images' and 'annotations' keys in the COCO annotation data
    coco_data['images'] = filtered_images
    coco_data['annotations'] = filtered_annotations
    print('Filtered COCO annotation file saved.')
    return coco_data



def save_coco_annotation(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f)



def remove_duplicate_filenames(coco_annotation_file):
    # Load the COCO annotation file
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)

    # Create a dictionary to store filename occurrences
    filename_counts = {}
    for image in coco_data['images']:
        if 'file_name' in image:
            filename = image['file_name']
            filename_counts[filename] = filename_counts.get(filename, 0) + 1

    # Filter out duplicate images
    unique_images = []
    unique_image_ids = set()
    duplicate_images = []
    for image in coco_data['images']:
        if 'file_name' in image and filename_counts[image['file_name']] == 1:
            unique_images.append(image)
            unique_image_ids.add(image['id'])
        else:
            duplicate_images.append(image['file_name'])

    # Filter out annotations that correspond to duplicate images
    unique_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in unique_image_ids]

    # Update the 'images' and 'annotations' keys in the COCO annotation data
    coco_data['images'] = unique_images
    coco_data['annotations'] = unique_annotations
    print('Filtered COCO annotation file saved.')
    return coco_data, duplicate_images



def remove_images(images_to_remove, images_folder):
    for image_file in images_to_remove:
        file_path = os.path.join(images_folder, image_file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f'Removed {file_path}')


def rename_all_images_in_filesystem(coco_file_name, images_folder):

    with open(coco_file_name, 'r') as f:
        coco_data = json.load(f)

    # Iterate through the images in the COCO data
    for image in coco_data['images']:
        old_filename = image['file_name']
        new_filename = f'{image["id"]}.jpg'  # Assuming all images are in jpg format

        # Construct the full file paths
        old_filepath = os.path.join(images_folder, old_filename)
        new_filepath = os.path.join(images_folder, new_filename)

        # Check if the old file exists and rename it
        if os.path.exists(old_filepath):
            os.rename(old_filepath, new_filepath)
            print(f'Renamed "{old_filename}" to "{new_filename}"')
        else:
            print(f'File not found: "{old_filename}"')

        
def rename_all_images_in_cocofile(coco_file_name, image_folder):
    # Load the COCO annotation file
    with open(coco_file_name, "r") as json_file:
        coco_data = json.load(json_file)

    # Iterate through all images and modify the path and file_name
    for image in coco_data["images"]:
        image_id = str(image["id"])
        image["file_name"] = image_id + ".jpg"
        image["path"] = image_folder + image_id + ".jpg"

    return coco_data



@hydra.main(config_path="../../../config/", config_name="config")
def clean(cfg):

    # Replace 'your_source_folder_path' with the path to the folder you want to search
    src_folder = os.path.join(cfg.project_path, 'data/orig/tmp')
    # Replace 'your_destination_folder_path' with the path to the destination folder
    dest_folder = os.path.join(cfg.project_path, cfg.preproc.orig.img_path)
    move_jpg_files_to_images_folder(src_folder, dest_folder)

    # Replace 'your_coco_annotation_file_path' with the path to your COCO annotation file
    coco_annotation_file = os.path.join(cfg.datasets.path, cfg.datasets.original_data, cfg.datasets.filenames.dataset)
    coco_annotation_file_tmp = os.path.join(cfg.datasets.path, cfg.datasets.original_data, 'preprocessed_' + cfg.datasets.filenames.dataset)

    coco_image_filenames = get_coco_image_filenames(coco_annotation_file)
    remove_images_not_in_coco(dest_folder, coco_image_filenames)

    filtered_coco_data = remove_missing_images_from_coco(coco_annotation_file, dest_folder)
    save_coco_annotation(filtered_coco_data, coco_annotation_file_tmp)


    filtered_coco_data, duplicate_image_files = remove_duplicate_filenames(coco_annotation_file_tmp)
    remove_images(duplicate_image_files, dest_folder)
    save_coco_annotation(filtered_coco_data, coco_annotation_file_tmp)
    
    rename_all_images_in_filesystem(coco_annotation_file_tmp, dest_folder)
    
    renamed_coco_data = rename_all_images_in_cocofile(coco_annotation_file_tmp, dest_folder)
    save_coco_annotation(renamed_coco_data, coco_annotation_file_tmp)

    shutil.rmtree(src_folder)


if __name__ == '__main__':
    clean()