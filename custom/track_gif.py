import os
from PIL import Image
import imageio

def get_folders_in_directory(directory):
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    return folders

def create_gif(input_folder, output_gif):
    images = []

    # Ensure the output folder exists
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)

    # Get a list of image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Sort the image files to maintain order
    image_files.sort()

    # Read images and append to the list
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        img = Image.open(image_path)
        images.append(img)

    # Save the images as a GIF
    imageio.mimsave(output_gif, images, duration=.5)  # Adjust the duration as needed

if __name__ == "__main__":
    video_name = "video-00030-2019_06_26_13_41_08/"
    root = "C:/Users/Aaron/Desktop/uchicago-aviansolar-detect-track/data/" + video_name
    input_folder = root + video_name + video_name
    tracks = get_folders_in_directory(input_folder)

    print(tracks)

    for i in tracks:
        output_gif = root + video_name + i + '.gif'
        print(output_gif)
        create_gif(input_folder + i, output_gif)
        print(f"GIF created at {output_gif}")
