"""
module that a result file based on the samples. The result file will be used for further analysis
"""

__author__ = "Maryam Jalali"

import os
import pandas as pd
import numpy as np
import img_functions as imf
import yaml


def get_config():
    with open("config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    return config

config = get_config()
path_to_dir = config['path_to_dir']
lookup_file = config['srm']

list_of_filenames = []
list_of_keynames = []
list_of_paths = []

#fetch files
files = os.listdir(path_to_dir)
for image_name in files:
    list_of_paths.append(f'{path_to_dir}{image_name}') 
    list_of_filenames.append(image_name.split('.')[:-1:]) 
    
#create empty dataframe
df_results = pd.DataFrame(columns=['sample', 
                                   'method',
                                   'masked',
                                   'grey',
                                   'red', 
                                   'green', 
                                   'blue', 
                                   'SRM', 
                                   'EBC', 
                                   'brightness', 
                                  ])


df_lookup_file = pd.read_csv(lookup_file)

methods = ['random sampling', 'pca + random sampling', 'most dominant', 'pca + most dominant', 'raw']

# loop over the files and apply the methods on them
for image in list_of_paths:
    
    for method in methods:
        
        if method == 'random sampling':
            pca = False
            masked = True
            dominant_color = False
        
        if method == 'pca + random sampling':
            pca = True
            masked = True
            dominant_color = False  
            
        if method == 'most dominant':
            pca = False
            masked = True
            dominant_color = True 
            
        if method == 'pca + most dominant':
            pca = True
            masked = True
            dominant_color = True 
            
        if method == 'raw':
            pca = False
            masked = False
            dominant_color = False
            
            
        #load image
        img = imf.read_images(image)

        if pca == True:
            img = imf.compress_img(img,20)

        if masked == True:
            img = imf.masking_function(img, img, path_to_dir)

        random_color = imf.random_list(img) 
        
        if dominant_color == True:
            red, green, blue = imf.cluster_colors(img)
            
        else:
            # random list function returns a list, get_average-color needs 3d array
            arr = np.array(random_color).reshape(len(random_color),1,3)
            # to get average of RGB numbers withouth black colors 
            red, green, blue = imf.get_average_color(arr)
            
            
        srm = int(imf.srm_value(red, green, blue, df_lookup_file))
    
        # to get the average of brightness of each image
        list_of_brightness = imf.rgb_to_brightness(random_color,imf.perceived_luminescence)
        calculate_brightness = round(imf.avg_brightness(list_of_brightness))
    
        # EBC formula is 1.97*SRM 
        ebc = round(1.97*srm)
        
        # to get the average of brightness of each image
        list_of_grayscale = imf.rgb_to_grayscale(random_color,imf.perceived_grayscale)
        calculate_grayscale = round(imf.avg_grayscale(list_of_grayscale))
    
        # it shows only the name of the file and the path is removed 
        df_results.loc[len(df_results.index)] = [image.split('_')[1].split('.')[0], #sample
                                                 method, #method
                                                 masked, 
                                                 calculate_grayscale,
                                                 red,
                                                 green,
                                                 blue,
                                                 srm, 
                                                 ebc, 
                                                 calculate_brightness,
                                                ] 

df_results.to_csv('generated_results.csv', index = False, sep = '\t')