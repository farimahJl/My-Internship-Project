#!/usr/bin/env python3

"""
Module with functions for the beer-0-meter project
"""

__author__ = "Maryam Jalali"


import numpy as np

#todo this needs to be written from a configfile
PATH_TO_DIR = r'C:\Users\Farimah\Desktop\SGPapertronics\Python Files\Colour test\ebc 14.07.22 (1)'


   
def read_images(path_to_img: str) -> np.array:
    """
    Reads an image from a file and returns it as a numpy array in RGB format.

    Parameters
    ----------
    path_to_img : str
        The file path to the image.

    Returns
    -------
    np.array
        The image as a numpy array in RGB format.

    """
    read_img = cv2.imread(path_to_img)
    img1 = cv2.cvtColor(read_img,cv2.COLOR_BGR2RGB)
    return np.array(img1)



# in case if we want to see how the image looks and what size is the image, 
def image_show(image_name):
    """
    This function displays an image using OpenCV library. The function takes an image as input
    and displays it in a window with the title "Image:". 
    The function uses the cv2.imshow() method to display the image and 
    cv2.waitKey() to wait for a keyboard event. 

    Parameters
    ----------
    image_name : numpy array
        The image to be displayed.

    """
    cv2.imshow("Image:", image_name)
    cv2.waitKey()



def random_list(image_name: np.array) -> list:
    """
    Creates a list of 50 randomly selected RGB values from an image.

    Parameters
    ----------
    image_name : numpy array
        The image from which to select the RGB values.

    Returns
    -------
    randomized_list: list
        A list of 50 randomly selected RGB values from the image.

    """
    randomized_list = []
    for item in range(0, 50):
        selected_item = random.choice(random.choice(image_name))
        if not np.all(np.logical_not(selected_item)):
            randomized_list.append(selected_item)
    return (randomized_list)



# This function applies 2 masks (inner and outer) on the unimportant parts of the image and 
# it has been calibrated via several trials 
def masking_function(image_name:np.array, file_name: str) -> np.array:

    """
    Function to apply masks on unimportant parts of the image.
    author: M. Jalali & T. Kok
    
    Parameters:
    ----------
        image_name (np.array): The input image as a numpy array
        file_name (str): The name of the file to which the masked image is saved.
    
    Returns:
    ----------
        np.array: The masked image as a numpy array.
    
    Note:
    ----------
        The function applies two masks on the image using cv2.circle function to create two circles - one outer and one inner. 
        The area outside the outer circle and inside the inner circle is masked. The masked image is then saved to the specified 
        path as a PNG image.
    """
    mask1 = np.zeros(image_name.shape[:2], dtype="uint8")
    mask2 = np.zeros(image_name.shape[:2], dtype="uint8")
    
    # we put 2 circles( outer ( big one ) and inner ( small one ) to get rid of the unimportant data
    
    mask_1 = cv2.circle(mask1, (250, 250), 200, 255, -1)
    mask_2 = cv2.circle(mask2, (250, 250), 50, 255, -1)
    
    mask = mask_1 - mask_2

    masked = cv2.bitwise_and(image_name, image_name, mask=mask)

    # via these lines we apply the masking function     
    
    path_to_dir = PATH_TO_DIR
    if not cv2.imwrite(f'{path_to_dir}\\{file_name}_masked.png', masked):
         raise Exception("Could not write image")
    return masked



def get_average_color(arr:np.array) -> tuple:
    """
    Calculates the average RGB values of an image. The input is a numpy array arr which represents the image. 
    The function first changes any black pixels in the image to np.nan values, 
    as np.nan values are ignored when calculating the mean with np.nanmean(). 
    The average values for red, green, and blue are calculated along both axis 0 and 1, 
    resulting in a single average value for each of the RGB channels. 
    The final average values for red, green, and blue are rounded

    Parameters
    ----------
    arr : numpy array
        The image to calculate the average RGB values for.

    Returns
    -------
    tuple
        A tuple of integers representing the average RGB values for the image.

    """
    #read the image into numpy array, pic_x, pic_y, RGB
    arr = arr.astype('float') # make floats for using np.nan otherwise it will crash
    # change black colors in nan
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            if arr[row, col, :].sum() == 0: #if red, blue, green == 0 == black
                arr[row, col, :] = np.nan # change it to nan
  
    red, green, blue = np.nanmean(arr, axis = (0,1)) # calculate average rgb values keeping RGB column
  
    return round(red), round(green), round(blue)



def closest_color(r:int, g:int, b:int, lookup_table:dict) -> str:
    """
    This function finds the closest color in the `lookup_table` to the RGB values (r, g, b).

    Parameters
    ----------
        r: The red component of the RGB color.
        g: The green component of the RGB color.
        b: The blue component of the RGB color.
        lookup_table (pandas df): The lookup table of RGB color values.

    Returns
    -------
        closest_color: The closest color in the `lookup_table` to the RGB values (r, g, b).

    """
    COLORS = lookup_table['RGB']
    color_diffs = []
    for color in COLORS:
        color = color.replace('(', '')
        color = color.replace(')', '')
        cr, cg, cb = color.split(',')
        cr = int(cr)
        cg = int(cg)
        cb = int(cb)
        color_diff = np.sqrt(abs(r - cr)**2 + abs(g - cg)**2 + abs(b - cb)**2)
        color_diffs.append((color_diff, color)) # store diff with color from srm table
        closest_color = min(color_diffs)[1] # search minimum diff and extract color part
    return closest_color



def srm_value(r:int, g:int, b:int, lookup_table:dict) -> str:
    """
    This function finds the closest srm value in the `lookup_table` to the RGB values (r, g, b).

    Parameters
    ----------
        r: The red component of the RGB color.
        g: The green component of the RGB color.
        b: The blue component of the RGB color.
        lookup_table (pandas df): The lookup table of srm values.

    Returns
    -------
        srm value
    """
    cc = closest_color(r, g, b, lookup_table)
    #print(f'the closest color of ({r,g,b}) is {cc}')
    #get location from table with closest color and extract srm value
    srm = lookup_table.loc[lookup_table['RGB'] == f'({cc})', 'srm']
    #print(f'the srm value that belongs to closest color is {srm.values[0]}')
    return f'{srm.values[0]}'



def perceived_luminescence(rgb_pair: list) -> float:
    '''
    calculate brightness using the formula: Brightness = 0.2126*R + 0.7152*G + 0.0722*B
    
    
    Parameters
    ----------
        rgb_pair: list with RGB colors
    
    Returns
    -------
        brightness_value
    '''
    brightness_value = 0.2126*rgb_pair[0] + 0.2126*rgb_pair[1] + 0.2126*rgb_pair[2]
    return brightness_value



def rgb_to_brightness(list_of_values: list, brightness_formula) -> list:
    '''
    convert list of rgb values to list of perceived luminescence (brighntess) values.
    
    
    Parameters:
    ----------
        list_of_values(list(list(int)) : list containing 50 randomly samples RGB values from the image.
        brightness_formula(function)   : function that defines how to calculate the brightness from RGB.
    
    Returns:
    ----------
        list_of_brightness(list(float)) : list with the brightness values.
    '''
    list_of_brightness = []
    for rgb_pair in list_of_values:
        list_of_brightness.append(brightness_formula(rgb_pair))
    return list_of_brightness



def avg_brightness(list_brightness: list) -> float:
    '''
    calculate average of list of brightness values

    Parameters:
    -----------
        list_brightness(list(float)) : list with the brightness values.

    Returns:
    --------
         average brightness value (float)

    '''
    avg_val = np.average(list_brightness)
    return avg_val



def perceived_grayscale(rgb_pair: list) -> float:
    '''
    get the perceived grayscale using the formula: grayscale = 0.299*R + 0.587*G + 0.114*B
    
    Parameters
    ----------
        rgb_pair: list with RGB colors
    
    Returns
    -------
        grayscale_value (float)

    '''
    grayscale_value = 0.299*rgb_pair[0] + 0.587*rgb_pair[1] + 0.114*rgb_pair[2]
    return grayscale_value



def rgb_to_grayscale(list_of_values: list, grayscale_formula) -> list:
    '''
    convert list of rgb values to list of grayscale values.
    
    
    Parameters:
    -------
        list_of_values(list(list(int)) : list containing 50 randomly samples RGB values from the image.
        grayscale_formula(function)   : function that defines how to calculate the grayscale from RGB.
    
    Returns:
    -------
        list_of_grayscale(list(float)) : list with grayscale values
    '''
    list_of_grayscale = []
    for rgb_pair in list_of_values:
        list_of_grayscale.append(grayscale_formula(rgb_pair))
    return list_of_grayscale



def avg_grayscale(list_grayscale: list) -> float:
    '''
    Parameters:
    -----------
        list_grayscale(list(float)) : list with the grayscale values.

    Returns:
    --------
         average grayscale value (float)

    '''
    avg_val_gray = np.average(list_grayscale)
    return avg_val_gray


# to compress the picture into bigger pixels and shrink the data and deduct the dimention 
# using the method of PCA
def compress(X:np.array, n:int) -> np.array:
    """
    compressing color channel using PCA
    source: https://github.com/fenna/student_BFVM19DATASC3
    
    Parameters:
    -----------
        X (np.array): 2 dimensional channel array (pixels) of the picture, either red, green or blue chanel
        n (int): number of components

    Returns:
    --------
        recovered: rebuild picture chanel with compressed colors
    """
    pca = PCA(n_components=n)
    reduced = pca.fit_transform(X) 
    recovered = pca.inverse_transform(reduced)
    return recovered



def compress_img(img: np.array, n:int) -> np.array:
    """
    compressing picture using PCA. 
    First split the 3 dimonsional array in channels (2D) then compress them using PCA
    then rebuild the picture from compressed channels.
    source: https://github.com/fenna/student_BFVM19DATASC3
    
    Parameters:
    -----------
        X (np.array): 3 dimensional image array
        n (int): number of components

    Returns:
    --------
        img: rebuild picture compressed colors
    """
    # n = number of pixels and it is adjustable 
    # run PCA on each channel and combine afterwards in
    X_red = img[:, : , 0]
    X_green = img[:, : , 1]
    X_blue = img[:,:,2]
    compressed_red = compress(X_red, n)
    compressed_green = compress(X_green, n)
    compressed_blue = compress(X_blue, n)
    img = (np.dstack((compressed_red, compressed_green, compressed_blue))).astype(np.uint8)
    return img



def get_dominant_color(cluster:object, centroids:list, mask:bool = True) -> tuple:

    """
    This function returns the most dominant color in an image based on the cluster labels and centroids of the image.
    The function computes the histogram of the cluster labels to find the percentage of each color cluster present in the image. 
    It then sorts the colors cluster by their frequency and returns the most dominant color as an RGB tuple. 
    If mask is set to True, the second most dominant color is returned instead, to ignore the dominant color black from the mask. 
    # source: https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv

    Parameters:
    ----------
        cluster (object): An object that has been fit using some clustering algorithm and contains the cluster labels.
        centroids (list): A list of centroid RGB color values.
        mask (bool, optional): A flag indicating whether to ignore the most dominant color in case of a mask. Defaults to True.

    Returns:
    --------
        tuple: An RGB tuple representing the most dominant color in the image.
    """

    n = len(np.unique(cluster.labels_))
    labels = np.arange(0, n + 1)
    # trick to cluster in bins per label
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum() # get percentage included black
 
    # combine percentage with colors
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
   
    #ignore the most dominant in case of mask, since dominant will be black !!!!
    if mask: 
        dominant_color = list(*colors[-2:-1])[1] 
    else:
        dominant_color = list(*colors[-1:])[1]

    red, green, blue = dominant_color
    return round(red), round(green), round(blue)


def cluster_colors(img) -> tuple: 
    """
    This function returns the dominant color in an image using KMeans clustering.
    # source: https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv


    Parameters:
    ----------
        img (ndarray): An image in BGR format.

    Returns:
    --------
        tuple: An RGB tuple representing the dominant color in the image.
    """
    image = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    reshape = image.reshape((img.shape[0] * image.shape[1], 3))
    # Find and display most dominant colors
    cluster = KMeans(n_clusters=5).fit(reshape)
    blue, green, red  = get_dominant_color(cluster, cluster.cluster_centers_, True)
    return red, green, blue



def visualize_colors(cluster, centroids):
    """
    function to visualize most dominant colors used for validation purpose
    # source: https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv
    
    Parameters:
    ----------
        cluster (object): An object that has been fit using some clustering algorithm and contains the cluster labels.
        centroids (list): A list of centroid RGB color values.

    Returns:
    ----------
        rectangle with colors
    
    """
    rect_len = 400
    # Get the number of different clusters, create histogram, and normalize
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # Create frequency rect and iterate through each cluster's color and percentage
    rect = np.zeros((50, rect_len, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    start = 0

    for (percent, color) in colors:     
        print(color, "{:0.2f}%".format(percent * 100))
        end = start + (percent * rect_len)  
        cv2.rectangle(rect, (int(start), 0), (int(end), 50), \
                      color.astype("uint8").tolist(), -1)
        start = end

    for (percent, color) in colors[:-1]: #ignore black   
        pass

    return rect