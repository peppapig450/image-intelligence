import argparse

import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
Use opencv2 to attempt to identify the amount of bills in a photo.
"""

def load_image(image_path: str):
    """
    Load an image from the specified file path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        image (numpy.ndarray): Loaded image.
    """
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


def preprocess_image(image):
    """
    Preprocess the image by converting it to grayscale, applying Gaussian blur, and performing edge detection.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        edges (numpy.ndarray): Edge-detected image.
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    plt.subplot(121),plt.imshow(image,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
 
    plt.show()
    return edges

def filter_out_images(countour):
    """
    Filter based on size and aspect ratio in order to detect the bills.
    
    Args:
        countour (numpy.ndarray): Countour being filtered.
    
    Returns:

    """

def find_bills(edges):
    """
    Find contours in the edge-detected image and filter them to detect money bills.

    Args:
        edges (numpy.ndarray): Edge-detected image.

    Returns:
        bills (list): List of contours representing money bills.
    """
    countours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    aspect_ratio_range = (2.2, 2.7) # Acceptable range for the aspect ratio of a bill
    bills = []
    
    for countour in countours:
        epsilon = 0.02 * cv2.arcLength(countour, True)
        approx = cv2.approxPolyDP(countour, epsilon, True)
        
        if len(approx) == 4: # Check if the countour is a quadrilateral
            x, y, w, h = cv2.boundingRect(countour)
            aspect_ratio = float(w) / h
            print(aspect_ratio)
            
            if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                bills.append(countour)
    return bills

def draw_bills(image, bills):
    """
    Draw the contours of detected money bills on the original image.

    Args:
        image (numpy.ndarray): Original image.
        bills (list): List of contours representing money bills.

    Returns:
        image_with_bills (numpy.ndarray): Image with drawn contours of money bills.
    """
    cv2.drawContours(image, bills, -1, (0, 255, 0), 3)
    return image

def display_image(image):
    """
    Display the image using OpenCV.

    Args:
        image (numpy.ndarray): Image to be displayed.
    """
    cv2.imshow("Image with Detected Bills", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(image_path):
    """
    Main function to detect and display the number of money bills in an image.

    Args:
        image_path (str): Path to the image file.
    """
    image = load_image(image_path)
    edges = preprocess_image(image)
    bills = find_bills(edges)
    image_with_bills = draw_bills(image, bills)
    display_image(image_with_bills)
    print(f"Number of bills detected: {len(bills)}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect money bills in an image.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    args = parser.parse_args()
    main(args.image_path)