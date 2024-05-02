from cProfile import label
import numpy as np
import base64
import pywt
import urllib.request
from flask import Flask, json, redirect, render_template, request, jsonify, send_from_directory, url_for
import os
import shutil
from skimage.transform import warp
from skimage.segmentation import flood
from skimage.filters import threshold_otsu
from scipy.interpolate import Rbf
from scipy.spatial import Delaunay
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter
from skimage.segmentation import active_contour
from cv2 import SIFT_create, BFMatcher, pointPolygonTest, remap, INTER_LINEAR

from scipy.ndimage import distance_transform_edt
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import map_coordinates
# from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
from scipy.ndimage import zoom
from skimage.transform import warp,ProjectiveTransform
from werkzeug.utils import secure_filename
from skimage.segmentation import felzenszwalb
from skimage import color, data, restoration
from skimage.segmentation import active_contour
from scipy.signal import convolve2d
from skimage.transform import PiecewiseAffineTransform, warp
from cv2 import (CHAIN_APPROX_SIMPLE, COLOR_BGR2LAB, COLOR_BGR2XYZ, GC_INIT_WITH_RECT, MORPH_HITMISS, RETR_EXTERNAL, BFMatcher, add, bitwise_and, boundingRect, contourArea, convertScaleAbs, fillPoly, findContours, findHomography, flip, getGaussianKernel, grabCut, imread,imdecode,equalizeHist,IMREAD_GRAYSCALE,
                 imencode,createCLAHE,GaussianBlur, imwrite,medianBlur,Laplacian,Sobel,normalize,addWeighted, pointPolygonTest, polylines, rectangle, remap,subtract,filter2D,
                 NORM_MINMAX,CV_8U,CV_64F,pyrDown,pyrUp,resize,erode,dilate,getRotationMatrix2D,warpAffine,getAffineTransform,
                 morphologyEx,MORPH_OPEN,MORPH_CLOSE,MORPH_GRADIENT,MORPH_TOPHAT,MORPH_BLACKHAT,MORPH_ELLIPSE,MORPH_RECT,
                 COLOR_BGR2RGB,COLOR_RGB2HSV,COLOR_RGB2LAB,cvtColor,IMREAD_COLOR,Canny,threshold,DIST_L2,THRESH_BINARY_INV,THRESH_OTSU,
                 connectedComponents,distanceTransform,COLOR_BGR2GRAY,split,merge,THRESH_BINARY,INTER_LINEAR,COLOR_BGR2HSV,
                 COLOR_BGR2YCrCb,Scharr, warpPerspective,)#Prewitt,Roberts,COLOR_BGR2CMYK ,watershed
from cv2 import cvtColor, COLOR_BGR2GRAY, imread, watershed, imwrite, connectedComponents

app = Flask(__name__)

UPLOAD_FOLDER = r'static/uploads'
OPERATIONS_FOLDER = r'static/operations'
RESULT_FOLDER = r'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OPERATIONS_FOLDER'] = OPERATIONS_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def main():
  return render_template('index.html')




@app.route('/upload', methods=['POST'])
def upload_images():
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images found in request'}), 400
        
        images = request.files.getlist('images')

        for image in images:
            if image and allowed_file(image.filename):
                filename = secure_filename(image.filename)
                image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                shutil.copy(os.path.join(app.config['UPLOAD_FOLDER'], filename), os.path.join(app.config['OPERATIONS_FOLDER'], filename))
            else:
                return jsonify({'error': 'Invalid file type or extension'}), 400
        
        return redirect(url_for("image_enhancement"))

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/image_enhancement')
def image_enhancement():
    # Get the path to the images folder from app.config
    image_folder = app.config['OPERATIONS_FOLDER']
    # Get list of image filenames from the folder
    image_files = os.listdir(image_folder)
    # Pass the list of image filenames to the HTML template
    return render_template('image_enhancement.html', image_files=image_files)


@app.route('/geometrical_transformation_page')
def geometrical_transformation_page():
    # Get the path to the images folder from app.config
    image_folder = app.config['OPERATIONS_FOLDER']
    # Get list of image filenames from the folder
    image_files = os.listdir(image_folder)

    return render_template('geometrical_transformation.html',image_files=image_files)

@app.route('/morphological_operation_page')
def morphological_operation_page():
     # Get the path to the images folder from app.config
    image_folder = app.config['OPERATIONS_FOLDER']
    # Get list of image filenames from the folder
    image_files = os.listdir(image_folder)

    return render_template('morphological_operation.html',image_files=image_files)

@app.route('/color_space_transform_page')
def color_space_transform_page():
    image_folder = app.config['OPERATIONS_FOLDER']
    image_files = os.listdir(image_folder)
    return render_template('color_space_transform.html',image_files=image_files)

@app.route('/edge_detection_page')
def edge_detection_page():
    image_folder = app.config['OPERATIONS_FOLDER']
    image_files = os.listdir(image_folder)
    return render_template('edge_detection.html',image_files=image_files)


@app.route('/roi_page')
def roi_page():
    image_folder = app.config['OPERATIONS_FOLDER']
    image_files = os.listdir(image_folder)
    return render_template('roi_extraction.html',image_files=image_files)

###################################### Image Enhacement 


@app.route('/histogram_equalization', methods=['POST'])
def histogram_equalization():
    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path, IMREAD_GRAYSCALE)
            equalized_image = equalizeHist(image)
            imwrite(img_path, equalized_image)

    return jsonify({'message': 'Histogram equalization applied successfully'}), 200

@app.route('/adaptive_histogram_equalization', methods=['POST'])
def adaptive_histogram_equalization():
    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path, IMREAD_GRAYSCALE)
            equalized_image = equalizeHist(image)
            clahe = createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            adaptive_equalized_image = clahe.apply(equalized_image)
            imwrite(img_path, adaptive_equalized_image)

    return jsonify({'message': 'Histogram equalization applied successfully'}), 200

@app.route('/cla_histogram_equalization', methods=['POST'])
def cla_histogram_equalization():
    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path, IMREAD_GRAYSCALE)
            clahe = createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_image = clahe.apply(image)
            imwrite(img_path, clahe_image)
    return jsonify({'message': 'Histogram equalization applied successfully'}), 200

@app.route('/guass_noise', methods=['POST'])
def guass_noise():
    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path, IMREAD_GRAYSCALE)
            blurred_image = GaussianBlur(image, (5, 5), 0)
            imwrite(img_path, blurred_image)
    return jsonify({'message': 'Histogram equalization applied successfully'}), 200

@app.route('/median_noise', methods=['POST'])
def median_noise():
    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path, IMREAD_GRAYSCALE)
            filtered_image = medianBlur(image,ksize=5)
            imwrite(img_path, filtered_image)
    return jsonify({'message': 'Histogram equalization applied successfully'}), 200

def wavelet_denoise(image):
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs
    threshold = 10  # Adjust threshold value as needed
    cA_thresh = pywt.threshold(cA, threshold, mode='soft')
    cH_thresh = pywt.threshold(cH, threshold, mode='soft')
    cV_thresh = pywt.threshold(cV, threshold, mode='soft')
    cD_thresh = pywt.threshold(cD, threshold, mode='soft')
    denoised_coeffs = (cA_thresh, (cH_thresh, cV_thresh, cD_thresh))
    denoised_image = pywt.idwt2(denoised_coeffs, 'haar')
    return denoised_image

@app.route('/wavelet_noise', methods=['POST'])
def wavelet_noise():
    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path, IMREAD_GRAYSCALE)
            # Apply wavelet transformation for noise reduction
            denoised_image = wavelet_denoise(image)
            imwrite(img_path, denoised_image)
    return jsonify({'message': 'Noise reduction applied successfully'}), 200

@app.route('/laplacian_edge', methods=['POST'])
def laplacian_edge():
    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path, IMREAD_GRAYSCALE)
            # Apply Laplacian filter for edge enhancement
            filtered_image = Laplacian(image, CV_64F)
            filtered_image = convertScaleAbs(filtered_image)
            imwrite(img_path, filtered_image)
    return jsonify({'message': 'Edge enhancement applied successfully'}), 200

@app.route('/highpass_edge', methods=['POST'])
def highpass_edge():
    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path, IMREAD_GRAYSCALE)
            # Apply high-pass filter for edge enhancement
            kernel = np.array([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]], dtype=np.float32)  # Define the high-pass filter kernel
            filtered_image = filter2D(image, -1, kernel)
            imwrite(img_path, filtered_image)
    return jsonify({'message': 'highpass_edge applied successfully'}), 200

@app.route('/unsharp_edge', methods=['POST'])
def unsharp_edge():
    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path, IMREAD_GRAYSCALE)
            # Apply unsharp masking for edge enhancement
            blurred = GaussianBlur(image, (5, 5), 0)
            unsharp_mask = addWeighted(image, 2, blurred, -1, 0)
            imwrite(img_path, unsharp_mask)
    return jsonify({'message': 'unsharp_edge applied successfully'}), 200

@app.route('/wiener_deblur', methods=['POST'])
def wiener_deblur():
    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path, IMREAD_GRAYSCALE)
            # Apply Wiener filter for deblurring
            psf = np.ones((5, 5)) / 25  # Assuming a simple averaging blur kernel
            deblurred_image = filter2D(image, -1, psf)
            deblurred_image = convertScaleAbs(deblurred_image)
            imwrite(img_path, deblurred_image)
    return jsonify({'message': 'wiener_deblur applied successfully'}), 200


def richardson_lucy_deconv(image, psf, iterations=10):
    # Initial estimation of the latent image
    latent_image = np.copy(image)
    
    # Perform Richardson-Lucy deconvolution iterations
    for _ in range(iterations):
        # Convolve the current estimate with the PSF
        blurred_image = filter2D(latent_image, -1, psf)
        
        # Calculate the error between the observed and blurred images
        error = image / (blurred_image + 1e-10)
        
        update = filter2D(error, -1, flip(psf, -1))
        update_scaled = ((latent_image * update - latent_image.min()) / (latent_image.max() - latent_image.min()) * 255).astype(np.uint8)
        latent_image = update_scaled

    return latent_image

@app.route('/blind_deconvolution', methods=['POST'])
def blind_deconvolution():
    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path, IMREAD_GRAYSCALE)
            # Assuming you have a PSF, for example, a Gaussian blur kernel
            psf = getGaussianKernel(5, 1)
            psf = np.outer(psf, psf.transpose())
            
            # Perform blind deconvolution using Richardson-Lucy algorithm
            deblurred_image = richardson_lucy_deconv(image, psf)
            deblurred_image = np.uint8(deblurred_image)
            
            imwrite(img_path, deblurred_image)
    return jsonify({'message': 'Blind deconvolution applied successfully'}), 200

def build_gaussian_pyramid(image, levels=3):
    pyramid = [image]
    for _ in range(levels - 1):
        image = pyrDown(image)
        pyramid.append(image)
    return pyramid

def build_laplacian_pyramid(image, levels=3):
    gaussian_pyramid = build_gaussian_pyramid(image, levels)
    laplacian_pyramid = []
    for i in range(levels - 1):
        expanded = pyrUp(gaussian_pyramid[i + 1], dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
        print("Expanded shape:", expanded.shape)
        print("Gaussian shape:", gaussian_pyramid[i].shape)
        laplacian = subtract(gaussian_pyramid[i], expanded)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid

@app.route('/pyramid_multiscale', methods=['POST'])
def pyramid_multiscale():
    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path, IMREAD_GRAYSCALE)
            # Build Laplacian pyramid
            laplacian_pyramid = build_laplacian_pyramid(image)
            # Reconstruct image from Laplacian pyramid
            reconstructed_image = laplacian_pyramid[0]
            for i in range(1, len(laplacian_pyramid)):
                # expanded = pyrUp(reconstructed_image, dstsize=(2 * laplacian_pyramid[i].shape[1], 2 * laplacian_pyramid[i].shape[0]))
                expanded = pyrUp(reconstructed_image, dstsize=None)

                print("Reconstructed shape:", reconstructed_image.shape)
                print("Laplacian shape:", laplacian_pyramid[i].shape)
                # reconstructed_image = add(expanded, laplacian_pyramid[i])
                # Resize expanded to match laplacian_pyramid[i]
                expanded_resized = resize(expanded, (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0]))

                # Add resized expanded image to laplacian_pyramid[i]
                reconstructed_image = add(expanded_resized, laplacian_pyramid[i])

            imwrite(img_path, reconstructed_image)
    return jsonify({'message': 'Multiscale processing applied successfully'}), 200



@app.route('/wavelet_multiscale', methods=['POST'])
def wavelet_multiscale():
    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path, IMREAD_GRAYSCALE)          
            # Perform wavelet transform
            coeffs = pywt.dwt2(image, 'haar')
            # Reconstruction
            reconstructed_image = pywt.idwt2(coeffs, 'haar')
            imwrite(img_path, reconstructed_image)
    return jsonify({'message': 'highpass_edge applied successfully'}), 200

@app.route('/logcompression_dra', methods=['POST'])
def logcompression_dra():
    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path, IMREAD_GRAYSCALE)          
            # Apply log compression to adjust dynamic range
            c = 255 / np.log(1 + np.max(image))
            compressed_image = np.uint8(c * (np.log(image + 1)))
            imwrite(img_path, compressed_image)
    return jsonify({'message': 'logcompression_dra applied successfully'}), 200


@app.route('/exponential_dra', methods=['POST'])
def exponential_dra():
    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path, IMREAD_GRAYSCALE)          
            # Apply exponential transformation to adjust dynamic range
            c = 255 / np.exp(np.max(image))
            adjusted_image = np.uint8(c * (np.exp(image)))
            imwrite(img_path, adjusted_image)
    return jsonify({'message': 'exponential_dra applied successfully'}), 200



################################   Geometrical Transformation  

@app.route('/translation', methods=['POST'])
def translate_image():
    translation_x = 50  # Translation along x-axis
    translation_y = 30  # Translation along y-axis

    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path)
            rows, cols = image.shape[:2]
            translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
            translated_image = warpAffine(image, translation_matrix, (cols, rows))
            imwrite(img_path, translated_image)

    return jsonify({'message': 'Images translated successfully'}), 200

@app.route('/rotation', methods=['POST'])
def rotate_image():
    rotation_angle = 90  # Rotation angle in degrees

    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path)
            rows, cols = image.shape[:2]
            rotation_matrix = getRotationMatrix2D((cols/2, rows/2), rotation_angle, 1)
            rotated_image = warpAffine(image, rotation_matrix, (cols, rows))
            imwrite(img_path, rotated_image)

    return jsonify({'message': 'Images rotated successfully'}), 200

@app.route('/scaling', methods=['POST'])
def scale_image():
    scale_factor = 1.5  # Scale factor for resizing the image

    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path)
            scaled_image = resize(image, None, fx=scale_factor, fy=scale_factor)
            imwrite(img_path, scaled_image)

    return jsonify({'message': 'Images scaled successfully'}), 200


@app.route('/shearing', methods=['POST'])
def shear_image():
    shear_factor = 0.5  # Shear factor for shearing the image

    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path)
            rows, cols = image.shape[:2]
            shear_matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])
            sheared_image = warpAffine(image, shear_matrix, (cols, rows))
            imwrite(img_path, sheared_image)

    return jsonify({'message': 'Images sheared successfully'}), 200


@app.route('/flipping', methods=['POST'])
def flip_image():
    print('entered flipping')
    flip_direction = -1  # 1 for horizontal flip, 0 for vertical flip, -1 for both

    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path)
            flipped_image = flip(image, flip_direction)
            imwrite(img_path, flipped_image)

    return jsonify({'message': 'Images flipped successfully'}), 200


@app.route('/elastic_deformation', methods=['POST'])
def elastic_deformation():
    sigma = 15  # Standard deviation for the random displacement
    alpha = 50  # Alpha parameter for controlling the intensity of the displacement field

    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path)
            rows, cols = image.shape[:2]

            # Generate random displacement fields
            dx = alpha * GaussianBlur(np.random.randn(rows, cols), (0, 0), sigma)
            dy = alpha * GaussianBlur(np.random.randn(rows, cols), (0, 0), sigma)

            # Create meshgrid of pixel coordinates
            x, y = np.meshgrid(np.arange(cols), np.arange(rows))
            
            # Apply displacement to pixel coordinates
            x_distorted = (x + dx).astype(np.float32)
            y_distorted = (y + dy).astype(np.float32)

            # Map distorted pixel coordinates to original image
            distorted_image = remap(image, x_distorted, y_distorted, interpolation=INTER_LINEAR)

            imwrite(img_path, distorted_image)

    return jsonify({'message': 'Images elastic deformation applied successfully'}), 200

@app.route('/cropping', methods=['POST'])
def crop_image():
    crop_x = 100  # Starting x-coordinate of the crop region
    crop_y = 50   # Starting y-coordinate of the crop region
    crop_width = 200  # Width of the crop region
    crop_height = 150  # Height of the crop region

    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path)
            cropped_image = image[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
            imwrite(img_path, cropped_image)

    return jsonify({'message': 'Images cropped successfully'}), 200

@app.route('/resampling', methods=['POST'])
def resample_image():
    new_width = 800  # New width of the image
    new_height = 600  # New height of the image

    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path)
            resized_image = resize(image, (new_width, new_height))
            imwrite(img_path, resized_image)

    return jsonify({'message': 'Images resampled successfully'}), 200




@app.route('/projective_transformation', methods=['POST'])
def projective_transformation():
    # Load source and destination points
    source_points = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
    destination_points = np.float32([[50, 50], [150, 50], [200, 150], [0, 150]])

    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path)
            
            # Perform projective transformation (homography)
            h, _ = findHomography(source_points, destination_points)
            transformed_image = warp(image, h, output_shape=(image.shape[0], image.shape[1]))
            imwrite(img_path, transformed_image)

    return jsonify({'message': 'Images transformed using projective transformation successfully'}), 200




@app.route('/non_rigid_registration', methods=['POST'])
def non_rigid_registration():
    # Load source and destination control points
    source_control_points = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
    destination_control_points = np.float32([[50, 50], [150, 50], [200, 150], [0, 150]])

    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path)

            # Perform non-rigid registration
            tform = PiecewiseAffineTransform()
            tform.estimate(source_control_points, destination_control_points)
            transformed_image = warp(image, tform, output_shape=(image.shape[0], image.shape[1]))
            imwrite(img_path, transformed_image)

    return jsonify({'message': 'Images transformed using non-rigid registration successfully'}), 200


#################################       Morphological Operations

def perform_morphological_operation(image, operation):
    # Define structuring element (kernel)
    kernel = np.ones((5, 5), np.uint8)   

    # Perform morphological operations
    if operation == 'erosion':
        return erode(image, kernel, iterations=1)
    elif operation == 'dilation':
        return dilate(image, kernel, iterations=1)
    elif operation == 'opening':
        return morphologyEx(image, MORPH_OPEN, kernel)
    elif operation == 'closing':
        return morphologyEx(image, MORPH_CLOSE, kernel)
    elif operation == 'gradient':
        return morphologyEx(image, MORPH_GRADIENT, kernel)
    elif operation == 'tophat':
        return morphologyEx(image, MORPH_TOPHAT, kernel)
    elif operation == 'bottomhat':
        return morphologyEx(image, MORPH_BLACKHAT, kernel)
    elif operation == 'blackhat':
        return morphologyEx(image, MORPH_BLACKHAT, kernel)
    elif operation == 'hitmiss':
        kernel_hit_miss = np.array([[0, 1, 0], [1, -1, 1], [0, 1, 0]], dtype=np.int8)
        return morphologyEx(image, MORPH_HITMISS, kernel_hit_miss)
    else:
        return None

@app.route('/erosion', methods=['POST'])
def erosion_operation():
    return perform_operation('erosion')

@app.route('/dilation', methods=['POST'])
def dilation_operation():
    return perform_operation('dilation')

@app.route('/opening', methods=['POST'])
def opening_operation():
    return perform_operation('opening')

@app.route('/closing', methods=['POST'])
def closing_operation():
    return perform_operation('closing')

@app.route('/gradient', methods=['POST'])
def gradient_operation():
    return perform_operation('gradient')

@app.route('/tophat', methods=['POST'])
def tophat_operation():
    return perform_operation('tophat')

@app.route('/bottomhat', methods=['POST'])
def bottomhat_operation():
    return perform_operation('bottomhat')

@app.route('/blackhat', methods=['POST'])
def blackhat_operation():
    return perform_operation('blackhat')

@app.route('/hitmiss', methods=['POST'])
def hitmiss_operation():
    return perform_operation('hitmiss')


def perform_operation(operation):
    operation_images = os.listdir(OPERATIONS_FOLDER)

    
    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path, IMREAD_GRAYSCALE)
        
        # Perform the specified morphological operation
        result_image2 = perform_morphological_operation(image, operation)
        imwrite(img_path, result_image2)
    return jsonify({'message': 'Morphological operations  applied successfully'}), 200



############################   COLOR SPACE TRANSFORMATION    #########################################################

def perform_color_space_transformation(image, transformation):
    if len(image.shape) == 2:  # Grayscale image
        if transformation == 'grayscale':
            return image
        else:
            return None  # Cannot perform other transformations on grayscale image
    elif len(image.shape) == 3 and image.shape[2] == 3:  # BGR image
        if transformation == 'grayscale':
            return cvtColor(image, COLOR_BGR2GRAY)
        elif transformation == 'hsv':
            return cvtColor(image, COLOR_BGR2HSV)
        elif transformation == 'ycbcr':
            return cvtColor(image, COLOR_BGR2YCrCb)
        elif transformation == 'cmyk':
            return None  # OpenCV does not directly support CMYK
        elif transformation == 'lab':
            return cvtColor(image, COLOR_BGR2LAB)
        else:
            return None
    else:
        return None  # Unsupported image format

@app.route('/grayscale', methods=['POST'])
def grayscale_operation():
    return perform_transformation2('grayscale')

@app.route('/hsv', methods=['POST'])
def hsv_operation():
    return perform_transformation2('hsv')

@app.route('/ycbcr', methods=['POST'])
def ycbcr_operation():
    return perform_transformation2('ycbcr')

@app.route('/cmyk', methods=['POST'])
def cmyk_operation():
    return perform_transformation2('cmyk')

@app.route('/lab', methods=['POST'])
def lab_operation():
    return perform_transformation2('lab')



def perform_transformation2(transformation):
    operation_images = os.listdir(OPERATIONS_FOLDER)    
    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path)
            result_image = perform_color_space_transformation(image, transformation)
            if result_image is not None:
                imwrite(img_path, result_image)
                print(f"Saved image: {img_path}")  # Debug print
            else:
                print(f"Failed to transform image: {img_path}")  # Debug print
        else:
            print(f"Invalid image path: {img_path}")  # Debug print
    return jsonify({'message': 'Color Space operations applied successfully'}), 200

###################  EDGE DETECTION ##################################################################################
def perform_edge_detection_main(image, operation):
    if operation == 'sobel':
        return Sobel(image, -1, 1, 1, ksize=3)
    elif operation == 'canny':
        return Canny(image, 100, 200)
    elif operation == 'laplacian_of_gaussian':
        blurred = GaussianBlur(image, (3, 3), 0)
        return Laplacian(blurred, -1)
    elif operation == 'roberts_cross':
        # return Roberts(image)
        return image
    elif operation == 'prewitt':
        # return Prewitt(image)
        return image
    elif operation == 'scharr':
        return Scharr(image, -1, 1, 0)
    else:
        return None

@app.route('/sobel', methods=['POST'])
def sobel_operation():
    return perform_edge_detection('sobel')

@app.route('/canny', methods=['POST'])
def canny_operation():
    return perform_edge_detection('canny')

@app.route('/laplacian_of_gaussian', methods=['POST'])
def laplacian_of_gaussian_operation():
    return perform_edge_detection('laplacian_of_gaussian')

@app.route('/roberts_cross', methods=['POST'])
def roberts_cross_operation():
    return perform_edge_detection('roberts_cross')

@app.route('/prewitt', methods=['POST'])
def prewitt_operation():
    return perform_edge_detection('prewitt')

@app.route('/scharr', methods=['POST'])
def scharr_operation():
    return perform_edge_detection('scharr')

# @app.route('/zero_crossing', methods=['POST'])
# def zero_crossing_operation():
#     return perform_edge_detection('zero_crossing')

def perform_edge_detection(operation):
    operation_images = os.listdir(OPERATIONS_FOLDER)  
    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path, IMREAD_GRAYSCALE)
        
        # Perform the specified morphological operation
        result_image4 = perform_edge_detection_main(image, operation)  # Issue here
        imwrite(img_path, result_image4)
    return jsonify({'message': 'Morphological operations  applied successfully'}), 200

@app.route('/zero_crossing', methods=['POST'])
def zero_crossing():
    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path, IMREAD_GRAYSCALE)
            # if image is None:
            #     return jsonify({'error': f'Failed to load image: {img_path}'}), 400

            # Apply Laplacian filter for edge enhancement
            filtered_image = Laplacian(image, CV_64F)
            filtered_image = convertScaleAbs(filtered_image)
            
            # Apply zero-crossing detector
            edge_image = zero_crossing_detector(filtered_image)

            imwrite(img_path, edge_image)

    return jsonify({'message': 'Zero-crossing edge detection applied successfully'}), 200

def zero_crossing_detector(image):
    # Define kernel for zero-crossing detector
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    # Apply convolution
    convolved_image = filter2D(image, -1, kernel)

    # Apply thresholding to highlight zero crossings
    threshold_value = threshold_otsu(convolved_image)
    edge_image = np.where(convolved_image > threshold_value, 255, 0).astype(np.uint8)

    return edge_image
############################################    ROI EXtraction 

@app.route('/threshold_roi', methods=['POST'])
def threshold_roi():
    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path, IMREAD_GRAYSCALE)
            # Apply Laplacian filter for edge enhancement
            filtered_image = Laplacian(image, CV_64F)
            filtered_image = convertScaleAbs(filtered_image)
            
            # Apply thresholding
            threshold_value = threshold_otsu(filtered_image)
            binary_image = np.where(filtered_image > threshold_value, 255, 0).astype(np.uint8)

            imwrite(img_path, binary_image)
    return jsonify({'message': 'Edge enhancement and thresholding applied successfully'}), 200

def define_region_of_interest(image):
    # You need to define your region of interest here, for example, using thresholding
    threshold_value = threshold_otsu(image)
    binary_mask = image > threshold_value
    
    # Ensure that the footprint size is odd along all dimensions
    footprint = np.ones((3, 3), dtype=np.uint8)  # Example footprint with size 3x3
    return binary_mask, footprint


def region_growing(image, seed_point, region_of_interest, footprint):
    # Perform region growing from the seed point within the region of interest
    segmented_image = flood(image, seed_point, tolerance=10, connectivity=1, footprint=footprint)
    return segmented_image.astype(np.uint8)

@app.route('/region_growing_roi', methods=['POST'])
def region_growing_roi():
    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            image = imread(img_path, IMREAD_GRAYSCALE)
            # Apply Laplacian filter for edge enhancement
            filtered_image = Laplacian(image, CV_64F)
            filtered_image = convertScaleAbs(filtered_image)

            # Define seed point for region growing (you may need to adjust this)
            seed_point = (50, 50)  # Example seed point

            # Define region of interest and footprint
            region_of_interest, footprint = define_region_of_interest(filtered_image)
            
            # Perform region growing with defined region of interest and footprint
            segmented_image = region_growing(filtered_image, seed_point, region_of_interest, footprint)

            imwrite(img_path, segmented_image)

    return jsonify({'message': 'Region growing applied successfully'}), 200




@app.route('/watershed_roi', methods=['POST'])
def watershed_roi():
    # Path to folder containing operation images
    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            # Read the image
            image = imread(img_path)
            gray = cvtColor(image, COLOR_BGR2GRAY)
            
            # Threshold the image to obtain markers for watershed
            ret, thresh = threshold(gray, 0, 255, THRESH_BINARY_INV + THRESH_OTSU)
            
            # Noise removal
            kernel = np.ones((3,3),np.uint8)
            opening = morphologyEx(thresh,MORPH_OPEN,kernel, iterations = 2)
            
            # Sure background area
            sure_bg = dilate(opening,kernel,iterations=3)
            
            # Finding sure foreground area
            dist_transform = distanceTransform(opening,DIST_L2,5)
            ret, sure_fg = threshold(dist_transform,0.7*dist_transform.max(),255,0)
            
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = subtract(sure_bg,sure_fg)
            
            # Marker labelling
            ret, markers = connectedComponents(sure_fg)
            
            # Add one to all labels so that sure background is not 0, but 1
            markers = markers+1
            
            # Now, mark the region of unknown with zero
            markers[unknown==255] = 0
            
            # Apply watershed algorithm
            markers = watershed(image,markers)
            image[markers == -1] = [255,0,0]  # Color regions where watershed algorithm placed boundaries
            
            # Save the image with watershed segmentation
            imwrite(img_path, image)
            
    return jsonify({'message': 'Watershed segmentation applied successfully to segment ROIs'}), 200


@app.route('/active_contour_roi', methods=['POST'])
def active_contour_roi():
    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            # Read the image
            image = imread(img_path)
            gray = cvtColor(image, COLOR_BGR2GRAY)
            
            # Initialize the snake contour
            snake = np.array([[100, 100], [100, 200], [200, 200]])
            
            # Apply active contour model
            snake = active_contour(gray, snake, alpha=0.015, beta=10, gamma=0.001)
            
            # Convert snake to integer coordinates
            snake = np.array(snake, dtype=int)
            
            # Draw the snake contour on the image
            polylines(image, [snake], isClosed=False, color=(0, 255, 0), thickness=2)
            
            # Save the image with the snake contour
            imwrite(img_path, image)
            
    return jsonify({'message': 'Active contour model applied successfully to segment ROIs'}), 200



@app.route('/graph_cut_roi', methods=['POST'])
def graph_cut_roi():
    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            # Read the image
            image = imread(img_path)
            
            # Create a mask initialized with zeros
            mask = np.zeros(image.shape[:2], np.uint8)
            
            # Define the region of interest (ROI) rectangle
            rect = (50, 50, 450, 290)  # Format: (x, y, width, height)
            
            # Apply GrabCut algorithm
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            grabCut(image, mask, rect, bgd_model, fgd_model, 5, GC_INIT_WITH_RECT)
            
            # Modify the mask to keep only the probable foreground pixels
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Apply the modified mask to the original image
            segmented_image = image * mask2[:, :, np.newaxis]
            
            # Save the segmented image
            imwrite(img_path, segmented_image)
            
    return jsonify({'message': 'Graph cut applied successfully to segment ROIs'}), 200


@app.route('/level_set_roi', methods=['POST'])
def level_set_roi():
    operation_images = os.listdir(OPERATIONS_FOLDER)

    for img_name in operation_images:
        img_path = os.path.join(OPERATIONS_FOLDER, img_name)
        if os.path.isfile(img_path):
            # Read the image
            image = imread(img_path)
            gray = cvtColor(image, COLOR_BGR2GRAY)
            
            # Create an initial level set contour
            x = np.linspace(50, 400, 100)
            y = np.linspace(50, 250, 100)
            snake_init = np.array([x, y]).T
            
            # Evolve the contour using the level set method
            snake = active_contour(distance_transform_edt(1 - gray), snake_init, alpha=0.01, beta=1.0, gamma=0.001, w_line=0, w_edge=1)
            
            # Convert snake to integer coordinates
            snake = np.array(snake, dtype=int)
            
            # Create a binary mask from the evolved contour
            mask = np.zeros(gray.shape, dtype=np.uint8)
            fillPoly(mask, [snake], color=255)
            
            # Dilate the mask to ensure coverage of the ROI
            mask = binary_dilation(mask, iterations=3)
            
            # Apply the mask to the original image
            # segmented_image = bitwise_and(image, image, mask=mask)
            # Apply the mask to the original image
            segmented_image = bitwise_and(image, image, mask=mask.astype(np.uint8))

            
            # Save the segmented image
            imwrite(img_path, segmented_image)
            
    return jsonify({'message': 'Level set method applied successfully to segment ROIs'}), 200


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)