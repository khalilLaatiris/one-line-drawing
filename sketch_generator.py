import cv2
import numpy as np
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.FileHandler('sketch_generator.log', mode='a')
try:
    import linedraw
except ImportError:
    print("linedraw not installed. Vectorization will be disabled.")
    linedraw = None

def sketch(img, blur_strength=21, dodge_intensity=2.5):
    """
    Applies a pencil sketch effect to an image.

    Args:
        img_path (str): Path to the input image.
        blur_strength (int): Strength of the Gaussian blur (odd number).
        dodge_intensity (float): Intensity of the dodge blend.

    Returns:
        numpy.ndarray: The sketched image.
    """
    

    # Pre-process for low contrast
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    logging.info("Converted image to grayscale")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    logging.info("Applied CLAHE to grayscale image")

    # 2. Invert
    gray_inv = 255 - gray
    logging.info("Inverted grayscale image")

    # 3. Gaussian Blur
    gray_blur = cv2.GaussianBlur(gray_inv, (blur_strength, blur_strength), 0)
    logging.info("Applied Gaussian blur")

    # 4. Dodge Blend
    def dodge_blend(gray_img, gray_blur_img, dodge_intensity):
        return cv2.divide(gray_img, 255 - gray_blur_img, scale=256*dodge_intensity)

    dodge = dodge_blend(gray, gray_blur, dodge_intensity)
    logging.info("Applied dodge blend")

    # 5. Final Inversion
    final = 255 - dodge
    logging.info("Inverted dodge blend")
    final = np.clip(final, 0, 255).astype('uint8')
    logging.info("Clipped final image")

    logging.info("Sketch function completed successfully")
    return final


def apply_style_preset(img, preset):
    """Applies a style preset to the image.
        manga: Manga style,
        watercolor: Watercolor style,
        blueprint: Blueprint style,
    """
    logging.info(f"Entering apply_style_preset with preset: {preset}")
    if preset == 'manga':
        kernel_size = (3, 3)
        dilation_iterations = 2
        img = cv2.dilate(img, np.ones(kernel_size, np.uint8), iterations=dilation_iterations)
        logging.info("Applied manga style")
    elif preset == 'watercolor':
        sigma_s = 150
        sigma_r = 0.3
        img = cv2.bilateralFilter(img, d=15, sigmaColor=sigma_r * 255, sigmaSpace=sigma_s)
        logging.info("Applied watercolor style")
    elif preset == 'blueprint':
        img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
        logging.info("Applied blueprint style")
    logging.info("apply_style_preset completed")
    return img


def vectorize(img_path, vector_output_path, draw_contours=2, draw_hatch=4):
    """
    Vectorizes the sketch using linedraw.
    Args:
        img_path (str): Path to the input image.
        output_path (str): Path to the sketch image.
        vector_output_path (str): Path to save the SVG file.
        draw_contours (int): Number of contour levels to draw.
        draw_hatch (int): Density of hatching.
    """
    if linedraw is None:
        print("linedraw is not installed. Cannot vectorize.")
        return

    lines = linedraw.sketch(img_path, draw_contours=draw_contours, draw_hatch=draw_hatch)
    logging.info(f"Vectorized sketch using linedraw.sketch")
    vector_output_path = f"output/{image_name}_vector.svg"
    linedraw.save_svg(lines, vector_output_path)
    logging.info(f"Saved vectorized sketch to {vector_output_path}")
    print(f"Vectorized sketch saved to {vector_output_path}")
    logging.info("vectorize function completed")


def canny_edge_detection(img):
    """Applies Canny edge detection to an image."""
    logging.info("Entering canny_edge_detection function")
    edges = cv2.Canny(img, 200, 115)
    logging.info("Applied Canny edge detection")
    cv2.imwrite(f"output/{image_name}_canny.png", edges)
    logging.info(f"canny_edge_detection function completed and Saved Canny edges to output/{image_name}_canny.png")


def xdog(img,sigma=1.0, k=1.6, tau=0.98):
    """Applies XDoG (Extended Difference-of-Gaussians) to an image.
    default parameters are set for a typical XDoG effect.
    sigma: 1.0 Standard deviation for the small Gaussian.
    k: 1.6 Ratio of the large Gaussian to the small Gaussian.
    tau: 0.98 Threshold for the tanh function.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    logging.info("Converted image to grayscale for XDoG")
    
    gauss_small = cv2.GaussianBlur(img, (0, 0), sigma)
    logging.info("Applied Gaussian blur (small)")
    gauss_large = cv2.GaussianBlur(img, (0, 0), sigma * k)
    logging.info("Applied Gaussian blur (large)")
    dog = gauss_small - gauss_large
    result = 255 * (1 - np.tanh(tau * dog))
    logging.info("Applied tanh function")
    result = np.clip(result, 0, 255).astype(np.uint8)
    logging.info("Clipped XDoG result")
    cv2.imwrite(f"output/{image_name}_xdog.png", result)
    logging.info(f"Saved XDoG to output/{image_name}_xdog.png")
    print(f"XDoG saved to output/{image_name}_xdog.png")
    logging.info("xdog function completed")


if __name__ == '__main__':
    logging.info("Starting script")
    parser = argparse.ArgumentParser(description='Convert an image to a pencil sketch.')
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    parser.add_argument('--blur_strength', type=int, default=21, help='Strength of the Gaussian blur (odd number).')
    parser.add_argument('--dodge_intensity', type=float, default=2.5, help='Intensity of the dodge blend.')
    parser.add_argument('--vectorize', action='store_true', help='Enable vectorization of the sketch.')
    parser.add_argument('--style_preset', type=str, default=None, help='Apply a style preset (manga, watercolor, blueprint).')
    parser.add_argument('--canny', action='store_true', help='Enable Canny edge detection.')
    parser.add_argument('--xdog', action='store_true', help='Enable XDoG (Extended Difference-of-Gaussians).')

    args = parser.parse_args()

    try:
        
        
        # Extract filename from image path
        image_filename = os.path.basename(args.image_path)
        global image_name # This variable is used in xdog and vectorize functions
        image_name = os.path.splitext(image_filename)[0]

        img = cv2.imread(args.image_path)
        if img is None:
            logging.error(f"Image not found at {args.image_path}")
            raise FileNotFoundError(f"Image not found at {args.image_path}")
        logging.info(f"Image loaded successfully from {args.image_path}")
        
        # Standardize image size
        height, width = img.shape[:2]
        max_dim = max(height, width)
        max_appropriate_size = 1080  # Standard size for processing
        if max_dim > max_appropriate_size:
            scale = max_appropriate_size / max_dim
            new_height = int(height * scale)
            new_width = int(width * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logging.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        else:
            logging.info(f"Image size {width}x{height} is within standard size (max {max_appropriate_size}px)")

        logging.info("Calling sketch function")
        sketch_img = sketch(img.copy(), args.blur_strength, args.dodge_intensity)
        logging.info("sketch function completed")

        # Determine the output filename based on style preset
        if args.style_preset:
            logging.info("Calling apply_style_preset function")
            sketch_img = apply_style_preset(sketch_img, args.style_preset)
            logging.info("apply_style_preset function completed")
            output_filename = f"output/{image_name}_{args.style_preset}.png"
        else:
            output_filename = f"output/{image_name}_sketch.png"

        # Ensure the output directory exists
        os.makedirs("output", exist_ok=True)

        cv2.imwrite(output_filename, sketch_img)
        print(f"Sketch saved to {output_filename}")

        if args.vectorize:
            logging.info("Calling vectorize function")
            vectorize(img.copy(), output_filename, f"output/{image_name}_vector.svg")
        logging.info("vectorize function completed")

        if args.canny:
            image_filename = os.path.basename(args.image_path)
            image_name = os.path.splitext(image_filename)[0]
            canny_edge_detection(img.copy())

        if args.xdog:
            xdog(img.copy())
    except Exception as e:
        print(f"An error occurred: {e}")