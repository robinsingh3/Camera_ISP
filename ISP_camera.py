import numpy as np
import cv2

# ISP Pipeline Functions

def black_level_correction(raw_image, black_level):
    corrected_image = raw_image - black_level
    corrected_image = np.clip(corrected_image, 0, None)
    return corrected_image

def defective_pixel_correction(image):
    from scipy.ndimage import median_filter
    corrected_image = median_filter(image, size=3)
    return corrected_image

def noise_reduction(image):
    denoised_image = cv2.bilateralFilter(image, d=5, sigmaColor=75, sigmaSpace=75)
    return denoised_image

def lens_shading_correction(image, shading_map):
    corrected_image = image / shading_map
    corrected_image = np.clip(corrected_image, 0, 255)
    return corrected_image

def demosaic(image):
    demosaiced_image = cv2.cvtColor(image, cv2.COLOR_BayerBG2RGB)
    return demosaiced_image

def white_balance(image):
    result = image.copy().astype(np.float32)
    avgR = np.mean(result[:, :, 0])
    avgG = np.mean(result[:, :, 1])
    avgB = np.mean(result[:, :, 2])
    avgGray = (avgR + avgG + avgB) / 3

    result[:, :, 0] *= avgGray / avgR
    result[:, :, 1] *= avgGray / avgG
    result[:, :, 2] *= avgGray / avgB

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def color_correction(image, ccm):
    shape = image.shape
    image_flat = image.reshape(-1, 3)
    corrected_flat = np.dot(image_flat, ccm.T)
    corrected_image = corrected_flat.reshape(shape)
    corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)
    return corrected_image

def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    corrected_image = cv2.LUT(image, table)
    return corrected_image

def tone_mapping(image):
    image = image.astype(np.float32) / 255.0
    mapped = image / (image + 1)
    mapped = np.clip(mapped * 255, 0, 255).astype(np.uint8)
    return mapped

def sharpening(image):
    gaussian = cv2.GaussianBlur(image, (9, 9), 10.0)
    sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened

# Main ISP Pipeline Function

def isp_pipeline(raw_image, black_level, shading_map, ccm, gamma):
    # Step 1: Black Level Correction
    image = black_level_correction(raw_image, black_level)

    # Step 2: Defective Pixel Correction
    image = defective_pixel_correction(image)

    # Step 3: Noise Reduction (Optionally enhanced by ML)
    # image = denoise_with_dncnn(image)  # Uncomment if using ML
    image = noise_reduction(image)

    # Step 4: Lens Shading Correction
    image = lens_shading_correction(image, shading_map)

    # Step 5: Demosaicing (Optionally enhanced by ML)
    image = demosaic(image)

    # Step 6: White Balance
    image = white_balance(image)

    # Step 7: Color Correction (Optionally enhanced by ML)
    image = color_correction(image, ccm)

    # Step 8: Gamma Correction
    image = gamma_correction(image, gamma)

    # Step 9: Tone Mapping (Optionally enhanced by ML)
    image = tone_mapping(image)

    # Step 10: Sharpening
    image = sharpening(image)

    return image

# Example Usage

if __name__ == "__main__":
    # Load raw image (simulated for this example)
    raw_image = cv2.imread('raw_image.png', cv2.IMREAD_GRAYSCALE)

    # Simulated parameters
    black_level = 64  # Example black level
    shading_map = np.ones_like(raw_image, dtype=np.float32)  # Assuming flat shading map
    ccm = np.array([[1.5, -0.3, -0.2],
                    [-0.2, 1.4, -0.2],
                    [-0.1, -0.5, 1.8]])  # Example CCM
    gamma = 2.2  # Example gamma value

    # Run ISP Pipeline
    final_image = isp_pipeline(raw_image, black_level, shading_map, ccm, gamma)

    # Display the final image
    cv2.imshow('Final Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
