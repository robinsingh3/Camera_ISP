# Camera_ISP


### Overview of the ISP Pipeline
The ISP pipeline processes raw sensor data through several stages:

Black Level Correction: Adjusts the sensor's black level offset.
Defective Pixel Correction: Fixes dead or stuck pixels.
Noise Reduction (Denoising): Reduces sensor noise.
Lens Shading Correction: Corrects vignetting effects.
Demosaicing: Converts the Bayer pattern to RGB image.
White Balance: Adjusts colors based on lighting conditions.
Color Correction: Maps sensor colors to standard color spaces.
Gamma Correction: Adjusts brightness non-linearly.
Tone Mapping: Compresses dynamic range for display.
Sharpening: Enhances image details.
