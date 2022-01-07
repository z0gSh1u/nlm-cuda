# Calculates numeric metrics between denoised images and grount-truth.

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import os.path as path

NoiseTypes = ['gaussian', 'poisson', 'snp']
DenoiseMethods = ['NLM', 'GaussianLP', 'DnCNN']
Datasets = ['C052_1-150', 'DIV2K_0832', 'lena512']
NoisyPath = r'F:\nlm-cuda\experiment\Noisy'
GTPath = r'F:\nlm-cuda\experiment\GroundTruth'
DenoisedPath = r'F:\nlm-cuda\experiment\Denoised'

if __name__ == '__main__':
    for data in Datasets:
        # Read GT
        GTFile = path.join(GTPath, data + '.bmp')
        GTImage = np.array(Image.open(GTFile))
        for type_ in NoiseTypes:
            for method in DenoiseMethods:
                # Read Denoised
                desc = '[Data] {}; [Noise] {}; [Method] {}'.format(data, type_, method)
                DenoisedFile = path.join(DenoisedPath, method, data + '_' + type_ + '.bmp')
                DenoisedImage = np.array(Image.open(DenoisedFile))
                # Calculate metrics
                PSNR = peak_signal_noise_ratio(GTImage, DenoisedImage)
                if len(GTImage.shape) == 2:
                    SSIM = structural_similarity(GTImage, DenoisedImage)
                else:
                    SSIM = -1 # dont calculate SSIM for color images
                metrics = '>>> [PSNR] {:.2f}; [SSIM] {:.2f}'.format(PSNR, SSIM)
                print(
                    desc,
                    '\n',
                    metrics,
                    '\n'
                )

# [Running Result]
'''
[Data] C052_1-150; [Noise] gaussian; [Method] NLM 
 >>> [PSNR] 26.47; [SSIM] 0.56 

[Data] C052_1-150; [Noise] gaussian; [Method] GaussianLP 
 >>> [PSNR] 23.90; [SSIM] 0.47 

[Data] C052_1-150; [Noise] gaussian; [Method] DnCNN 
 >>> [PSNR] 24.98; [SSIM] 0.50 

[Data] C052_1-150; [Noise] poisson; [Method] NLM 
 >>> [PSNR] 29.82; [SSIM] 0.75 

[Data] C052_1-150; [Noise] poisson; [Method] GaussianLP 
 >>> [PSNR] 27.15; [SSIM] 0.66 

[Data] C052_1-150; [Noise] poisson; [Method] DnCNN 
 >>> [PSNR] 30.58; [SSIM] 0.79 

[Data] C052_1-150; [Noise] snp; [Method] NLM 
 >>> [PSNR] 19.56; [SSIM] 0.39 

[Data] C052_1-150; [Noise] snp; [Method] GaussianLP 
 >>> [PSNR] 24.36; [SSIM] 0.48 

[Data] C052_1-150; [Noise] snp; [Method] DnCNN 
 >>> [PSNR] 18.86; [SSIM] 0.40 

[Data] DIV2K_0832; [Noise] gaussian; [Method] NLM 
 >>> [PSNR] 27.79; [SSIM] -1.00 

[Data] DIV2K_0832; [Noise] gaussian; [Method] GaussianLP 
 >>> [PSNR] 26.45; [SSIM] -1.00 

[Data] DIV2K_0832; [Noise] gaussian; [Method] DnCNN 
 >>> [PSNR] 28.81; [SSIM] -1.00 

[Data] DIV2K_0832; [Noise] poisson; [Method] NLM 
 >>> [PSNR] 31.81; [SSIM] -1.00 

[Data] DIV2K_0832; [Noise] poisson; [Method] GaussianLP 
 >>> [PSNR] 28.82; [SSIM] -1.00 

[Data] DIV2K_0832; [Noise] poisson; [Method] DnCNN 
 >>> [PSNR] 32.80; [SSIM] -1.00 

[Data] DIV2K_0832; [Noise] snp; [Method] NLM 
 >>> [PSNR] 19.61; [SSIM] -1.00 

[Data] DIV2K_0832; [Noise] snp; [Method] GaussianLP 
 >>> [PSNR] 26.70; [SSIM] -1.00 

[Data] DIV2K_0832; [Noise] snp; [Method] DnCNN 
 >>> [PSNR] 28.54; [SSIM] -1.00 

[Data] lena512; [Noise] gaussian; [Method] NLM 
 >>> [PSNR] 26.32; [SSIM] 0.74 

[Data] lena512; [Noise] gaussian; [Method] GaussianLP 
 >>> [PSNR] 29.02; [SSIM] 0.76 

[Data] lena512; [Noise] gaussian; [Method] DnCNN 
 >>> [PSNR] 32.25; [SSIM] 0.87 

[Data] lena512; [Noise] poisson; [Method] NLM 
 >>> [PSNR] 23.42; [SSIM] 0.84 

[Data] lena512; [Noise] poisson; [Method] GaussianLP 
 >>> [PSNR] 30.68; [SSIM] 0.85 

[Data] lena512; [Noise] poisson; [Method] DnCNN 
 >>> [PSNR] 35.77; [SSIM] 0.92 

[Data] lena512; [Noise] snp; [Method] NLM 
 >>> [PSNR] 23.31; [SSIM] 0.55 

[Data] lena512; [Noise] snp; [Method] GaussianLP 
 >>> [PSNR] 27.95; [SSIM] 0.72 

[Data] lena512; [Noise] snp; [Method] DnCNN 
 >>> [PSNR] 20.81; [SSIM] 0.42
'''