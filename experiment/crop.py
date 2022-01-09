imgPaths = [
    'GroundTruth/DIV2K_0832',
    'Noisy/DIV2K_0832_gaussian',
    'Noisy/DIV2K_0832_poisson',
    'Noisy/DIV2K_0832_snp',
    'Denoised/DnCNN/DIV2K_0832_gaussian',
    'Denoised/DnCNN/DIV2K_0832_poisson',
    'Denoised/DnCNN/DIV2K_0832_snp',
    'Denoised/GaussianLP/DIV2K_0832_gaussian',
    'Denoised/GaussianLP/DIV2K_0832_poisson',
    'Denoised/GaussianLP/DIV2K_0832_snp',
    'Denoised/NLM/DIV2K_0832_gaussian',
    'Denoised/NLM/DIV2K_0832_poisson',
    'Denoised/NLM/DIV2K_0832_snp',
]
imgPaths = [x + '.bmp' for x in imgPaths]

CROP_BOX = (1527, 297, 1527 + 512, 297 + 512)

from os import path
from PIL import Image
from tqdm import tqdm
for imgPath in tqdm(imgPaths, ncols=80):
    img = Image.open('F:/nlm-cuda/experiment/' + imgPath)
    img = img.crop(CROP_BOX)
    img.save(path.join('F:/nlm-cuda/experiment/Crop/', imgPath.replace('/', '-')))
