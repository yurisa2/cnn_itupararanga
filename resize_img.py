import os
from PIL import Image
import numpy as np

os.chdir("/home/yurisa2/lampstack-7.3.7-1/apache2/htdocs/rPDI/artigo/sat")


img_files = os.listdir("rgb_files/")

widths = np.empty(1).astype("uint8")
heights = np.empty(1).astype("uint8")

widths.min()
heights.min()

for i in img_files:
    im = Image.open("rgb_files/" + i)
    widths = np.append(widths,im.size[0])
    heights = np.append(heights,im.size[1])

widths = np.delete(widths,0)
heights = np.delete(heights,0)

size = min(widths), min(heights)


for i in img_files:
    im = Image.open("rgb_files/" + i)
    im = im.resize(size, Image.NEAREST)
    im.save('rgb_files_resized/' + i, "JPEG")
