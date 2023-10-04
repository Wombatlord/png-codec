#!/usr/bin/env python
from src.png_decoder import PNGDecoder, Reconstructor
from src.printer import Printer
from io import BytesIO
from PIL import Image

def show_pub_locals(locals_):
    for name, var in {**locals_}.items():
        if not name.startswith("_"):
            svar = f"{var}"
            if len(svar) > 80:
                svar = svar[:80] + "..."
            print(f"{name} = {svar}")

buf = BytesIO()
png = PNGDecoder("tile_normals.png")
PNGDecoder.inflate_IDAT_data(buf, png.idat_chunk)

filtered = buf.read()
buf.seek(0)
recon = Reconstructor.reconstruct(bytearray(buf.read()), png.ihdr.dimensions[0] * 4, png.png_reconstructor.bytes_per_pixel)

# example_recon = png.example_recon(bytearray(filtered))

index = png.png_reconstructor.filter_bytes_index
p = Printer()

show_pub_locals(locals())

import numpy as np
import matplotlib.pyplot as plt
# plt.imshow(np.array(example_recon).reshape((png.ihdr.dimensions[1], png.ihdr.dimensions[0], 4)))
# plt.show()