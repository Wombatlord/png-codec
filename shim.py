#!/usr/bin/env python
from src.png_decoder import PNGDecoder, Transformer
from src.png_encoder import PNGEncoder
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
decoder = PNGDecoder("test_images/tile_normals.png")
PNGDecoder.inflate_IDAT_data(buf, decoder.idat_chunk)



filtered = buf.read()
buf.seek(0)
recon = Transformer.reconstruct(bytearray(buf.read()), decoder.ihdr.dimensions[0] * 4, decoder.png_reconstructor.bytes_per_pixel)

rgbas = []
for i in range(0, len(recon), 4):
    rgba = recon[i:i+4]
    rgba=tuple(rgba)
    rgbas.append(rgba)

encoder = PNGEncoder(wh=decoder.ihdr.dimensions, raw_source=rgbas)



index = decoder.png_reconstructor.filter_bytes_index
p = Printer()

show_pub_locals(locals())
