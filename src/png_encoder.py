from typing import Generator
import zlib
from src.chunks import IHDR, Chunk, IHDRData
from src.filters import Filters
from src.png_decoder import Transformer
from src.square import Square


def gen_lines(width, source_data) -> Generator[bytearray, None, None]:
    line_bytes = bytearray()
    for i in range(0, len(source_data), width):
        line = source_data[i:i+width]
        for rgba in line:
            line_bytes.extend(rgba)
        
        yield line_bytes
        
        line_bytes.clear()
        
def gen_line_pairs(width, stride, source_data) -> Generator[tuple[bytearray, bytearray], None, None]:
    line_buf = (bytearray(b'\x00'*stride),)
    for line in gen_lines(width, source_data):
        line_buf += (line,)
        if line_buf[2:]:
            line_buf = line_buf[1:]
        yield line_buf
        
def consume_lines(width, stride, source_data, filter_func, filter_byte):
    
    for prev, current in gen_line_pairs(width, stride, source_data):
        scan_src = prev + current
        
        filtered_data = bytearray(b'\x00') + bytearray(stride)
        filtered_data += bytearray([filter_byte])
        while next_square := Square.next_filter_square(scan_src, filtered_data, stride, 4):
            filtered_data.append(filter_func(next_square) & 0xFF)
        filtered_scanline = filtered_data[stride+1:]
        
        yield filtered_scanline


class PNGEncoder:
    raw_source: list[tuple[int, int, int, int]]
    PNG_SIGNATURE = bytes.fromhex("89504E470D0A1A0A")
    
    def __init__(self, wh: tuple[int, int], raw_source) -> None:
        self.raw_source = [*raw_source]
        self.width = wh[0] # width of the image data in pixels, ie tuples of ints
        self.height = wh[1]
        self.bytes_per_pixel = 4
        self.stride = self.width * self.bytes_per_pixel # stride is width in terms of bytes per pixel

    def _source_to_byte_array(self) -> bytearray:
        arr = bytearray()
        for t in self.raw_source:
            arr.extend(t)
        
        return arr
    
    def prepare_ihdr(self) -> Chunk:
        bit_depth = 8
        colour_type = 6
        compression_method = 0
        filter_method = 0
        interlace_method = 0
        ihdr_data = IHDRData(
            self.width,
            self.height,
            bit_depth,
            colour_type,
            compression_method,
            filter_method,
            interlace_method,
        )
        
        ihdr = IHDR(
            length=13,
            chunk_type=b'IHDR',
            chunk_data=ihdr_data,
        )
        
        chunked = Chunk(ihdr.length, ihdr.chunk_type, bytes(ihdr_data), Chunk.calc_crc(bytes(ihdr_data), b'IHDR'))
        return chunked
    
    def apply_filtering(self) -> bytearray:
        source_bytes = self._source_to_byte_array()
        filter_bytes = self._best_filters()
        filtered_data = Transformer.filter(source_bytes, filter_bytes, self.stride, self.bytes_per_pixel)
        return filtered_data

    def _compress_to_idat_chunks(self, filtered_data) -> list[Chunk]:
        arr = bytearray()
        arr.extend(zlib.compress(filtered_data))
            
        max_size = 30
        
        chunks = []
        for i in range(0, len(arr), max_size):
            if len(arr) > max_size:
                chunks.append(
                    Chunk(
                        length=max_size,
                        chunk_type=b'IDAT',
                        chunk_data=arr[i:i+max_size],
                        crc=Chunk.calc_crc(arr[i:i+max_size],b'IDAT')
                    )
                )
            else:
                chunks.append(
                    Chunk(
                        length=len(arr[i:-1]),
                        chunk_type=b'IDAT',
                        chunk_data=arr[i:-1],
                        crc=Chunk.calc_crc(arr[i:-1], b'IDAT')
                    )
                )                
                break
        
        return chunks
    
    def iend_chunk(self) -> Chunk:
        return Chunk(length=0, chunk_type=b'IEND', chunk_data=b'', crc=Chunk.calc_crc(b'', b'IEND'))
    
    def final_datastream(self, filtered_data) -> bytes:
        chunks = [
            self.prepare_ihdr(),
            *self._compress_to_idat_chunks(filtered_data),
            self.iend_chunk()
        ]
        
        d = b''.join([bytes(chunk) for chunk in chunks])
        d = self.PNG_SIGNATURE + d
        
        return d
    
    @staticmethod
    def to_file(fp, final_datastream):
        with open(fp, "wb") as file:
            file.write(final_datastream)
    
    def _best_filters(self) -> list[int]:
        scores = self._filter_scores()
        filter_bytes = []
        for score in scores:
            filter_bytes.append(score.index(min(score)))
        
        return filter_bytes

    def _filter_scores(self) -> list[tuple[int,int,int,int,int]]:
        return [
            *zip(self._calculate_filter_score(i) for i in range(5))
        ]

    def _calculate_filter_score(self, filter_byte:int) -> list[int]:
        b = bytearray()
        for line in consume_lines(self.width, self.stride, self.raw_source, Filters.select_filter_func(filter_byte), filter_byte):
            b.extend(line)
        return Filters.minumum_sum_of_absolute_differences(b, self.stride)
