from typing import Callable, Generator
from src.filters import Filters
from src.square import Square

class I8(int):
    def __new__(cls, val):
        i = int.__new__(cls, val) & 0xFF
        if i > 127:
            i = (i & 0b01111111) - 128
        return i

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
    
    def __init__(self, wh: tuple[int, int], raw_source) -> None:
        self.raw_source = [*raw_source]
        self.width = wh[0] # width of the image data in pixels, ie tuples of ints
        self.height = wh[1]
        self.stride = self.width * 4 # stride is width in terms of bytes per pixel

    def best_filters(self):
        scores = self.filter_scores()
        filter_bytes = []
        for score in scores:
            filter_bytes.append(score.index(min(score)))
        
        return filter_bytes

    def filter_scores(self) -> list[tuple[int,int,int,int,int]]:
        return [
            *zip
            (
                self.calculate_filter_score(Filters.none_filter, 0),
                self.calculate_filter_score(Filters.sub_filter, 1),
                self.calculate_filter_score(Filters.up_filter, 2),
                self.calculate_filter_score(Filters.average_filter, 3),
                self.calculate_filter_score(Filters.paeth_filter, 4),
            )
        ]

    def calculate_filter_score(self, filter_func: Callable[[Square], int], filter_byte:int) -> list:
        b = bytearray()
        for line in consume_lines(self.width, self.stride, self.raw_source, filter_func, filter_byte):
            b.extend(line)
        return self.minumum_sum_of_absolute_differences(b, self.stride)
    
    def minumum_sum_of_absolute_differences(self, filtered_data, stride) -> list:
        line_scores = []
        filter_stride = stride + 1
        
        for line in range(0, len(filtered_data), filter_stride):
            score = 0
            for i, b in enumerate(filtered_data[line:line+filter_stride]):
                if i % (filter_stride) == 0:
                    continue
                score += abs(I8(b))
                print(f"{b=}")
            
            line_scores.append(score)
        
        return line_scores