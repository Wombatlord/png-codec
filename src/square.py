from typing import NamedTuple, Self, Generator

class Square(NamedTuple):
    x: int
    a: int
    b: int
    c: int
    
    @classmethod
    def sample_x(cls, filt_x: int, x_idx: int, current_scanline: bytearray, previous_scanline: bytearray) -> Self:
        return cls(
            x=filt_x,
            a=current_scanline[x_idx],
            b=previous_scanline[x_idx+1],
            c=previous_scanline[x_idx],
        )
        
    @classmethod
    def next_recon_square(cls, filter_data: bytearray, recon_data: bytearray, stride: int, bytes_per_pixel: int) -> Self | None:
        scan_offset = len(recon_data) % stride
        line_offset = len(recon_data) // stride
        
        scan_incr = bytes_per_pixel
        line_incr = 1
        
        x_offsets = (scan_offset, line_offset)
        a_offsets = (scan_offset - scan_incr, line_offset)
        b_offsets = (scan_offset, line_offset - line_incr)
        c_offsets = (scan_offset - scan_incr, line_offset - line_incr)
        
        filter_stride = stride + 1
        
        #                       go down by line offset scans          skip filter byte
        #                                             |----------|    v
        filter_byte_index = lambda scan, line: line * filter_stride + 1 + scan
        recon_byte_index = lambda scan, line: line * stride + scan
        
        if filter_byte_index(*x_offsets) >= len(filter_data):
            return None
        
        x = filter_data[filter_byte_index(*x_offsets)]
        
        a = 0
        rec_idx_a = recon_byte_index(*a_offsets)
        if a_offsets[0] >= 0:
            a = recon_data[rec_idx_a]
        
        b = 0
        rec_idx_b = recon_byte_index(*b_offsets)
        if b_offsets[1] >= 0:
            b = recon_data[rec_idx_b]
        
        c = 0
        rec_idx_c = recon_byte_index(*c_offsets)
        if c_offsets[0] >= 0 and c_offsets[1] >= 0:
            c = recon_data[rec_idx_c]
            
        return cls(x, a, b, c)

    @classmethod
    def next_filter_square(cls, source_data: bytearray, filtered_data: bytearray, stride: int, bytes_per_pixel: int) -> Self | None:
        source_byte_index = lambda scan, line: line * stride + scan
        
        scan_offset, line_offset = cls.filtered_data_offsets(filtered_data, stride)
        scan_incr = bytes_per_pixel
        x_offsets = (scan_offset, line_offset)
        a_offsets = (scan_offset - scan_incr, line_offset)
        b_offsets = (scan_offset, line_offset - scan_incr)
        c_offsets = (scan_offset - scan_incr, line_offset - scan_incr)

        if source_byte_index(*x_offsets) >= len(source_data):
            return None
        
        x = source_data[source_byte_index(*x_offsets)]
        
        a = 0
        if a_offsets[0] >= 0:
            a = source_data[source_byte_index(*a_offsets)]
        
        b = 0
        if b_offsets[1] >= 0:
            b = source_data[source_byte_index(*b_offsets)]
        
        c = 0
        if c_offsets[0] >= 0 and c_offsets[1] >= 0:
            c = source_data[source_byte_index(*c_offsets)]
            
        return cls(x, a, b, c)
        
    @classmethod
    def filtered_data_offsets(cls, filtered_data: bytearray, stride: int) -> tuple[int, int]:
        filter_stride = stride + 1
        scan_offset = max((len(filtered_data) % filter_stride) - 1, 0)
        line_offset = len(filtered_data) // filter_stride
        return scan_offset, line_offset
    
    @classmethod
    def is_scan_start(cls, filtered_data: bytearray, stride: int) -> bool:
        return cls.filtered_data_offsets(filtered_data, stride)[0] == 0

    @classmethod
    def iter_source_scanline(cls, source_data: bytearray, filtered_data: bytearray, stride: int) -> Generator[Self, None, None]:
        scan_offset, line_offset = cls.filtered_data_offsets(filtered_data, stride)
        if not cls.is_scan_start(filtered_data, stride):
            raise ValueError(f"We are trying to yield a scanline but the data has lenth implying a non-zero scan offset of {scan_offset}")
        
        yield cls.next_filter_square(source_data, filtered_data, stride)
        
        while not cls.is_scan_start(filtered_data, stride):
            yield cls.next_filter_square(source_data, filtered_data, stride)
            