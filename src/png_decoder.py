from __future__ import annotations
from io import BytesIO
from pathlib import Path
import struct
from typing import NamedTuple, Self
import zlib
from itertools import count



class IHDR(NamedTuple):
    length: int
    chunk_type: bytes
    chunk_data: bytes

    @property
    def ihdr_data(self) -> IHDRData:
        return IHDRData.from_bytes(self.chunk_data)
    
    @property
    def dimensions(self) -> tuple[int, int]:
        return self.ihdr_data.dimensions


class IHDRData(NamedTuple):
    width: int
    height: int
    bit_depth: int
    colour_type: int
    compression_method: int
    filter_method: int
    interlace_method: int

    def __bytes__(self) -> bytes:
        return struct.pack(">IIBBBBB", *self)

    @classmethod
    def from_bytes(cls, data: bytes):
        return cls(*struct.unpack(">IIBBBBB", data))
    
    @property
    def dimensions(self) -> tuple[int, int]:
        return self.width, self.height


class Chunk:
    length: int
    chunk_type: str
    chunk_data: bytes
    crc:int
    def __init__(self, length: int, chunk_type: str, chunk_data: bytes, crc: int) -> None:
        self.length = length
        self.chunk_type = chunk_type
        self.chunk_data = chunk_data
        self.crc = crc

    def combine_chunks(self, chunk_b: Self):
        self.chunk_data = b''.join([self.chunk_data, chunk_b.chunk_data])
        self.length += chunk_b.length
        self.crc = zlib.crc32(
            self.chunk_data, zlib.crc32(struct.pack(">4s", self.chunk_type))
        )

    def __add__(self, other):
        self.combine_chunks(other)


# https://pyokagan.name/blog/2019-10-14-png/
# https://www.w3.org/TR/png-3/#5Chunk-layout


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
    def recon_next_square(cls, filter_data: bytearray, recon_data: bytearray, stride: int, bytes_per_pixel: int) -> Self | None:
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
    def filter_next_square(cls, source_data: bytearray, filtered_data: bytearray, stride: int) -> Self | None:
        filter_stride = stride + 1
        filter_byte_index = lambda scan, line: line * filter_stride + 1 + scan
        source_byte_index = lambda scan, line: line * stride + scan
        
        scan_offset = max((len(filtered_data) % filter_stride) - 1, 0)
        line_offset = len(filtered_data) // filter_stride
        
        x_offsets = (scan_offset, line_offset)
        a_offsets = (scan_offset - 1, line_offset)
        b_offsets = (scan_offset, line_offset - 1)
        c_offsets = (scan_offset - 1, line_offset - 1)

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
        
class Filters:
    def __init__(self, width: int) -> None:
        self.bytes_per_pixel = 4
        self.stride = width * self.bytes_per_pixel

    @staticmethod
    def none_filter(square: Square) -> int:
        return square.x

    @staticmethod
    def none_recon(square: Square) -> int:
        return square.x
    
    @staticmethod
    def sub_filter(square: Square) -> int:
        return square.x - square.a

    @staticmethod
    def sub_recon(square: Square) -> int:
        return square.x + square.a
    
    @staticmethod
    def up_filter(square: Square) -> int:
        return square.x - square.b

    @staticmethod
    def up_recon(square: Square) -> int:
        return square.x + square.b
    
    @staticmethod
    def average_filter(square: Square) -> int:
        return square.x - (square.a + square.b) // 2
    
    @staticmethod
    def average_recon(square: Square) -> int:
        return square.x + (square.a + square.b) // 2

    @staticmethod
    def paeth_filter(square: Square) -> int:
        return square.x - Filters.paeth_predictor(square.a, square.b, square.c)
    
    @staticmethod
    def paeth_recon(square: Square) -> int:
        return square.x + Filters.paeth_predictor(square.a, square.b, square.c)
    
    @staticmethod
    def paeth_predictor(a, b, c):
        p = a + b - c
        pa = abs(p - a)
        pb = abs(p - b)
        pc = abs(p - c)
        if pa <= pb and pa <= pc:
            Pr = a
        elif pb <= pc:
            Pr = b
        else:
            Pr = c
        return Pr

    @staticmethod
    def select_filter_func(filter_byte):
        return [
            Filters.none_filter,
            Filters.sub_filter,
            Filters.up_filter,
            Filters.average_filter,
            Filters.paeth_filter,
        ][filter_byte]

    @staticmethod
    def select_reconstruction_func(filter_byte):
        return [
            Filters.none_recon,
            Filters.sub_recon,
            Filters.up_recon,
            Filters.average_recon,
            Filters.paeth_recon,
        ][filter_byte]


class Reconstructor:
    def __init__(self, width, height) -> None:
        self.bytes_per_pixel = 4
        self.height = height
        self.stride = width * self.bytes_per_pixel
        self.filter_bytes_index = []

    def reconstruct_buf(self, buf: BytesIO):
        prev_recon_line = bytearray(b"\x00" * (self.stride + 1))
        reconstructed = bytearray()
        # Loop over scanlines
        for h in range(self.height):
            filt_line = buf.read(self.stride + 1)
            filter_byte, filt_scan = filt_line[0], filt_line[1:]
            self.filter_bytes_index.append(filter_byte)
            # Loop over pixels in a scanline
            recon_line = bytearray(b"\x00")
            for i, int_byte in enumerate(filt_scan):
                # line_idx = i + 1
                filt_x = int_byte
                # recon_a = recon_line[line_idx - 1]
                # recon_b = prev_recon_line[line_idx]
                # recon_c = prev_recon_line[line_idx - 1]
                # square = Square(filt_x, recon_a, recon_b, recon_c)
                square = Square.sample_x(filt_x, i, recon_line, prev_recon_line)
                
                if filter_byte not in range(len(self.reconstruction_funcs)):
                    raise ValueError(f"Unknown filter type: {filter_byte}")
                
                reconstruction_func = Filters.select_reconstruction_func[filter_byte]

                recon_x = reconstruction_func(square)
                
                assert recon_x >= 0, f"Detected negative reconstructed byte value: {recon_x=}, y={h}, x={i}"
                
                recon_line.append(recon_x & 0xFF)

            recon_scan = recon_line[1:]
            reconstructed.extend(recon_scan)
            prev_recon_line = recon_line

        return reconstructed
    
    @staticmethod
    def filter(source_data: bytearray, filter_bytes: list[int], stride: int) -> bytearray:
        filter_byte_gen = (filter_bytes[i % len(filter_bytes)] for i in count())
        filter_byte = next(filter_byte_gen)
        filter_data = bytearray([filter_byte])
        while square := Square.filter_next_square(source_data, filter_data, stride):
            filter_func = Filters.select_filter_func(filter_byte)
            filter_data.append(filter_func(square) & 0xFF)
            if (len(filter_data) % (stride + 1)) == 0:
                filter_byte = next(filter_byte_gen)
                filter_data.append(filter_byte)
        
        return filter_data

    @staticmethod
    def reconstruct(filter_data: bytearray, stride: int, bytes_per_pixel) -> bytearray:
        recon_data = bytearray()
        while square := Square.recon_next_square(filter_data, recon_data, stride, bytes_per_pixel):
            if (len(recon_data) % stride) == 0:
                filter_byte_index = (len(recon_data) // stride) * (stride + 1)
                filter_byte = filter_data[filter_byte_index]
                
                reconstruction_func = Filters.select_reconstruction_func(filter_byte)
            
            recon_data.append(reconstruction_func(square) & 0xFF)

        return recon_data


class PNGDecoder:
    chunks: list[Chunk]
    _ihdr: IHDR
    PNG_SIGNATURE = bytes.fromhex("89504E470D0A1A0A")

    def __init__(self, fp: str | Path) -> None:
        self.path = self._parse_fp(fp)
        self.data_buffer = BytesIO()
        self._read_from_file()
        self._ihdr = self._extract_IHDR()
        self._validate_IHDR()
        self.idat_chunk_idx: int | None = None
        self.chunks = self._chunker()
        self.png_reconstructor = Reconstructor(
            self.ihdr.ihdr_data.width, self.ihdr.ihdr_data.height
        )
        self.idat = self.chunks[2]

    @property
    def ihdr(self) -> IHDR:
        return self._ihdr

    @property
    def idat_chunk(self) -> Chunk:
        return self.chunks[self.idat_chunk_idx]

    def _parse_fp(self, fp: str | Path) -> Path:
        if not isinstance(fp, (str, Path)):
            raise ValueError(
                "Invalid path type. Path should be a string or Pathlib.Path"
            )
        if isinstance(fp, Path):
            return fp
        else:
            return Path(fp)

    def _read_from_file(self):
        """
        Checks the first 8 bytes of the file match the expected PNG signature bytes.
        Reads a compressed PNG datastream into the data_buffer attribute before closing the file.
        The PNG Signature bytes are not read into the buffer as they are not required after this validation check.

        Raises:
            ValueError: Raised if the first 8 bytes do not match the expected PNG signature bytes.
        """
        if not self.path.exists():
            raise ValueError("Path does not exist.")

        with self.path.open("rb") as file:
            signature = file.read(8)
            if signature == self.PNG_SIGNATURE:
                print("That there's a PNG.")
                self._write_data_to_data_buffer(self.data_buffer, file.read())

            else:
                raise ValueError("That's not a PNG chief.")

    def _extract_IHDR(self) -> IHDR:
        """
        Reads the first 23 bytes from the data_buffer which are the IHDR chunk bytes.
        The chunk type and data remain as raw bytes, with other fields assigned the int representation of the appropriate byte.
        Moves the buffer back to the start afterwards.

        Returns:
            IHDR: NamedTuple representation of an IHDR chunk
        """
        self.data_buffer.seek(0)
        ihdr_bytes = self.data_buffer.read(21)
        ihdr = IHDR(
            length=int.from_bytes(ihdr_bytes[:4]),
            chunk_type=ihdr_bytes[4:8],
            chunk_data=ihdr_bytes[8:21],
        )
        self.data_buffer.seek(0)
        return ihdr

    def _validate_IHDR(self):
        """
        This decoder is only a partial implementation of the PNG spec.
        Validation ensures we aren't trying to decode PNGs that we don't have the facilites for, big man.

        See IHDR section of PNG Spec for details about these settings.
        Raises:
            ValueError: Compression Method - The only valid value as defined by the PNG spec is 0.
            ValueError: Filter Method - The only valid value as defined by the PNG spec is 0
            ValueError: Colour Type - This decoder only implements support for Truecolour with Alpha, indicated by 6.
            ValueError: Bit Depth - This decoder only supports a bit depth of 8.
            ValueError: Interlace Method - This decoder does not support decoding of interlaced PNG data.
        """
        if self._ihdr.ihdr_data.compression_method != 0:
            raise ValueError(
                f"Invalid compression method: Expected 0, Got: {self._ihdr.ihdr_data.compression_method}"
            )

        if self._ihdr.ihdr_data.filter_method != 0:
            raise ValueError(
                f"Invalid filter method: Expected 0, Got: {self._ihdr.ihdr_data.filter_method}"
            )

        if self._ihdr.ihdr_data.colour_type != 6:
            raise ValueError(
                f"We only support truecolour with alpha. Got: {self._ihdr.ihdr_data.colour_type}"
            )

        if self._ihdr.ihdr_data.bit_depth != 8:
            raise ValueError(
                f"We only support bit depth of 8. Got {self._ihdr.ihdr_data.bit_depth}"
            )

        if self._ihdr.ihdr_data.interlace_method != 0:
            raise ValueError(
                f"We only support no interlacing. Got {self._ihdr.ihdr_data.interlace_method}"
            )

    @staticmethod
    def _write_data_to_data_buffer(buffer: BytesIO, data: bytes):
        buffer.write(data)
        buffer.seek(0)

    def _chunker(self) -> list[Chunk]:
        """
        Reads from the BytesIO buffer in self.data_buffer in order to split a compressed PNG datastream into its composite chunks.
        Before creating Chunk objects, the checksum for the chunk is validated.
        chunk_idx tracks the chunk iterations to make note of the first IDAT chunk index,
        this will ultimately be the index of the singular combined IDAT chunk in the returned chunk array.

        Raises:
            ValueError: Checksum validation failure.
            Exception:  If the while loop completes without returning,
                        we have reached the end of the data without finding the requisite IEND chunk.
                        A databuffer.read operation is likely to error before this in this case,
                        but it is possible for the amount of data to align with the read amounts.

        Returns:
            list[Chunk] with a single combined IDAT chunk indexed at self.idat_chunk_idx
        """
        chunks = []
        chunk_idx = 0
        buffer_length = self.data_buffer.getbuffer().nbytes
        while self.data_buffer.tell() <= buffer_length:
            chunk_length, chunk_type = struct.unpack(">I4s", self.data_buffer.read(8))

            if chunk_length + 4 + self.data_buffer.tell() > buffer_length:
                raise Exception(
                    f"Chunk length + checksum offset exceeds buffer length.{chunk_length=} {self.data_buffer.tell()=} {buffer_length=}"
                )

            chunk_data = self.data_buffer.read(chunk_length)
            expected_crc, *_ = struct.unpack(">I", self.data_buffer.read(4))
            chunk_crc = self._validate_crc(
                chunk_length, chunk_type, chunk_data, expected_crc
            )

            if chunk_type == b"IDAT" and not self.idat_chunk_idx:
                # As IDAT chunks are consecutive, keeping a reference to this index allows us
                # to only iterate over IDAT chunks when combining the chunk data for decompression.
                self.idat_chunk_idx = chunk_idx

            chunks.append(Chunk(chunk_length, chunk_type, chunk_data, chunk_crc))
            chunk_idx += 1

            if chunk_type == b"IEND":
                self.data_buffer.seek(0)
                self._combine_IDAT_data(chunks)
                return chunks

        raise Exception("No IEND Chunk was found but the data was fully read.")

    def _validate_crc(
        self, chunk_length: int, chunk_type: bytes, chunk_data: bytes, expected_crc: int
    ) -> int:
        actual_crc = zlib.crc32(chunk_data, zlib.crc32(struct.pack(">4s", chunk_type)))
        if actual_crc != expected_crc:
            raise ValueError(
                f"Checksum Failed on chunk type {chunk_type} with length {chunk_length}"
            )

        return expected_crc

    def _combine_IDAT_data(self, chunks: list[Chunk]):
        previous_chunk = None
        for chunk in chunks[self.idat_chunk_idx : -1]:
            if previous_chunk:
                previous_chunk.combine_chunks(chunk)
                chunks.remove(chunk)
            previous_chunk = chunk

    @staticmethod
    def inflate_IDAT_data(buf: BytesIO, chunk: Chunk):
        PNGDecoder._write_data_to_data_buffer(buf, zlib.decompress(chunk.chunk_data))

    def reconstruct_from_inflated_data(self, buf):
        return self.png_reconstructor.reconstruct_buf(buf)
    
    
    #######################
    def example_recon(self, filtered):
        Recon = []
        bytesPerPixel = 4
        stride = self.ihdr.dimensions[0] * bytesPerPixel

        def Recon_a(r, c):
            # Height_idx * stride + Scanline_idx - 4
            return Recon[r * stride + c - bytesPerPixel] if c >= bytesPerPixel else 0

        def Recon_b(r, c):
            return Recon[(r-1) * stride + c] if r > 0 else 0

        def Recon_c(r, c):
            return Recon[(r-1) * stride + c - bytesPerPixel] if r > 0 and c >= bytesPerPixel else 0

        i = 0
        for r in range(self.ihdr.dimensions[1]): # for each scanline
            filter_type = filtered[i] # first byte of scanline is filter type
            i += 1
            for c in range(stride): # for each byte in scanline
                Filt_x = filtered[i]
                i += 1
                if filter_type == 0: # None
                    Recon_x = Filt_x
                elif filter_type == 1: # Sub
                    Recon_x = Filt_x + Recon_a(r, c)
                elif filter_type == 2: # Up
                    Recon_x = Filt_x + Recon_b(r, c)
                elif filter_type == 3: # Average
                    Recon_x = Filt_x + (Recon_a(r, c) + Recon_b(r, c)) // 2
                elif filter_type == 4: # Paeth
                    Recon_x = Filt_x + Filters.paeth_predictor(Recon_a(r, c), Recon_b(r, c), Recon_c(r, c))
                else:
                    raise Exception('unknown filter type: ' + str(filter_type))
                Recon.append(Recon_x & 0xff) # truncation to byte
        
        return Recon