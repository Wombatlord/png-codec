from __future__ import annotations
from functools import cached_property
from io import BytesIO
from pathlib import Path
import struct
import zlib
from itertools import count
from src.filters import Filters
from src.square import Square
from src.chunks import IHDR, Chunk


# https://pyokagan.name/blog/2019-10-14-png/
# https://www.w3.org/TR/png-3/#5Chunk-layout

class Transformer:
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
                filt_x = int_byte
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
    def filter(source_data: bytearray, filter_bytes: list[int], stride: int, bytes_per_pixel) -> bytearray:
        filter_byte_gen = (filter_bytes[i % len(filter_bytes)] for i in count())
        filter_byte = next(filter_byte_gen)
        filter_data = bytearray([filter_byte])
        while square := Square.next_filter_square(source_data, filter_data, stride, bytes_per_pixel):
            filter_func = Filters.select_filter_func(filter_byte)
            filter_data.append(filter_func(square) & 0xFF)
            if (len(filter_data) % (stride + 1)) == 0:
                filter_byte = next(filter_byte_gen)
                filter_data.append(filter_byte)
        
        return filter_data

    @staticmethod
    def reconstruct(filter_data: bytearray, stride: int, bytes_per_pixel) -> bytearray:
        recon_data = bytearray()
        while square := Square.next_recon_square(filter_data, recon_data, stride, bytes_per_pixel):
            if (len(recon_data) % stride) == 0:
                filter_byte_index = (len(recon_data) // stride) * (stride + 1)
                filter_byte = filter_data[filter_byte_index]
                
                reconstruction_func = Filters.select_reconstruction_func(filter_byte)
            
            recon_data.append(reconstruction_func(square) & 0xFF)

        return recon_data

class Services:
    def __init__(self, config) -> None:
        self.config = config
    
    @cached_property
    def data_buffer(self) -> BytesIO:
        return BytesIO()
    
    @cached_property
    def png_decoder(self) -> PNGDecoder:
        return PNGDecoder(self.config["fp"], self.data_buffer)


class PNGDecoder:
    chunks: list[Chunk]
    _ihdr: IHDR
    PNG_SIGNATURE = bytes.fromhex("89504E470D0A1A0A")

    def __init__(self, fp: str | Path, buf=None) -> None:
        self.path = self._parse_fp(fp)
        self.data_buffer = BytesIO()
        self._read_from_file()
        self._ihdr = self._extract_IHDR()
        self._validate_IHDR()
        self.idat_chunk_idx: int | None = None
        self.chunks = self._chunker()
        self.png_reconstructor = Transformer(
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
