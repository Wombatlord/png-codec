from __future__ import annotations
from io import BytesIO
from pathlib import Path
import struct
from typing import NamedTuple, Self
import zlib


class IHDR(NamedTuple):
    length: int
    chunk_type: bytes
    chunk_data: bytes

    @property
    def ihdr_data(self) -> IHDRData:
        return IHDRData.from_bytes(self.chunk_data)


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


class Chunk:
    def __init__(self, length, chunk_type, chunk_data, crc) -> None:
        self.length = length
        self.chunk_type = chunk_type
        self.chunk_data = chunk_data
        self.crc = crc

    def combine_chunks(self, chunk_b: Self):
        self.chunk_data += chunk_b.chunk_data
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


class Filters:
    def __init__(self, width: int) -> None:
        self.bytes_per_pixel = 4
        self.stride = width * self.bytes_per_pixel

    @staticmethod
    def none_filter(square: Square, *_) -> int:
        return square.x

    @staticmethod
    def sub_filter(square: Square, inverse=False) -> int:
        match inverse:
            case False:
                return square.x - square.a
            case True:
                return square.x + square.a

    @staticmethod
    def up_filter(square: Square, inverse=False) -> int:
        match inverse:
            case False:
                return square.x - square.b
            case True:
                return square.x + square.b

    @staticmethod
    def average_filter(square: Square, inverse=False) -> int:
        match inverse:
            case False:
                return square.x - (square.a + square.b) // 2
            case True:
                return square.x + (square.a + square.b) // 2

    @staticmethod
    def paeth_filter(square: Square, inverse=False) -> int:
        match inverse:
            case False:
                return square.x - Filters.paeth_predictor(square.a, square.b, square.c)
            case True:
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


class Reconstructor:
    def __init__(self, width, height) -> None:
        self.bytes_per_pixel = 4
        self.height = height
        self.stride = width * self.bytes_per_pixel
        self.reconstructed = bytearray()
        self._reconstruction = True
        self.reconstruction_funcs = [
            Filters.none_filter,
            Filters.sub_filter,
            Filters.up_filter,
            Filters.average_filter,
            Filters.paeth_filter,
        ]

    def reconstruct(self, buf: BytesIO):
        last_line = bytearray(b"\x00" * (self.stride + 1))
        # Loop over scanlines
        for h in range(self.height):
            filt_line = buf.read(self.stride + 1)
            filter_byte, filt_scan = filt_line[0], filt_line[1:]

            # Loop over pixels in a scanline
            recon_line = bytearray(b"\x00")
            for i, int_byte in enumerate(filt_scan):
                line_idx = i + 1
                filt_x = int_byte
                recon_a = recon_line[line_idx - 1]
                recon_b = last_line[line_idx]
                recon_c = last_line[line_idx - 1]
                square = Square(filt_x, recon_a, recon_b, recon_c)
                match filter_byte:
                    case 0:
                        reconstruction_func = self.reconstruction_funcs[0]

                    case 1:
                        reconstruction_func = self.reconstruction_funcs[1]

                    case 2:
                        reconstruction_func = self.reconstruction_funcs[2]

                    case 3:
                        reconstruction_func = self.reconstruction_funcs[3]

                    case 4:
                        reconstruction_func = self.reconstruction_funcs[4]

                    case _:
                        raise ValueError(f"Unknown filter type: {filter_byte}")

                recon_x = reconstruction_func(square, self._reconstruction)
                recon_line.append(recon_x & 0xFF)

            recon_scan = recon_line[1:]
            self.reconstructed.extend(recon_scan)
            last_line = recon_line

        return self.reconstructed


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
        ihdr_bytes = self.data_buffer.read(23)
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
        return self.png_reconstructor.reconstruct(buf)


p = PNGDecoder("./assets/sprites/Isometric_MedievalFantasy_Tiles.png")
