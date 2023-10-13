from __future__ import annotations
from typing import NamedTuple, Self
import struct
import zlib


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

    def __bytes__(self) -> bytes:
        l = struct.pack(">i", self.length)
        ct = struct.pack(">4s", self.chunk_type)
        crc = struct.pack(">I", self.crc)
        b = l + ct + self.chunk_data + crc
        return b
        
    def combine_chunks(self, chunk_b: Self):
        self.chunk_data = b''.join([self.chunk_data, chunk_b.chunk_data])
        self.length += chunk_b.length
        self.crc = zlib.crc32(
            self.chunk_data, zlib.crc32(struct.pack(">4s", self.chunk_type))
        )

    @staticmethod
    def calc_crc(chunk_data, chunk_type) -> int:
        return zlib.crc32(
            chunk_data, zlib.crc32(struct.pack(">4s", chunk_type))
        )
    
    def __add__(self, other):
        self.combine_chunks(other)