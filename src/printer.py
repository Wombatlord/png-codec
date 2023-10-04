from typing import Generator
from itertools import count


PixelStream = Generator[tuple[int,int,int,int], None, None]
ScanLineStream = Generator[PixelStream, None, None]

class Printer:
    ESC = "\x1B"
    CSI = f"{ESC}["

    @classmethod
    def paint(cls, s: str, r: int, g: int, b: int) -> str:
        return "".join(
            [
                f"{cls.CSI}38;2;{r};{g};{b}m",
                s,
                f"{cls.CSI}0m",
            ]
        )

    def __init__(self, px_bytes: int = 4):
        self.px_bytes = px_bytes

    def apply_rgb(self, s: str, rgba: tuple[int, int, int, int]):
        *rgb, a = tuple(rgba)
        if a < 200:
            return " " * 2
        else:
            return self.paint(s, *rgb)
        
    @staticmethod
    def pop_rgba(data: bytearray) -> PixelStream:
        buf = []
        while data:
            buf.append(data.pop(0))
            if len(buf) >= 4:
                rgba = tuple(buf)
                buf.clear()
                yield rgba
    
    def enumerate_rows(self, data: bytearray, dimensions: tuple[int, int]) -> ScanLineStream:
        w, _ = dimensions
        scan_len = self.px_bytes * w
        remaining_data = data.copy()
        for i in count():
            if not remaining_data:
                break
            
            scan_up_to = min(scan_len, len(remaining_data))
            output, remaining_data = remaining_data[:scan_up_to], remaining_data[scan_up_to:]
            
            yield (i, Printer.pop_rgba(output))
            
    def print(self, image: bytearray, dims: tuple[int, int]):
        # This prints column numbers
        print("\t" + "".join(f"{x:>#2d}" for x in range(dims[0])))
        
        for i, row in self.enumerate_rows(image, dims):
            print(f"{i=}", end="\t")
            for rgba in row:
                print(self.apply_rgb("██", rgba), end="")
            print("")