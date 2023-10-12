from src.square import Square

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
