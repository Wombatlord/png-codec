from src.png_decoder import Filters, Square, Transformer
from ptpython import embed
import pytest

def image_fixture() -> bytearray:
    red = (255,0,0,255)
    green = (0,255,0,255)
    blue = (0,0,255,255)
    image = b"".join(map(bytes, [red, green, blue] * 3))
    
    return bytearray(image)


def test_none_filter():
    filter_byte = 0
    stride = 12
    bytes_per_pixel = 4
    source_data = image_fixture()
    filter_data = Transformer.filter(source_data, [filter_byte], stride, bytes_per_pixel)
  
    filter_byte_arr = bytearray([filter_byte])
    expected_filter_data = filter_byte_arr + source_data[:stride]
    expected_filter_data += filter_byte_arr + source_data[stride: 2*stride]
    expected_filter_data += filter_byte_arr + source_data[2*stride: 3*stride]
    
    diff = [
        expected - actual
        for expected, actual
        in zip(expected_filter_data, filter_data)
    ]
    
    assert all(expected == actual for expected, actual in zip(expected_filter_data, filter_data)), (
        f"{diff=}\n"
        f"ex={expected_filter_data}\n"
        f"ac={filter_data}\n"
    )
    
@pytest.mark.parametrize(["filter_byte"], (
    (0,),
    (1,),
    (2,),
    (3,),
    (4,),
))
def test_recon_is_filter_inverse(filter_byte):
    # Arrange
    stride = 12
    source_data = image_fixture()
    bytes_per_pixel = 4
    # Act
    filter_data = Transformer.filter(source_data, [filter_byte], stride, bytes_per_pixel)
    recon_data = Transformer.reconstruct(filter_data, stride, 4)

    # Assert
    diff = [
        expected - actual
        for expected, actual
        in zip(source_data, recon_data)
    ]
    
    assert all(expected == actual for expected, actual in zip(source_data, recon_data)), (
        f"{diff=}\n"
        f"ex={source_data}\n"
        f"fi={filter_data}\n"
        f"ac={recon_data}\n"
    )

@pytest.mark.parametrize(["filter_byte_pattern"], (
    ([0,1,2,3,4],),
    ([0,1,2,2,2,0,1,4,2,0,0,0,1,1,2],),
))
def test_recon_is_filter_inverse_varying_filters(filter_byte_pattern):
    # Arrange
    stride = 12
    bytes_per_pixel = 4
    source_data = image_fixture()
  
    # Act
    filter_data = Transformer.filter(source_data, filter_byte_pattern, stride, bytes_per_pixel)
    recon_data = Transformer.reconstruct(filter_data, stride, 4)

    # Assert
    diff = [
        expected - actual
        for expected, actual
        in zip(source_data, recon_data)
    ]
    
    assert all(expected == actual for expected, actual in zip(source_data, recon_data)), (
        f"{diff=}\n"
        f"ex={source_data}\n"
        f"fi={filter_data}\n"
        f"ac={recon_data}\n"
    )

    
def test_filter_next_square():
    # Arrange
    stride = 12
    filter_byte = 1
    source_data = image_fixture()
    filter_data = bytearray([filter_byte])
    bytes_per_pixel = 4
    # Act
    square = Square.next_filter_square(source_data, filter_data, stride, bytes_per_pixel)
    filter_output = Filters.select_filter_func(filter_byte)(square)
    
    # Assert
    assert square == Square(x=source_data[0], a=0, b=0, c=0)
    assert filter_output == source_data[0]
    
def test_filter_next_next_square():
    # Arrange
    stride = 12
    filter_byte = 1
    source_data = image_fixture()
    filter_data = bytearray([filter_byte, source_data[0]])
    bytes_per_pixel = 4
    # Act
    square = Square.next_filter_square(source_data, filter_data, stride, bytes_per_pixel)
    filter_output = Filters.select_filter_func(filter_byte)(square)
    
    # Assert
    assert square == Square(x=source_data[1], a=source_data[0], b=0, c=0)
    assert filter_output == source_data[1] - source_data[0]
    
def test_filter_next_next_next_square():
    # Arrange
    stride = 12
    filter_byte = 1
    source_data = image_fixture()
    filter_data = bytearray([filter_byte, source_data[0], (source_data[1] - source_data[0]) & 0xFF])
    bytes_per_pixel = 4
    # Act
    square = Square.next_filter_square(source_data, filter_data, stride, bytes_per_pixel)
    filter_output = Filters.select_filter_func(filter_byte)(square)
    
    # Assert
    assert square == Square(x=source_data[2], a=source_data[1], b=0, c=0)
    assert filter_output == source_data[2] - source_data[1]