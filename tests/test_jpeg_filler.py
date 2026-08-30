#    Copyright 2021-2023 SECTRA AB
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import threading
from ctypes import c_short, pointer
from io import BytesIO
from typing import Optional

import numpy as np
import pytest
from PIL import Image
from turbojpeg import (
    CUSTOMFILTER,
    TJXOP_NONE,
    TJXOPT_CROP,
    TJXOPT_GRAY,
    TJXOPT_PERFECT,
    BackgroundStruct,
    CroppingRegion,
    TransformStruct,
    fill_background,
)

from opentile.jpeg.jpeg import find_turbojpeg_path
from opentile.jpeg.jpeg_filler import (
    BlankImage,
    BlankStruct,
    JpegFiller,
)

test_file_path = "tests/testdata/turbojpeg/frame_2048x512.jpg"


@pytest.fixture()
def filler():
    yield JpegFiller(find_turbojpeg_path())


@pytest.fixture()
def buffer():
    with open(test_file_path, "rb") as file:
        yield file.read()


SOI = bytes([0xFF, 0xD8])
MARKER = bytes([0xFF, 0xDB])


def dqt_segment(table_index: int, length: int = 67) -> bytes:
    """A quantization table segment for table_index, with 64 8 bit elements."""
    return MARKER + length.to_bytes(2, "big") + bytes([table_index << 4]) + bytes(64)


@pytest.mark.unittest
class TestJpegFiller:
    @pytest.mark.parametrize(
        ["luminance", "expected_value"],
        [
            (0.0, 0),
            (0.5, 128),
            (1.0, 255),
        ],
    )
    def test_fill_image(
        self,
        filler: JpegFiller,
        buffer: bytes,
        luminance: float,
        expected_value: int,
    ):
        # Act
        filled = filler.fill_image(buffer, background_luminance=luminance)

        # Assert
        image = Image.open(BytesIO(filled))
        pixels = np.array(image.convert("L"))
        assert (pixels == expected_value).all()

    def test_fill_background_callback(self):
        # Arrange
        mcu_size = 64
        original_width = 8
        original_height = 8
        extended_width = 16
        extended_height = 16
        callback_row_height = 8
        background_luminance = 255
        gray = False
        componentID = 0
        transformID = 0

        crop_region = CroppingRegion(0, 0, extended_width, extended_height)

        # Create coefficient array, filled with 0:s. The data is arranged in
        # mcus, i.e. first 64 values are for mcu (0, 0), second 64 values for
        # mcu (1, 0)
        coeffs = np.zeros(extended_width * extended_height, dtype=c_short)
        # Fill the mcu corresponding to the original image with 1:s.
        coeffs[0 : original_width * original_height] = 1

        # Make a copy of the original data and change the coefficients for the
        # extended mcus ((0, 0), (1, 0), (1, 1)) manually.
        expected_results = np.copy(coeffs)
        for index in range(mcu_size, extended_width * extended_height, mcu_size):
            expected_results[index] = background_luminance

        planeRegion = CroppingRegion(0, 0, extended_width, extended_width)

        transform_struct = TransformStruct(
            crop_region,
            TJXOP_NONE,
            TJXOPT_PERFECT | TJXOPT_CROP | (gray and TJXOPT_GRAY),
            pointer(
                BackgroundStruct(original_width, original_height, background_luminance)
            ),
            CUSTOMFILTER(fill_background),
        )

        # Act
        # Iterate the callback with one mcu-row of data.
        for row in range(extended_height // callback_row_height):
            data_start = row * callback_row_height * extended_width
            data_end = (row + 1) * callback_row_height * extended_width
            arrayRegion = CroppingRegion(
                0, row * callback_row_height, extended_width, callback_row_height
            )
            _ = fill_background(
                coeffs[data_start:data_end].ctypes.data,
                arrayRegion,
                planeRegion,
                componentID,
                transformID,
                pointer(transform_struct),
            )

        # Assert
        # Compare the modified data with the expected result
        assert np.array_equal(expected_results, coeffs)

    def test_blank_background_callback(self):
        # Arrange
        mcu_size = 64
        extended_width = 16
        extended_height = 16
        callback_row_height = 8
        background_luminance = 508
        transformID = 0
        blank_image_transform = BlankImage()

        crop_region = CroppingRegion(0, 0, extended_width, extended_height)

        # Create coefficient array, filled with 1:s. The data is arranged in
        # mcus, i.e. first 64 values are for mcu (0, 0), second 64 values for
        # mcu (1, 0)
        coeffs = np.ones(extended_width * extended_height, dtype=c_short)

        # The expected result is field with 0:s and luminance dc component
        # changed

        planeRegion = CroppingRegion(0, 0, extended_width, extended_width)

        transform_struct = blank_image_transform.transform(
            crop_region,
            BlankStruct(0, background_luminance),
        )

        # Act
        # Iterate through components
        for componentID in range(3):
            # Expected result is array with 0
            expected_results = np.zeros(extended_width * extended_height, dtype=c_short)
            # For luminance add background luminance to expected result
            if componentID == 0:
                for index in range(0, extended_width * extended_height, mcu_size):
                    expected_results[index] = background_luminance
            # Iterate the callback with one mcu-row of data.
            for row in range(extended_height // callback_row_height):
                data_start = row * callback_row_height * extended_width
                data_end = (row + 1) * callback_row_height * extended_width
                arrayRegion = CroppingRegion(
                    0, row * callback_row_height, extended_width, callback_row_height
                )
                _ = blank_image_transform.callback(
                    coeffs[data_start:data_end].ctypes.data,  # type: ignore
                    arrayRegion,
                    planeRegion,
                    componentID,
                    transformID,
                    pointer(transform_struct),  # type: ignore
                )

            # Assert
            # Compare the modified component with the expected result
            assert np.array_equal(expected_results, coeffs)


@pytest.mark.unittest
class TestFindDqt:
    @staticmethod
    def find_dqt(data: bytes, dqt_index: int) -> Optional[int]:
        """Run the search in a worker thread and fail if it does not terminate. A
        malformed segment length used to leave the scan looping on one marker."""
        result: list[Optional[int]] = []
        thread = threading.Thread(
            target=lambda: result.append(JpegFiller._find_dqt(data, dqt_index)),
            daemon=True,
        )
        thread.start()
        thread.join(5.0)
        assert not thread.is_alive(), "_find_dqt did not terminate"
        return result[0]

    def test_find_first_table(self):
        # Arrange
        data = SOI + dqt_segment(0) + dqt_segment(1)

        # Act
        offset = self.find_dqt(data, 0)

        # Assert
        assert offset == len(SOI)

    def test_find_second_table(self):
        # Arrange
        data = SOI + dqt_segment(0) + dqt_segment(1)

        # Act
        offset = self.find_dqt(data, 1)

        # Assert
        assert offset == len(SOI) + len(dqt_segment(0))

    def test_missing_table_returns_none(self):
        # Arrange
        data = SOI + dqt_segment(0)

        # Act
        offset = self.find_dqt(data, 1)

        # Assert
        assert offset is None

    @pytest.mark.parametrize("length", [0, 1])
    def test_segment_length_below_minimum_returns_none(self, length: int):
        # Arrange
        # A length field below 2 does not move the scan past the marker. Table 1 is
        # searched for while the segment declares table 0, so the index does not
        # match and the scan reaches the advance step.
        data = SOI + dqt_segment(0, length=length)

        # Act
        offset = self.find_dqt(data, 1)

        # Assert
        assert offset is None

    @pytest.mark.parametrize("kept", [0, 1, 2, 3, 4])
    def test_truncated_segment_returns_none(self, kept: int):
        # Arrange
        # Too little room after the marker for the length field and the table id.
        data = SOI + dqt_segment(0)[:kept]

        # Act
        offset = self.find_dqt(data, 0)

        # Assert
        assert offset is None

    def test_truncated_segment_raises_value_error_from_caller(self):
        # Arrange
        data = SOI + MARKER

        # Act, Assert
        with pytest.raises(ValueError, match="Quantisation table"):
            JpegFiller._get_dc_dqt_element(data, 0)

    def test_zero_dc_quantisation_element_raises_value_error(self):
        # Arrange
        # Quantisation values are 1-255, so a zero dc element is invalid and
        # used to divide by zero.
        data = SOI + dqt_segment(0)

        # Act, Assert
        with pytest.raises(ValueError, match="Zero dc quantisation element"):
            JpegFiller._map_luminance_to_dc_dct_coefficient(data, 1.0)
