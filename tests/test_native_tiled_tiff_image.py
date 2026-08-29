#    Copyright 2026 SECTRA AB
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

import pytest
from decoy import Decoy
from tifffile import COMPRESSION, TiffPage

from opentile.file import OpenTileFile
from opentile.geometry import Size, SizeMm
from opentile.tiff_image_bases import NativeTiledLevelImage

FRAMES = [b"tile-0", b"tile-1", b"tile-2", b"tile-3"]


@pytest.fixture()
def level(decoy: Decoy):
    """A level of two by two tiles, each serving its own identifiable frame."""
    page = decoy.mock(cls=TiffPage)
    decoy.when(page.compression).then_return(COMPRESSION.JPEG)
    decoy.when(page.imagewidth).then_return(4)
    decoy.when(page.imagelength).then_return(4)
    decoy.when(page.is_tiled).then_return(True)
    decoy.when(page.tilewidth).then_return(2)
    decoy.when(page.tilelength).then_return(2)
    decoy.when(page.jpegtables).then_return(None)
    decoy.when(page.dataoffsets).then_return(tuple(range(len(FRAMES))))
    decoy.when(page.databytecounts).then_return((1,) * len(FRAMES))
    file = decoy.mock(cls=OpenTileFile)
    for index, frame in enumerate(FRAMES):
        decoy.when(file.read(index, 1)).then_return(frame)
        decoy.when(file.read_multiple([(index, 1)])).then_return([frame])
    decoy.when(file.read_multiple([(0, 1), (3, 1)])).then_return([FRAMES[0], FRAMES[3]])
    yield NativeTiledLevelImage(page, file, Size(4, 4), SizeMm(0.25, 0.25))


@pytest.mark.unittest
class TestNativeTiledTiffImage:
    @pytest.mark.parametrize(
        ["tile_position", "expected"],
        [
            ((0, 0), FRAMES[0]),
            ((1, 0), FRAMES[1]),
            ((0, 1), FRAMES[2]),
            ((1, 1), FRAMES[3]),
        ],
    )
    def test_get_tile(
        self,
        level: NativeTiledLevelImage,
        tile_position: tuple[int, int],
        expected: bytes,
    ):
        # Arrange

        # Act
        tile = level.get_tile(tile_position)

        # Assert
        assert tile == expected

    @pytest.mark.parametrize("tile_position", [(2, 0), (0, 2), (2, 2), (9, 9)])
    def test_get_tile_past_tiled_size_raises(
        self, level: NativeTiledLevelImage, tile_position: tuple[int, int]
    ):
        # Arrange
        # A position one past the last column used to serve the first tile of the
        # next row, as both map to the same frame index.

        # Act, Assert
        with pytest.raises(ValueError, match="outside tiled size"):
            level.get_tile(tile_position)

    @pytest.mark.parametrize("tile_position", [(-1, 0), (0, -1), (-1, -1)])
    def test_get_tile_negative_position_raises(
        self, level: NativeTiledLevelImage, tile_position: tuple[int, int]
    ):
        # Arrange
        # A negative position used to index backwards from the last frame.

        # Act, Assert
        with pytest.raises(ValueError, match="outside tiled size"):
            level.get_tile(tile_position)

    @pytest.mark.parametrize("tile_position", [(2, 0), (-1, 0)])
    def test_get_tiles_outside_tiled_size_raises(
        self, level: NativeTiledLevelImage, tile_position: tuple[int, int]
    ):
        # Arrange

        # Act, Assert
        with pytest.raises(ValueError, match="outside tiled size"):
            list(level.get_tiles([(0, 0), tile_position]))

    def test_get_tiles(self, level: NativeTiledLevelImage):
        # Arrange

        # Act
        tiles = list(level.get_tiles([(0, 0), (1, 1)]))

        # Assert
        assert tiles == [FRAMES[0], FRAMES[3]]
