import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from PIL import Image
import cv2

class Panorama:
    def __init__(self, pano_id):
        self.pano_id = pano_id
        self.panorama_image = None
        self.zoom = None
        self._fetch_panorama()

    def _is_black_tile(self, tile):
        if tile is None:
            return True
        tile_array = np.array(tile)
        return np.all(tile_array == 0)

    def _fetch_tile(self, x, y, zoom=4):
        if self.zoom != None:
            zoom = self.zoom
        
        url = (
            f"https://streetviewpixels-pa.googleapis.com/v1/tile"
            f"?cb_client=maps_sv.tactile&panoid={self.pano_id}"
            f"&x={x}&y={y}&zoom={zoom}"
        )
        try:
            response = requests.get(url)
            if response.status_code == 200:
                tile = Image.open(io.BytesIO(response.content))
                if self.zoom != None or not self._is_black_tile(tile):
                    self.zoom = zoom
                    return x, y, tile
        except Exception as e:
            print(e)

        if self.zoom == None:
            # Try fallback with zoom=3
            fallback_url = (
                f"https://streetviewpixels-pa.googleapis.com/v1/tile"
                f"?cb_client=maps_sv.tactile&panoid={self.pano_id}"
                f"&x={x}&y={y}&zoom=3"
            )
            try:
                response = requests.get(fallback_url)
                if response.status_code == 200:
                    tile = Image.open(io.BytesIO(response.content))
                    self.zoom = 3
                    return x, y, tile
            except Exception as e:
                print(e)

        return x, y, None

    def _find_panorama_dimensions(self):
        tiles_cache = {}
        x, y = 5, 2

        is_first = True

        while True:
            tile = self._fetch_tile(x, y)[2]
            if tile is None:
                return None  # Invalid panorama

            if is_first:
                is_first = False
                if self._is_black_tile(tile):
                    return None  # Invalid panorama

            tiles_cache[(x, y)] = tile

            if self._is_black_tile(tile):
                y = y - 1

                while True:
                    tile = self._fetch_tile(x, y)[2]
                    tiles_cache[(x, y)] = tile

                    if self._is_black_tile(tile):
                        return x - 1, y, tiles_cache

                    x += 1

            x += 1
            y += 1

    def _fetch_remaining_tiles(self, max_x, max_y, existing_tiles):
        tiles_cache = existing_tiles.copy()

        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = []
            for x in range(max_x + 1):
                for y in range(max_y + 1):
                    if (x, y) not in tiles_cache:
                        futures.append(executor.submit(self._fetch_tile, x, y))

            for future in as_completed(futures):
                x, y, tile = future.result()
                if tile is not None:
                    tiles_cache[(x, y)] = tile

        return tiles_cache

    def _assemble_panorama(self, tiles, max_x, max_y):
        tile_size = list(tiles.values())[0].size[0]
        panorama = Image.new('RGB', (tile_size * (max_x + 1), tile_size * (max_y + 1)))

        for (x, y), tile in tiles.items():
            panorama.paste(tile, (x * tile_size, y * tile_size))

        return panorama

    def _crop(self, image):
        y_nonzero, x_nonzero, _ = np.nonzero(image)
        return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

    def _fetch_panorama(self):
        dimension_result = self._find_panorama_dimensions()
        if dimension_result is None:
            self.panorama_image = None
            return

        max_x, max_y, initial_tiles = dimension_result
        full_tiles = self._fetch_remaining_tiles(max_x, max_y, initial_tiles)
        result = cv2.cvtColor(np.array(self._assemble_panorama(full_tiles, max_x, max_y)), cv2.COLOR_RGB2BGR)
        self.panorama_image = cv2.resize(self._crop(result), (8192, 4096), 
               interpolation = cv2.INTER_LINEAR)

    def get_equi(self):
        return self.panorama_image