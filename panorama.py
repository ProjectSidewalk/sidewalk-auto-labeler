import requests
from requests.adapters import HTTPAdapter
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from PIL import Image
import cv2

def fetch_panorama(pano_id):
    def _fetch_tile(x, y, zoom=3):
        url = f"https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid={pano_id}&x={x}&y={y}&zoom={zoom}"
        try:
            s = requests.Session()
            s.mount("https://", HTTPAdapter(max_retries=1))
            response = s.get(url, timeout=20)
            if response.status_code == 200:
                return x, y, Image.open(io.BytesIO(response.content))
            return x, y, None
        except Exception as e:
            print(f"Error fetching tile for pano {pano_id}, x={x}, y={y}: {e}")
            return x, y, None

    def _is_black_tile(tile):
        if tile is None:
            return True
        tile_array = np.array(tile)
        return np.all(tile_array == 0)

    def _find_panorama_dimensions():
        tiles_cache = {}
        x, y = 4, 1
        is_first = True
        while True:
            tile_info = _fetch_tile(x, y)
            if tile_info is None:
                return None
            tile = tile_info[2]
            if tile is None:
                return None
            if is_first:
                is_first = False
                if _is_black_tile(tile):
                    return None
            tiles_cache[(x, y)] = tile
            if _is_black_tile(tile):
                y = y - 1
                while True:
                    tile_info = _fetch_tile(x, y)
                    if tile_info is None:
                        return None
                    tile = tile_info[2]
                    tiles_cache[(x, y)] = tile
                    if _is_black_tile(tile):
                        return x - 1, y, tiles_cache
                    x += 1
            x += 1
            y += 1

    def _fetch_remaining_tiles(max_x, max_y, existing_tiles):
        tiles_cache = existing_tiles.copy()
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = []
            for x in range(max_x + 1):
                for y in range(max_y + 1):
                    if (x, y) not in tiles_cache:
                        futures.append(executor.submit(_fetch_tile, x, y))
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    x, y, tile = result
                    if tile is not None:
                        tiles_cache[(x, y)] = tile
        return tiles_cache

    def _assemble_panorama(tiles, max_x, max_y):
        if not tiles:
            return None
        tile_size = list(tiles.values())[0].size[0]
        panorama = Image.new('RGB', (tile_size * (max_x + 1), tile_size * (max_y + 1)))
        for (x, y), tile in tiles.items():
            panorama.paste(tile, (x * tile_size, y * tile_size))
        return panorama

    def _crop(image):
        img_array = np.array(image)
        y_nonzero, x_nonzero, _ = np.nonzero(img_array)
        if y_nonzero.size > 0 and x_nonzero.size > 0:
            return img_array[np.min(y_nonzero):np.max(y_nonzero) + 1, np.min(x_nonzero):np.max(x_nonzero) + 1]
        return img_array

    dimension_result = _find_panorama_dimensions()
    if dimension_result is None:
        return None
    max_x, max_y, initial_tiles = dimension_result
    full_tiles = _fetch_remaining_tiles(max_x, max_y, initial_tiles)
    assembled_panorama = _assemble_panorama(full_tiles, max_x, max_y)
    if assembled_panorama is None:
        return None
    cropped_panorama = _crop(assembled_panorama)
    height, width = cropped_panorama.shape[:2]

    max_width = height * 2
    cropped_panorama = cropped_panorama[:, :max_width]
    
    resized = cv2.resize(cropped_panorama, (4096, 2048), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)