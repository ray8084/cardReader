# Mahjong Card Reader

A Python script to read mahjong card images from PNG files and extract hand information into JSON format.

## Features

- Reads mahjong card PNG images
- Detects and recognizes mahjong tiles
- Extracts hand information
- Exports results to JSON format

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd cardReader
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```bash
python card_reader.py <image_path> [output_json_path]
```

### Examples

```bash
# Basic usage
python card_reader.py mahjong_hand.png

# Specify output file
python card_reader.py mahjong_hand.png output.json
```

## How It Works

1. **Image Loading**: Loads the PNG image using OpenCV
2. **Tile Detection**: Uses contour detection to identify potential tile regions
3. **Tile Recognition**: Recognizes individual tiles (placeholder implementation)
4. **Hand Parsing**: Groups detected tiles into mahjong hands
5. **JSON Export**: Saves results to a JSON file

## Extending the Code

### Implementing Tile Recognition

The current implementation uses a placeholder for tile recognition. To implement actual recognition, you can:

1. **Template Matching**: Create reference images for each tile and use `cv2.matchTemplate()`
2. **Deep Learning**: Train a CNN model to classify tiles
3. **Feature Detection**: Use SIFT/ORB features for matching
4. **OCR**: Extract numbers and characters using Tesseract OCR

### Example: Template Matching

```python
def recognize_tile(self, tile_region: np.ndarray) -> str:
    best_match = None
    best_score = 0
    
    for tile_name, template in self.template_images.items():
        result = cv2.matchTemplate(tile_region, template, cv2.TM_CCOEFF_NORMED)
        score = np.max(result)
        if score > best_score:
            best_score = score
            best_match = tile_name
    
    return best_match if best_score > 0.7 else "unknown"
```

## Output Format

The JSON output includes:
- `image_path`: Path to the source image
- `tiles_detected`: Number of tiles detected
- `hands`: Information about detected hands
- `raw_tiles`: Raw tile detection data

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## License

MIT

