# Image Finder

**Image Finder** is a Python package that leverages the power of OpenCV to locate image and extract text within a range. 

It provides a simple and efficient way to perform image processing tasks such as template matching and text extraction.

## Features

- **Image Matching**: Image Finder can locate a template image within a image using OpenCV template matching techniques.

- **Text Extraction**: Image Finder can extract text within a specified rectangle in an image.

- **Drawing on Images**: Image Finder provides functions to draw points and rectangles on images.

## Usage

Here is a basic example of how to use Image Finder:

```python
from image_finder import ImageFinder


def main():
    # Load an image from a file
    with open('sample_images/pytho_cat.png', 'rb') as f:
        image = f.read()
    finder = ImageFinder(image=image)

    # Load a template image from a file
    with open('sample_images/template.png', 'rb') as f:
        template = f.read()

    # Find the template in the image
    image_coop = finder.find_image_in_screen(template=template, threshold=0.7)
    print(image_coop)

    # Draw a point at the center of the match
    return_image = finder.draw_point(x=image_coop['center_x'], y=image_coop['center_y'], color=(0, 255, 0),
                                     thickness=-1, radius=10)
    with open('sample_images/return_draw_point_image.png', 'wb') as f:
        f.write(return_image)

    # Extract text within a rectangle in an image
    texts = finder.find_text_in_rectangle(top_left_x=0, top_left_y=0, bottom_right_x=896, bottom_right_y=896,
                                          lang=['en'])
    print(texts)

    # Draw a rectangle on an image
    return_image = finder.draw_rectangle(top_left_x=10, top_left_y=10, bottom_right_x=800, bottom_right_y=800,
                                         color=(255, 0, 0), thickness=2)
    with open('sample_images/return_draw_rectangle_image.png', 'wb') as f:
        f.write(return_image)


if __name__ == '__main__':
    main()
```

## Installation

```bash
pip install image-finder
```

## Contributing

Contributions to Image Finder are welcome. Please open an issue or submit a pull request on GitHub.

## License

Image Finder is released under the Apache License.
