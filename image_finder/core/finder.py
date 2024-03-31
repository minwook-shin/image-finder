# noinspection PyPackageRequirements
import cv2
import easyocr
import numpy as np


class ImageFinder:
    """
    A class used to find images and text within image.
    ...
    Attributes
    ----------
    image : bytes
        the image in which to find other images or text

    Methods
    -------
    find_image_in_screen(template, threshold)
        Finds matches of a template in an image using template matching.
    find_text_in_rectangle(top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        Extracts the texts that are inside a rectangle.
    draw_point(x, y, color, thickness)
        Draws a point on an image at the coordinates.
    draw_rectangle(top_left_x, top_left_y, bottom_right_x, bottom_right_y, color, thickness)
        Draws a rectangle on an image at the coordinates.
    """

    def __init__(self, image):
        self.image = cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_COLOR)

    # https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
    def find_image_in_screen(self, template, threshold: float) -> dict:
        """
        This function finds matches of a template in an image using template matching.

        Parameters:
        template_input (bytes): The template image in bytes.
        threshold (float): The threshold for the match.

        Returns:
        dict: A dictionary containing the coordinates of the top left corner of the match,
              the coordinates of the center of the match, and the match score.
        """
        template_np = cv2.imdecode(np.fromstring(template, np.uint8), cv2.IMREAD_GRAYSCALE)

        # Perform template matching on the grayscale image and template
        res = cv2.matchTemplate(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), template_np, cv2.TM_CCOEFF_NORMED)

        # Get the minimum value, maximum value, and their respective locations from the result
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val < threshold:
            return {"x": -1, "y": -1, "center_x": -1, "center_y": -1, "score": max_val}

        # Calculate the center of the match
        bottom_right = (max_loc[0] + template_np.shape[::-1][0], max_loc[1] + template_np.shape[::-1][1])
        center = ((max_loc[0] + bottom_right[0]) // 2, (max_loc[1] + bottom_right[1]) // 2)

        return {"x": max_loc[0], "y": max_loc[1], "center_x": center[0], "center_y": center[1], "score": max_val}

    def find_text_in_rectangle(self, top_left_x: int, top_left_y: int,
                               bottom_right_x: int, bottom_right_y: int, lang: list = None) -> list:
        """
        extracts the texts that are inside a specified rectangle.

        Parameters:
        top_left_x (int): The x-coordinate of the top left corner of the rectangle.
        top_left_y (int): The y-coordinate of the top left corner of the rectangle.
        bottom_right_x (int): The x-coordinate of the bottom right corner of the rectangle.
        bottom_right_y (int): The y-coordinate of the bottom right corner of the rectangle.
        lang (list): The languages to be used for text extraction. The default is ['ko', 'en'].

        Returns:
        list: A list of texts that are inside the specified rectangle.
        """
        if lang is None:
            lang = ['ko', 'en']
        texts_in_rectangle = []
        for coord, text, _ in easyocr.Reader(lang).readtext(self.image):
            center = {"x": (coord[0][0] + coord[2][0]) // 2, "y": (coord[0][1] + coord[2][1]) // 2}
            if top_left_x <= center['x'] <= bottom_right_x and top_left_y <= center['y'] <= bottom_right_y:
                texts_in_rectangle.append(text)
        return texts_in_rectangle

    def draw_point(self, x: int, y: int, radius: int, color: tuple, thickness: int) -> bytes:
        """
        This function draws a point on an image at the coordinates.

        Parameters:
        x (int): The x-coordinate of the point.
        y (int): The y-coordinate of the point.
        radius (int): The radius of the point.
        color (tuple): The color of the point in BGR format. For example, (0, 255, 0) for green.
        thickness (int): The thickness of the point.

        Returns:
        ndarray: The image with the point drawn.
        """
        return cv2.imencode(".png", cv2.circle(self.image, (x, y),
                                               radius=radius, color=color, thickness=thickness))[1].tobytes()

    def draw_rectangle(self, top_left_x: int, top_left_y: int,
                       bottom_right_x: int, bottom_right_y: int, color: tuple, thickness: int) -> bytes:
        """
        This function draws a rectangle on an image at the specified coordinates.

        Parameters:
        top_left_x (int): The x-coordinate of the top left corner of the rectangle.
        top_left_y (int): The y-coordinate of the top left corner of the rectangle.
        bottom_right_x (int): The x-coordinate of the bottom right corner of the rectangle.
        bottom_right_y (int): The y-coordinate of the bottom right corner of the rectangle.
        color (tuple): The color of the rectangle in BGR format. For example, (0, 255, 0) for green.
        thickness (int): The thickness of the lines that make up the rectangle.

        Returns:
        bytes: The image with the rectangle drawn, returned as bytes.
        """
        return cv2.imencode(".png", cv2.rectangle(self.image,
                                                  (int(top_left_x), int(top_left_y)),
                                                  (int(bottom_right_x), int(bottom_right_y)),
                                                  color, thickness))[1].tobytes()
