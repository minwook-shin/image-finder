from image_finder import ImageFinder


def main():
    with open('sample_images/pytho_cat.png', 'rb') as f:
        image = f.read()
    finder = ImageFinder(image=image)

    with open('sample_images/template.png', 'rb') as f:
        template = f.read()
    image_coop = finder.find_image_in_screen(template=template, threshold=0.7)
    # image_coop : {'x': 763, 'y': 940, 'center_x': 995, 'center_y': 1172, 'score': 0.7702369093894958}
    print(image_coop)

    with open('sample_images/hubot.jpeg', 'rb') as f:
        image = f.read()
    finder = ImageFinder(image=image)
    # Downloading recognition model, please wait. This may take several minutes depending upon your network connection.
    texts = finder.find_text_in_rectangle(top_left_x=0, top_left_y=0, bottom_right_x=896, bottom_right_y=896,
                                          lang=['en'])
    # ['I HAVE', 'Found', 'THE', 'THINGS', 'Hu-Bot']
    print(texts)


if __name__ == '__main__':
    main()
