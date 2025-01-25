import os


def numbers_of_files(directory):
    n = 0

    for file in os.listdir(directory):
        if file.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            n += 1

    return n


