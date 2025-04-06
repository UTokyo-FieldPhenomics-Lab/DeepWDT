def find_closer_32k(img_size):
    sizes = []
    for size in img_size:
        sizes.append(round((size / 32)) * 32)
    print(f'Videos will be resize to {sizes} for detection.')
    return sizes
