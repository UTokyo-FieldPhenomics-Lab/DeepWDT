def find_closer_32k(img_size):
    sizes = []
    for size in img_size:
        sizes.append(round((size / 32)) * 32)
    print(f'Resize dimentions: {sizes}')
    return sizes