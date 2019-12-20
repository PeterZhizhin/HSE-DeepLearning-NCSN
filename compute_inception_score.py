import argparse
from pathlib import Path

import scipy.misc
from tqdm import tqdm

import inception_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_folder', required=True)
    return parser.parse_args()


def images_to_array(path):
    path = Path(path)
    result = []
    for image in tqdm(path.iterdir()):
        if image.is_file():
            image_np = scipy.misc.imread(str(image))
            result.append(image_np)
    return result


def main():
    args = parse_args()
    all_images = images_to_array(args.images_folder)

    inception_score_mean, inception_score_std = inception_score.get_inception_score(all_images)
    print('Inception score mean: {}'.format(inception_score_mean))
    print('Inception score std : {}'.format(inception_score_std))


if __name__ == "__main__":
    main()
