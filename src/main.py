import argparse
import readUtils
from train import train
from test import image_type

def main():
    parser = argparse.ArgumentParser(description='Read Input Data')
    parser.add_argument('--i', '--input-directory', help='path to input png directory')
    parser.add_argument('--img', help='path to test img.png')

    args = parser.parse_args()

    model = train(args.i)

    if(args.img):
        predictedType = image_type(model, args.img)
        print("Type of Tissue in image[{}] is {}".format(args.img, predictedType))

if __name__ == '__main__':
    main()
