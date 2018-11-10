import glob
import cv2


def main():
    for imgs in glob.glob('./data/dog/*'):
        img = cv2.imread(imgs)
        res = cv2.resize(img, (224, 224))
        cv2.imwrite(imgs, res)


if __name__ == '__main__':
    main()