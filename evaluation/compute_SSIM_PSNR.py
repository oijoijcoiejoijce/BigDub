import ssim.ssimlib as ssim
import PIL.Image as Image
import cv2
import os
from tqdm import tqdm


def get_ssim(img1, img2):
    img1 = Image.fromarray(img1)
    img2 = Image.fromarray(img2)
    img = ssim.SSIM(img1)
    return img.cw_ssim_value(img2)


def get_psnr(img1, img2):
    return cv2.PSNR(img1, img2)

def resize_images(img1, img2):
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return img1, img2


def get_metrics_video(video1_path, video2_path):

    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    ssim = 0
    psnr = 0
    n = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        frame1, frame2 = resize_images(frame1, frame2)
        ssim += get_ssim(frame1, frame2)
        psnr += get_psnr(frame1, frame2)
        n += 1
    cap1.release()
    cap2.release()
    return ssim / n, psnr / n


def main(args):

    if args.video1 and args.video2:
        ssim, psnr = get_metrics_video(args.video1, args.video2)
        print(f'SSIM: {ssim}, PSNR: {psnr}')
    elif args.dir1 and args.dir2:
        ssim = 0
        psnr = 0
        n = 0
        for video in tqdm(os.listdir(args.dir1)):
            ssim_, psnr_ = get_metrics_video(os.path.join(args.dir1, video), os.path.join(args.dir2, video))
            ssim += ssim_
            psnr += psnr_
            n += 1
        print(args.dir1, args.dir2)
        print(f'SSIM: {ssim / n}, PSNR: {psnr / n}')
    else:
        print('Please provide either two videos or two directories')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video1', type=str, default=None)
    parser.add_argument('--video2', type=str, default=None)
    parser.add_argument('--dir1', type=str, default=None)
    parser.add_argument('--dir2', type=str, default=None)
    args = parser.parse_args()
    main(args)
