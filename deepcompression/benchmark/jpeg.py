import cv2
import numpy as np


def encode(img, jpeg_quality=95):
    """
    :param img: (*, *, 3) array with RGB convention, with values in [0, 1]
    :param jpeg_quality: from 0 to 100
    :return:
    """

    img_bgr = cv2.cvtColor((img * 256).astype(np.uint8), cv2.COLOR_RGB2BGR)
    _, enc = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    return enc


def decode(enc):

    dec = cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB)
    return dec / 256


if __name__=='__main__':

    from deepcompression.data.imagenet import ImageNet
    import matplotlib.pyplot as plt

    data = ImageNet('train', 256, 10)
    img = data[0].permute(1, 2, 0).numpy()

    # Plot raw image and reconstructed
    enc = encode(img, jpeg_quality=10)
    print('Compression ratio:', np.product(img.shape) // enc.shape[0])
    img_ = decode(enc)

    fig = plt.figure(figsize=(12, 8))
    g = fig.add_subplot(121) ; plt.imshow(img)
    g = fig.add_subplot(122) ; plt.imshow(img_)
    plt.show()