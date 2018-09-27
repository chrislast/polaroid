from sklearn.cluster import MiniBatchKMeans
import numpy as np
import sys
import cv2

from pathlib import Path

def main(*args):
    """Take a picture.
    """
    print(args)
    photo = args[0][-1]
    img = cv2.imread(photo)
    small = cv2.resize(img, (210*1, 102*1))
    (h, w) = small.shape[:2]

    # convert the image from the RGB color space to the L*a*b*
    # color space -- since we will be clustering using k-means
    # which is based on the euclidean distance, we'll use the
    # L*a*b* color space where the euclidean distance implies
    # perceptual meaning
    image = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
     
    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))
 
    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters = 3)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
     
    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))
     
    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    palette = dict()
    for _ in quant.tolist():
        for x, y, z in _:
            palette[x*y*z] = [x, y, z]

    BYW = sorted(palette.keys())

    while True:
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        #cv2.imshow('small', small)

        # display the images and wait for a keypress
        #cv2.imshow("image", quant)
        cv2.imshow("image", np.hstack([image, quant]))

    cv2.imwrite("result.jpg", quant)

if __name__ == "__main__":
    sys.exit(main(sys.argv))
