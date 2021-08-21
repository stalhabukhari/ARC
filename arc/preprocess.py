"""
Preprocess data (before training/testing)
Run as:
    python preprocess.py <load_path> <save_path>
"""

import os, sys, math
import scipy, cv2
import numpy as np

from skimage.morphology import convex_hull_image
from skimage.measure import label, regionprops


def preprocess(load_dir, save_dir, objs=range(1, 100+1), imgs=range(1, 310+1)):
    """ Preprocessor """

    assert os.path.isdir(load_dir)
    assert os.path.isdir(save_dir)

    # file nomenclature:
    img_load_path = os.path.join(load_dir, 'obj{obj}', 'obj{obj}_img{img}.jpg')
    obj_save_path = os.path.join(save_dir, 'obj{obj}')
    img_save_path = os.path.join(obj_save_path, 'obj{obj}_img{img}.jpg')

    for obj in objs:
        print(f'*** Object: {obj}')
        if not os.path.isdir(obj_save_path.format(obj=obj)):
            os.mkdir(obj_save_path.format(obj=obj))

        for count in imgs:
            image = cv2.imread(img_load_path.format(obj=obj, img=count))

            if image is None:
                print('Empty:', count)
            else:
                image = image[:, 30:-80]
                image = rot_correct(image)
                image = image[40:-40, 40:-40]
                image = obj_extract(image)
                image = sharpen(image)
                image = hist_eq(image)
                image = resizer(image)

                write_path = img_save_path.format(obj=obj, img=count)
                cv2.imwrite(write_path, image)
                print('Image Written:', write_path)


def rot_correct(image):
    img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    r = img.shape[0]
    c = img.shape[1]
    img = img[5:-5, 5:-5]

    ret,img_bw = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    convex_hull = convex_hull_image(img_bw)

    m,n = convex_hull.shape
    im = np.zeros((m,n))
    for i in range(1,m):
        for j in range(1,n):
            if convex_hull[i,j]:
                im[i,j] = 1
            else:
                im[i,j] = 0

    label_img = label(im)
    region = regionprops(label_img)

    for props in region:
            y0, x0 = props.centroid
            orientation = props.orientation
            x1 = x0 + math.cos(orientation) * 0.5 * props.major_axis_length
            y1 = y0 - math.sin(orientation) * 0.5 * props.major_axis_length
            x2 = x0 - math.sin(orientation) * 0.5 * props.minor_axis_length
            y2 = y0 - math.cos(orientation) * 0.5 * props.minor_axis_length
            ln = cv2.line(im, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 1)

    m1 = (y1 - y0)/(x1 - x0)
    angle = math.atan(m1) * 180/math.pi
    image = scipy.misc.imrotate(image, angle)
    return image


def obj_extract(image):
    edged = cv2.Canny(image, 100, 250, 3,3, True)

    # apply morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # finding_contours
    _, contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    X, Y, W, H, Diag = [np.array([])]*5

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        # print(x,y,w,h,'| Area:',(x+w)*(y+h)) #<--
        X = np.append(X,x)
        Y = np.append(Y,y)
        W = np.append(W,w)
        H = np.append(H,h)
        Diag = np.append(Diag,(w**2+h**2)**(1/2))
        # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

    ind = np.argmax(Diag)
    x = int(X[ind])
    y = int(Y[ind])
    w = int(W[ind])
    h = int(H[ind])

    return image[y:y+h, x:x+w]


def sharpen(image):
    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]], np.float32)
    image = cv2.filter2D(image, -1, kernel)

    return image


def hist_eq(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.equalizeHist(y)

    ycrcb = cv2.merge((y, cr, cb))
    image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return image


def resizer(image):
    # Resizing to within 150x150 Box
    r, c, _ = image.shape
    # aspRatio = c/r

    if c > r:     # C -> 150, R -> C/aspRatio
        z = np.zeros((int((c-r)/2), c, 3), dtype=np.uint8)
        image = np.vstack((z, image, z))
    else:       # R -> 150, C -> R*aspRatio
        z = np.zeros((r, int((r-c)/2), 3), dtype=np.uint8)
        image = np.hstack((z, image, z))

    image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_AREA)
    return image


if __name__ == '__main__':
    assert len(sys.argv) == 3

    load_path = sys.argv[1]
    save_path = sys.argv[2]
    preprocess(load_path, save_path)
