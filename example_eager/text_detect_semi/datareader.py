import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import csv
import random

train_img_path = './task1/train/img/'
train_lab_path = './task1/train/gt/'
img_fn='./task1/train/img/'

def get_image(img_fn):

    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(img_fn):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
            # if len(files)==1:
            #     return files
    # print('Find {} images'.format(len(files)))
    return files



def load_annoataion(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    #print(p)
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)
    with open(p, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            label = line[-1]
            if label == '-':
                continue
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
            idx_thres = random.uniform(0,1)
            # if idx_thres>0.5:
            #     continue
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            if abs(x1-x3)<2 or abs(y1-y3)<2:
                continue
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '*' or label == '###':
                text_tags.append(True)
            else:
                text_tags.append(False)
    return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)

def resize_im(size,im, lab):
    img = im.copy()
    lab_pro = lab.copy()
    h,w,_ = im.shape
    if max(h,w)>size:
        ratio = float(size)/h if h>w else float(size)/w
    else:
        ratio = 1
    img = cv2.resize(img, dsize=None, fx=ratio, fy=ratio)
    labs = lab_pro*ratio
    return img, labs

def plot(img, lab, color):
    img_pro = img.copy()
    img_pro = np.uint8(img_pro)
    lab_pro = lab.copy()
    for lab in lab_pro:
        x1, y1, x2, y2 = lab[0], lab[1], lab[4], lab[5]
        cv2.rectangle(img_pro, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    # plt.ion()
    plt.imshow(img_pro)
    plt.show()
    # plt.pause(5)
    # plt.close()


def random_scale(img, lab):
    img_pro = img.copy()
    lab_pro = lab.copy()
    rand_sc = np.array([0.5,1,0.75])
    rand_sc = np.random.choice(rand_sc)
    #rand_sc = 1.0
    img_pro = cv2.resize(img_pro, dsize=None, fx=rand_sc, fy=rand_sc)
    lab_pro = lab_pro*rand_sc
    return img_pro, lab_pro

def random_crop(img, lab, tags,thresh):
    (min_thresh, max_thresh) = thresh
    img_pro = img.copy()
    lab_pro = lab.copy()
    tags_pro = tags.copy()
    max_tries = 80
    h,w, _= img_pro.shape
    x_pad = 2
    y_pad = 2
    canv = np.zeros((h+y_pad*2, w+x_pad*2), dtype=np.uint8)
    text_polys = lab_pro
    for text_poly in text_polys:
        text_poly = np.round(text_poly)
        xmin = np.min(text_poly[:,0])
        xmax = np.max(text_poly[:,0])
        ymin = np.min(text_poly[:,1])
        ymax = np.max(text_poly[:,1])
        canv[int(ymin+y_pad):int(ymax+y_pad), int(xmin+x_pad):int(xmax+x_pad)] = 1
    y_axis, x_axis = np.where(canv==0)
    for i in range(max_tries):
        x_chs = np.random.choice(x_axis, 2)
        y_chs = np.random.choice(y_axis, 2)
        xxmin = min(x_chs) - x_pad
        xxmax = max(x_chs) - x_pad
        yymin = min(y_chs) - y_pad
        yymax = max(y_chs) - y_pad
        xxmin = np.clip(xxmin, 0, w-1)
        xxmax = np.clip(xxmax, 0, w-1)
        yymin = np.clip(yymin, 0, h-1)
        yymax = np.clip(yymax, 0, h-1)

        x_range = xxmax-xxmin
        y_range = yymax-yymin
        if x_range<0.5*w or y_range<0.5*h:
            continue
        if text_polys.shape[0]!=0:
            polys = text_polys
            poly_axis_in_area = (polys[:,:,0]>=xxmin)&(polys[:,:,0]<=xxmax)&(polys[:,:,1]>=yymin)&(polys[:,:,1]<=yymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            continue

        if len(selected_polys) == 0:
            continue
        else:
            break

    img_pro = img_pro[yymin:yymax+1, xxmin:xxmax+1, :]
    try:
        polys_pro = text_polys[selected_polys]
    except:
        img_pro = img.copy()
        text_polys = lab.copy()
        tags_pro = tags.copy()
        img_pro, text_polys = resize_im(512, img_pro, text_polys)

        return img_pro, text_polys, tags_pro

    tags_pro = tags_pro[selected_polys]
    polys_pro[:,:,0] -= xxmin
    polys_pro[:,:,1] -= yymin

    h_range = []
    if polys_pro.sum() == 0:
        return img_pro, polys_pro, tags_pro
    for poly in polys_pro:
        h_range.append(max(poly[:,1]) - min(poly[:,1]))


    h_max = max(h_range)
    h_min = min(h_range)
    sc_up = min_thresh/h_min
    sc_lo = max_thresh/h_max
    ratio = random.uniform(sc_up, sc_lo) if sc_up>sc_lo else random.uniform(sc_lo, sc_up)
    img_pro = cv2.resize(img_pro, dsize=None, fx=ratio, fy=ratio)
    polys_pro = polys_pro*ratio

    h,w,_=img_pro.shape
    x_pad = 2
    y_pad = 2
    if max(h,w)>512:
        x_max = polys_pro[:,:,0].max()
        y_max = polys_pro[:,:,1].max()
        x_min = polys_pro[:,:,0].min()
        y_min = polys_pro[:,:,1].min()
        x_bin = int(x_min-x_pad)
        x_bin = np.clip(x_bin,0,w-1)
        x_end = int(x_max+x_pad)
        x_end = np.clip(x_end, 0, w - 1)
        y_bin = int(y_min-y_pad)
        y_bin = np.clip(y_bin, 0, h - 1)
        y_end = int(y_max+y_pad)
        y_end = np.clip(y_end, 0, h - 1)
        if (x_end-x_bin)<512 and (y_end-y_bin)<512:
            x_cen = (x_end+x_bin)/2
            y_cen = (y_end+y_bin)/2
            x_bin = int(x_cen) - 256
            x_tbin = np.clip(x_bin, 0, w-1)
            det_x = abs(x_tbin-x_bin)
            x_end = int(x_cen) + 256 + det_x
            x_tend = np.clip(x_end, 0, w-1)
            y_bin = int(y_cen) - 256
            y_tbin = np.clip(y_bin, 0, h-1)
            det_y = abs(y_tbin-y_bin)
            y_end = int(y_cen) + 256 + det_y
            y_tend = np.clip(y_end, 0, h-1)
            img_new = img_pro[y_tbin:y_tend, x_tbin:x_tend, :]
            polys_pro[:,:,0] -= x_tbin
            polys_pro[:,:,1] -= y_tbin
        else:
            img_new = img_pro[y_bin:y_end, x_bin:x_end, :]
            polys_pro[:,:,0] -= x_bin
            polys_pro[:,:,1] -= y_bin
            h,w,_ = img_new.shape
            if max(h,w)>512:
                ratio = float(512)/h if h>w else float(512)/w
            else:
                ratio = 1
            img_new = cv2.resize(img_new, dsize=None, fx=ratio, fy=ratio)
            polys_pro *= ratio
        img_pro = img_new
        #print(img_pro.shape)
        h,w,_ = img_pro.shape

    return img_pro, polys_pro, tags_pro

    # if min(h,w)<300:
    #     ratio2 = float(300)/h if h<w else float(300)/w
    # else:
    #     ratio2 = 1
    #
    # if max(h*ratio2,w*ratio2)> 512:
    #     ratio1 = float(512)/(h*ratio2) if h>w else float(512)/(w*ratio2)
    # else:
    #     ratio1 = 1
    # ratio = ratio1*ratio2
    #
    # img_pro = cv2.resize(img_pro, dsize=None, fx=ratio, fy=ratio)
    # polys_pro = polys_pro*ratio

    #return img_pro, polys_pro, tags_pro

def random_pad(img, lab):
    img_pro = img.copy()
    lab_pro = lab.copy()


    im_pad = np.zeros((512,512,3), dtype=np.uint8)
    h,w,_ = img_pro.shape
    try:
        xxmin = np.random.choice(range(0, 511-w+1))
    except:
        xxmin = 0
    xxmax = xxmin+w
    try:
        yymin = np.random.choice(range(0, 511-h+1))
    except:
        yymin = 0
    yymax = yymin+h

    im_pad[yymin:yymax, xxmin:xxmax, :] = img_pro

    if lab_pro.sum() != 0:
        lab_pro[:,:,0] += xxmin
        lab_pro[:,:,1] += yymin

    #print('pad', im_pad.shape)
    return im_pad, lab_pro


def data_reader(img_fn):
    files = get_image(img_fn)
    result = []
    for file in files:
        txt_fn = file.replace('/task1/train/img/', '/new_gt_50/').replace('.jpg', '.txt')
        text_polys, text_tags = load_annoataion(txt_fn)
        img = cv2.imread(file)
        h,w,_ = img.shape
        if max(h,w)>512:
            ratio = float(512)/h if h>w else float(512)/w
        else:
            ratio = 1
        img = cv2.resize(img, dsize=None, fx=ratio, fy=ratio)
        text_polys = text_polys*ratio
        bundle = [img, text_polys, text_tags]
        result.append(bundle)
    return result

def annoataion_to_grid(img,text_polys, grid_num):
    grid_size = 512/grid_num
    conf_map = np.zeros((grid_num, grid_num, 1), dtype=np.uint8)
    geo_map = np.zeros((grid_num, grid_num, 4), dtype=np.float32)
    geo_map[:,:,2:4] = 1
    result = []


    for poly in text_polys:
        x_cen = np.mean(poly[:, 0])
        y_cen = np.mean(poly[:, 1])
        x_idx = x_cen//grid_size
        y_idx = y_cen//grid_size
        cen_bx = x_idx * grid_size + 0.5*grid_size
        cen_by = y_idx * grid_size + 0.5*grid_size
        det_x = cen_bx - x_cen
        det_y = cen_by - y_cen

        w_poly = max(poly[:,0] - min(poly[:,0]))
        h_poly = max(poly[:,1] - min(poly[:,1]))
        ## x, y, w, h

        conf_map[int(y_idx), int(x_idx), :] = 1
        geo_map[int(y_idx), int(x_idx), 0] = det_x
        geo_map[int(y_idx), int(x_idx), 1] = det_y
        geo_map[int(y_idx), int(x_idx), 2] = np.log(w_poly)
        geo_map[int(y_idx), int(x_idx), 3] = np.log(h_poly)
    bundle = [img, conf_map, geo_map]
    return bundle

def map_to_box(conf_map, geo_map, thresh):

    [y_map, x_map, _] = np.where(conf_map > thresh)
    polys = []
    for i in range(len(x_map)):
        x_idx1 = x_map[i]
        y_idx1 = y_map[i]
        cen_bx1 = x_idx1 * 16 + 8
        cen_by1 = y_idx1 * 16 + 8
        det_x1 = geo_map[y_idx1, x_idx1, 0]
        det_y1 = geo_map[y_idx1, x_idx1, 1]
        w_poly1 = geo_map[y_idx1, x_idx1, 2]
        h_poly1 = geo_map[y_idx1, x_idx1, 3]
        w_poly1 = np.exp(w_poly1)
        h_poly1 = np.exp(h_poly1)
        x_cen1 = cen_bx1 - det_x1
        y_cen1 = cen_by1 - det_y1
        xmin = x_cen1 - 0.5 * w_poly1
        xmax = x_cen1 + 0.5 * w_poly1
        ymin = y_cen1 - 0.5 * h_poly1
        ymax = y_cen1 + 0.5 * h_poly1

        poly = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, 0]
        polys.append(poly)
    if polys!= []:

        polys = np.stack(polys, 0)
    return polys



def processe_bundle(bundle):
    [img, text_polys, text_tags] = bundle
    img, text_polys = random_scale(img, text_polys)
    img, text_polys, text_tags = random_crop(img, text_polys, text_tags, (15,30))
    img, text_polys = random_pad(img, text_polys)
    result = annoataion_to_grid(img,text_polys, grid_num=32)
    return result




if __name__=='__main__':
    files = get_image(img_fn)

    for file in files:
        txt_fn = file.replace('/img/', '/gt/').replace('.jpg', '.txt')
        text_polys, text_tags = load_annoataion(txt_fn)
        img = cv2.imread(file)
        img, text_polys = random_scale(img, text_polys)
        img, text_polys, text_tags = random_crop(img, text_polys, text_tags, (15,30))
        img, text_polys = random_pad(img, text_polys)
        [conf_map, geo_map] = annoataion_to_grid(text_polys, grid_num=32)
        plot(img, text_polys, (255,255,0))


        #print(text_polys)

