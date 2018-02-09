import numpy as np
import coco

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    scales,ratios = np.meshgrid(np.array(scales),np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Note that shape[0] is y and shape[1] is x
    # Get anchor centers
    x_coord = np.arange(0,shape[1],anchor_stride) * feature_stride
    y_coord = np.arange(0,shape[0],anchor_stride) * feature_stride
    x_coord, y_coord = np.meshgrid(x_coord,y_coord)

    # For every anchor, there needs (x,y,w,h), therefore, we use meshgrid to duplicate the grids

    heights, y_coord = np.meshgrid(heights,y_coord)
    widths, x_coord = np.meshgrid(widths,x_coord)

    # Get x,y,w,h for every center
    centers = np.stack([y_coord,x_coord],axis=2).reshape([-1,2])
    sizes = np.stack([heights,widths],axis=2).reshape([-1,2])

    # get y1,x1,y2,x2
    result_boxes = np.concatenate([centers-0.5*sizes,centers+0.5*sizes],axis=1)

    return result_boxes

def generate_all_anchors(scales, ratios, feature_shapes, feature_strides, anchor_strides):
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i],ratios,feature_shapes[i],feature_strides[i],anchor_strides))
    return np.concatenate(anchors,axis=0)

def get_config():
    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    return config