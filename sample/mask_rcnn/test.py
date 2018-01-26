import rcnn 
import coco

class INFConfig(coco.CocoConfig):
	IMAGES_PER_GPU = 1
config = INFConfig()

r = rcnn.MaskRCNN(config)