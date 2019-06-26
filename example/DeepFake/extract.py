import sul_tool
import os

if not os.path.exists('./imgs/'):
	os.mkdir('./imgs/')

sul_tool.extract_frames('a_Trim.mp4','./imgs/vid')
