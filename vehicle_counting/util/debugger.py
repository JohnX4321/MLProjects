import os,ast,cv2
from .logger import get_logger


logger=get_logger()

def mouse_callback(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        capture_pixel_pos(x,y,param['frame_width'],param['frame_height'])

def capture_pixel_pos(win_x,win_y,frame_w,frame_h):
    debug_win_size=ast.literal_eval(os.getenv('DEBUG_WINDOW_SIZE'))
    x=round((frame_w/debug_win_size[0])*win_x)
    y=round((frame_h/debug_win_size[1])*win_y)
    logger.info('Pixel pos captured.', extra={'meta': {'label': 'PIXEL_POSITION', 'position': (x, y)}})
