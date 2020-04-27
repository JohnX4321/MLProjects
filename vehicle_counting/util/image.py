import cv2,base64,pathlib,uuid,os
from .logger import get_logger

logger=get_logger()

def take_screenshot(frame):
    ss_dir='data/screenshots'
    pathlib.Path(ss_dir).mkdir(parents=True,exist_ok=True)
    ss_path=os.path.join(ss_dir,'img_'+uuid.uuid4().hex+'.jpg')
    cv2.imwrite(ss_path,frame,[cv2.IMWRITE_JPEG_QUALITY,85])

    logger.info('Screenshot captured.',extra={
        'meta': {'label': 'SCREENSHOT_CAPTURE', 'path': ss_path},
    })

def get_base64_image(image):
    try:
        _,img_buffer=cv2.imencode('.jpg',image)
        img_str=base64.b64encode(img_buffer).decode('utf-8')
        return 'data:image/{0}'.format(img_str)
    except:
        return None