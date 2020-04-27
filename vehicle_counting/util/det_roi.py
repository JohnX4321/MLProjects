import cv2,numpy as np

def get_roi_frame(curr_frame,polygon):
    mask=np.zeros(curr_frame.shape,dtype=np.uint8)
    polygon=np.array([polygon],dtype=np.int32)
    num_frame_channels=curr_frame.shape[2]
    msk_ign_color=(255,)*num_frame_channels
    cv2.fillPoly(mask,polygon,msk_ign_color)
    masked_frame=cv2.bitwise_and(curr_frame,mask)
    return masked_frame


def draw_roi(frame, polygon):
    frame_overlay = frame.copy()
    polygon = np.array([polygon], dtype=np.int32)
    cv2.fillPoly(frame_overlay, polygon, (0, 255, 255))
    alpha = 0.3
    output_frame = cv2.addWeighted(frame_overlay, alpha, frame, 1 - alpha, 0)
    return output_frame