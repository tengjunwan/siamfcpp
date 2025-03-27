from pathlib import Path
import time
import shutil

import numpy as np
import onnxruntime as ort
import cv2
from tqdm import tqdm

from kalman_filter import KalmanFilter, KalmanFilterNoNumpy


# model parameters
TEMPLATE_INPUT_SIZE = 127
SEARCH_INPUT_SIZE = 303
FAKE_TEMPLATE_INPUT_SIZE = 128  # (127 // 16 + 1)
FAKE_SEARCH_INPUT_SIZE = 304  # (303 // 16 + 1)
MODEL_OUTPUT_SIZE = 17
# INIT_BOX = (40, 30, 60, 80)   # x, y, w, h
INIT_BOX = (229, 95, 360-229, 212-95)   # x, y, w, h
# INIT_BOX = (670, 234, 145, 113)
# INIT_BOX = (554, 257, 24, 11)

# test data(images)
IMAGE_DIR = Path("/home/tengjunwan/project/ObjectTracking/SiamFC++/video_analyst-master/test_images/balloon")
IMAGE_PATHS = sorted(list(IMAGE_DIR.glob("*.png")))
VIS_SAVE_DIR = Path("/home/tengjunwan/project/ObjectTracking/SiamFC++/video_analyst-master/test_images/balloon_output")
DEBUG_SAVE_DIR = Path("/home/tengjunwan/project/ObjectTracking/SiamFC++/video_analyst-master/test_images/tmp")
DEBUG_FLAG = True
USE_KALMAN_FILTER = False

# hyper params 
CONTEXT_AMOUNT = 0.5
penalty_k = 0.08
window = np.outer(np.hanning(MODEL_OUTPUT_SIZE), np.hanning(MODEL_OUTPUT_SIZE)).flatten()  # (289=17*17,) 
window_influence = 0.2
test_lr = 0.58
SCORE_THRESH = 0.5


# state used in stmtrack_tracker
state = {"cx": 0,
         "cy": 0,
         "w": 0,
         "h": 0,
         "scale": 1.0,
         "score": 1.0}



def empty_folder(folder):
    if folder.exists() and folder.is_dir():
        shutil.rmtree(folder)  # Delete the entire folder and contents

    folder.mkdir(parents=True, exist_ok=True)  # Recreate empty folder

def get_measure_std_by_score(score):
    # higher score, lower measure_std
    # e.g., score=0.9 -> meas_std=0.01, score=0.5 -> meas_std=5
    # i wanna keep it simple first
    if score > 0.8:
        meas_std = 0.01
    elif score > 0.6:
        meas_std = 5
    else:
        meas_std = 25

    return meas_std




def next_multiple_of_16(x):
    return (x // 16 + 1) *  16


def nearest_multiple_of_16(x):
    return max(round(x / 16) * 16, 16)



def get_template_crop(state):
    size_template_crop = get_size_template_crop(state)
    # # maybe it must be multiples of 16
    size_template_crop = next_multiple_of_16(size_template_crop)
    state["scale"] = size_template_crop / TEMPLATE_INPUT_SIZE  

    crop_x = int(state["cx"] - size_template_crop * 0.5)
    crop_y = int(state["cy"] - size_template_crop * 0.5)
    crop_w = int(size_template_crop)
    crop_h = int(size_template_crop)
    return [crop_x, crop_y, crop_w, crop_h]


def get_search_crop(state):
    size_template_crop = get_size_template_crop(state)
    state["scale"] = size_template_crop / TEMPLATE_INPUT_SIZE  
    size_search_crop = SEARCH_INPUT_SIZE * state["scale"]
    # maybe it must be multiples of 16
    size_search_crop = next_multiple_of_16(size_search_crop) 

    crop_x = int(state["cx"] - size_search_crop * 0.5)
    crop_y = int(state["cy"] - size_search_crop * 0.5)
    crop_w = int(size_search_crop)
    crop_h = int(size_search_crop)
    return [crop_x, crop_y, crop_w, crop_h]


def get_size_template_crop(state):
    w, h = state["w"], state["h"]
    wc = w + CONTEXT_AMOUNT * (w + h)
    hc = h + CONTEXT_AMOUNT * (w + h)
    size_template_crop = np.sqrt(wc * hc) 
    return size_template_crop



def safe_crop(image, crop):
    """
    crop a region from the image, allowing out-of-bounds crop areas
    Args:
        image:(H,W,C) or (H,C)
        crop: [x, y, w, h]
    """
    x, y, w, h = crop

    if len(image.shape) == 3:
        cropped_area =  np.zeros((h, w, image.shape[2]), dtype=image.dtype)
    else:
        cropped_area =  np.zeros((h, w), dtype=image.dtype)

    # determine the area in the source image that overlaps with the crop
    src_x1 = max(0, x)
    src_y1 = max(0, y)
    src_x2 = min(image.shape[1], x + w)
    src_y2 = min(image.shape[0], y + h)

    # determin the area in the cropped area that receive data
    dst_x1 = max(0, -x)
    dst_y1 = max(0, -y)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    # copy
    cropped_area[dst_y1: dst_y2, dst_x1: dst_x2] = image[src_y1: src_y2, src_x1: src_x2]

    return cropped_area


def resize(image, model_size):
    resize_image = cv2.resize(image, (model_size, model_size), 
                              interpolation = cv2.INTER_NEAREST)
    return resize_image


def get_size(w, h): # for post-processing
    pad = (w + h) * 0.5
    sz = np.sqrt(((w + pad) * (h + pad)))
    return sz


def check_consistency_of_KF(kf_a, kf_b):
    consistency = True
    # check x
    x_a = kf_a.get_x()
    x_b = kf_b.get_x()
    x_ratio = np.maximum((x_a + 1e-10) / (x_b + 1e-10), (x_b + 1e-10) / (x_a + 1e-10))
    if x_ratio.max() > 1.1:
        print(f"x_ratio.max() : {x_ratio.max():.2f}")
        consistency = False

    # check x_bar
    x_bar_a = kf_a.get_x_bar()
    x_bar_b = kf_b.get_x_bar()
    x_bar_ratio = np.maximum((x_bar_a + 1e-10) / (x_bar_b + 1e-10), (x_bar_b + 1e-10) / (x_bar_a + 1e-10))
    if x_bar_ratio.max() > 1.1:
        print(f"x_bar_ratio.max() : {x_bar_ratio.max():.2f}")
        consistency = False

    # check sigma
    sigma_a = kf_a.get_sigma()
    sigma_b = kf_b.get_sigma()
    sigma_ratio = np.maximum((sigma_a + 1e-10) / (sigma_b + 1e-10), (sigma_b + 1e-10) / (sigma_a + 1e-10))
    if sigma_ratio.max() > 1.1:
        print(f"sigma_ratio.max() : {sigma_ratio.max():.2f}")
        consistency = False

     # check sigma_bar
    sigma_bar_a = kf_a.get_sigma_bar()
    sigma_bar_b = kf_b.get_sigma_bar()
    sigma_bar_ratio = np.maximum((sigma_bar_a + 1e-10) / (sigma_bar_b + 1e-10), 
                                 (sigma_bar_b + 1e-10) / (sigma_bar_a + 1e-10))
    if sigma_bar_ratio.max() > 1.1:
        print(f"sigma_bar_ratio.max() : {sigma_bar_ratio.max():.2f}")
        consistency = False

    return consistency


def get_resized_frame(state, frame):
    # resize frame to the scale of model input
    size_template_crop = get_size_template_crop(state)
    scale = size_template_crop / TEMPLATE_INPUT_SIZE 
    state["scale"] = scale
    orgin_h, orgin_w = frame.shape[:2]
    resize_h = int(orgin_h / scale)
    resize_w = int(orgin_w / scale)
    # maybe it must be multiples of 16
    # resize_h = int(nearest_multiple_of_16(resize_h)) 
    # resize_w = int(nearest_multiple_of_16(resize_w))
    resized_frame = cv2.resize(frame, (resize_w, resize_h))
    return resized_frame


def get_resized_search_crop(state):
    cx = state["cx"] / state["scale"]
    cy = state["cy"] / state["scale"]
    crop_x = int(cx - FAKE_SEARCH_INPUT_SIZE * 0.5)
    crop_y = int(cy - FAKE_SEARCH_INPUT_SIZE * 0.5)
    crop_w = int(FAKE_SEARCH_INPUT_SIZE)
    crop_h = int(FAKE_SEARCH_INPUT_SIZE)

    return [crop_x, crop_y, crop_w, crop_h]



def run_onnx(template_session, search_session):
    empty_folder(VIS_SAVE_DIR)
    if DEBUG_FLAG:
        empty_folder(DEBUG_SAVE_DIR)

    # ====part 0: crop resize and preprocess====
    # init setting(only called once)
    init_frame = cv2.imread(str(IMAGE_PATHS[0]))
    
    state["cx"] = INIT_BOX[0] + 0.5 * INIT_BOX[2]
    state["cy"] = INIT_BOX[1] + 0.5 * INIT_BOX[3]
    state["w"] = INIT_BOX[2]
    state["h"] = INIT_BOX[3]
    state["score"] = 1.0  

    if USE_KALMAN_FILTER:
        fps = 30
        dt = 1 / fps
        kf = KalmanFilter(dt, acc_std=20)
        kf_raw = KalmanFilterNoNumpy(dt, acc_std=20)
        kf.init([state["cx"], state["cy"], 0.0, 0.0])
        kf_raw.init([state["cx"], state["cy"], 0.0, 0.0])
    else:
        kf = None
    
    template_crop = get_template_crop(state)  # expand crop area according to bbox
    template_image = safe_crop(init_frame, template_crop)  # actual crop 
    template_image_resize = resize(template_image, FAKE_TEMPLATE_INPUT_SIZE) # template image resize
    if DEBUG_FLAG:
        cv2.imwrite(str(DEBUG_SAVE_DIR / IMAGE_PATHS[0].name), template_image_resize)

    # run template model
    template_image_input = np.transpose(template_image_resize, (2, 0, 1))  # (128, 128, 3) -> (3, 128, 128) 
    template_image_input = np.expand_dims(template_image_input, axis=0)  # (1, 3, 128, 128) 

    onnx_c_z_k, onnx_r_z_k = template_session.run(None, {"template": template_image_input})  # (1, 256, 4, 4), (1, 256, 4, 4)
    
    # Start timing
    start_time = time.time()
    time_per_image = []
    time_per_search = []
    # search image
    for i in tqdm(range(1, len(IMAGE_PATHS))):
        # =======kalman filter prediction=======
        if kf is not None:
            # logics for camera adjustment u=(ax, ay) according to last state
            # for now, no camera adjustment is implemented
            u = [0, 0] 
            kf_pred_cx, kf_pred_cy = kf.predict(u)
            kf_pred_cx, kf_pred_cy = kf_raw.predict(u)
            # kf_pred_cx, kf_pred_cy = kf.get_predcited_position()
            # update by prediction
            state["cx"] = kf_pred_cx
            state["cy"] = kf_pred_cy



        img_start = time.time()  # Start time for this image
        frame = cv2.imread(str(IMAGE_PATHS[i]))
        
        size_template_crop = get_size_template_crop(state)
        # target_too_big = size_template_crop > 256
        target_too_big = False
        if not target_too_big:
            # strategy: crop first then resize
            search_crop = get_search_crop(state)
            search_image = safe_crop(frame, search_crop)
            search_image_resize = resize(search_image, FAKE_SEARCH_INPUT_SIZE)
        else:
            # strategy: resize first then crop
            resized_frame = get_resized_frame(state, frame)
            resized_search_crop = get_resized_search_crop(state)
            search_image_resize = safe_crop(resized_frame, resized_search_crop)
            

        if DEBUG_FLAG:
            cv2.imwrite(str(DEBUG_SAVE_DIR / IMAGE_PATHS[i].name), search_image_resize)

        # run search model
        search_image_input = np.transpose(search_image_resize, (2, 0, 1))  # (304, 304, 3) -> (3, 304, 304) 
        search_image_input = np.expand_dims(search_image_input, axis=0)  # (1, 3, 304, 304) 
        search_start = time.time()
        onnx_score, onnx_bbox = search_session.run(None, {"search": search_image_input,
                                                          "c_z_k": onnx_c_z_k,
                                                          "r_z_k": onnx_r_z_k})   # (1, 289, 1), (1, 289, 4)
        search_end = time.time()
        time_per_search.append(search_end - search_start)

        
        # reshape
        score = onnx_score[0, :, 0]  # (289,)
        bbox = onnx_bbox[0]  # (289, 4)

        # ========post-process========
        # 1) penalty over score
        # size change
        bbox_w = bbox[..., 2] - bbox[..., 0]  # (289,)
        bbox_h = bbox[..., 3] - bbox[..., 1]
        prev_size = get_size(state["w"] / state["scale"], state["h"] / state["scale"])  
        current_size = get_size(bbox_w, bbox_h)  # (289,)
        size_change = np.maximum(prev_size / current_size, current_size / prev_size)

        # ratio change
        prev_ratio = state["w"] / state["h"]
        current_ratio = bbox_w / bbox_h
        ratio_change = np.maximum(prev_ratio / current_ratio, current_ratio / prev_ratio)

        # penalty(due to deformation)
        penalty = np.exp((1 - size_change * ratio_change) * penalty_k)  # (289,)
        
        pscore = penalty * score

        # reduce pscore by rapid position change
        pscore = pscore * (1 - window_influence) + window * window_influence

        # get best pscore id
        best_pscore_id = np.argmax(pscore)

        # 2) get bbox (update by EMA)
        # choose best pscore prediction
        bbox_best = bbox[best_pscore_id]  # (4,) xyxy

        # back to original scale
        bbox_best = bbox_best * state["scale"]

        # xyxy to cxcywh
        pred_cx = (bbox_best[2] + bbox_best[0]) * 0.5 
        pred_cx = pred_cx + state["cx"] - (SEARCH_INPUT_SIZE / 2) * state["scale"]  # back to global coordinate
        pred_cy = (bbox_best[3] + bbox_best[1]) * 0.5 
        pred_cy = pred_cy + + state["cy"] - (SEARCH_INPUT_SIZE / 2) * state["scale"]
        pred_w = bbox_best[2] - bbox_best[0]
        pred_h = bbox_best[3] - bbox_best[1]

        # update wh by EMA
        lr = penalty[best_pscore_id] * score[best_pscore_id] * test_lr
        pred_w = state["w"] * (1 - lr) + pred_w * lr
        pred_h = state["h"] * (1 - lr) + pred_h * lr
        
        # =======final results=======
        final_score = pscore[best_pscore_id]
        final_bbox = [pred_cx, pred_cy, pred_w, pred_h]

        # condidate state to update state for next frame tracking
        result_state = {}
        result_state["cx"] = pred_cx
        result_state["cy"] = pred_cy
        result_state["w"] = pred_w
        result_state["h"] = pred_h
        result_state["score"] = final_score

        is_detect = result_state["score"] > SCORE_THRESH

        # =======kalman filter correction=======
        if kf is not None:
            if is_detect:
                # meas_std = get_measure_std_by_score(result_state["score"])
                meas_std = 0.1
                kf_cor_cx, kf_cor_cy = kf.correct([result_state["cx"], result_state["cy"]], meas_std=meas_std)   
                kf_cor_cx, kf_cor_cy = kf_raw.correct([result_state["cx"], result_state["cy"]], meas_std=meas_std)   
            else:  
                kf_cor_cx, kf_cor_cy = kf.correct(None) 
                kf_cor_cx, kf_cor_cy = kf_raw.correct(None) 
        
            # kf_cor_cx, kf_cor_cy = kf.get_position()
            result_state["cx"] = kf_cor_cx
            result_state["cy"] = kf_cor_cy

            consistency = check_consistency_of_KF(kf, kf_raw)
            if not consistency:
                break

            

        # update state
        if is_detect:
            state["cx"] = result_state["cx"]
            state["cy"] = result_state["cy"]
            state["w"] = result_state["w"]
            state["h"] = result_state["h"]
            state["score"] = result_state["score"]
            

        img_end = time.time()  # End time for this image
        time_per_image.append(img_end - img_start)


        # visualization
        save_path = VIS_SAVE_DIR / IMAGE_PATHS[i].name
        frame_disp = frame.copy()
        
        if is_detect:
            # bounding box
            cv2.rectangle(frame_disp, 
                        (int(state["cx"] - state["w"] * 0.5), int(state["cy"] - state["h"] * 0.5)),
                        (int(state["cx"] + state["w"] * 0.5), int(state["cy"] + state["h"] * 0.5)), 
                        (0, 0, 255), thickness=3)
            # score
            cv2.putText(frame_disp,  f"{final_score:.2f}", 
                        (int(state["cx"] - state["w"] * 0.5), int(state["cy"] - state["h"] * 0.5)), 
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                        (0, 0, 255), thickness=3)
        # search box
        if target_too_big:
            search_crop[0] = resized_search_crop[0] * state["scale"]
            search_crop[1] = resized_search_crop[1] * state["scale"]
            search_crop[2] = resized_search_crop[2] * state["scale"]
            search_crop[3] = resized_search_crop[3] * state["scale"]
        cv2.rectangle(frame_disp, 
                (int(search_crop[0]), int(search_crop[1])),
                (int(search_crop[0] + search_crop[2]), int(search_crop[1] + search_crop[3])), 
                (123, 0, 123), thickness=2)
       
            
        if kf is not None:
            cv2.circle(frame_disp, (int(kf_pred_cx), int(kf_pred_cy)), 5, (255, 0, 0), -1)  # blue dot
            cv2.circle(frame_disp, (int(kf_cor_cx), int(kf_cor_cy)), 5, (0, 0, 255), -1)  # red dot
            
            
        cv2.imwrite(str(save_path), frame_disp)

        

    print("logic in one piece done!")
    # Print results
    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    print(f"✅ Total time: {total_time:.4f} seconds")
    print(f"⏱️ Average time per image: {sum(time_per_image) / len(time_per_image):.4f} seconds")
    print(f"⏱️ Average time per query: {sum(time_per_search) / len(time_per_image):.4f} seconds")




if __name__ == "__main__":
    # Load ONNX model(DIRECT)
    template_onnx_path = "./onnx/direct/siamfcpp_template_direct.onnx"
    search_onnx_path = "./onnx/direct/siamfcpp_search_direct.onnx"

    # Load ONNX model(OPTIMIZED)
    # template_onnx_path = "./onnx/opt/siamfcpp_template_opt.onnx"
    # search_onnx_path = "./onnx/opt/siamfcpp_search_opt.onnx"

    # # Load ONNX model(PREPROCESSED)
    # query_onnx_path = "./onnx/preprocessed/STMTrack_FeatureExtractionQuery_infer.onnx"
    # memory_onnx_path = "./onnx/preprocessed/STMTrack_FeatureExtractionMemory_infer.onnx"
    # head_onnx_path = "./onnx/preprocessed/STMTrack_ReadMemoryAndHead_infer.onnx"

    # # load ONNX model(QUANTIZED:QDQ FORMAT)
    # query_onnx_path = "./onnx/qdq/STMTrack_FeatureExtractionQuery_qdq.onnx"
    # memory_onnx_path = "./onnx/qdq/STMTrack_FeatureExtractionMemory_qdq.onnx"
    # head_onnx_path = "./onnx/qdq/STMTrack_ReadMemoryAndHead_qdq.onnx"

    template_session = ort.InferenceSession(template_onnx_path, providers=["CPUExecutionProvider"])
    search_session = ort.InferenceSession(search_onnx_path, providers=["CPUExecutionProvider"])
    
    # query_session = ort.InferenceSession(query_onnx_path, providers=["CUDAExecutionProvider"])
    # memory_session = ort.InferenceSession(memory_onnx_path, providers=["CUDAExecutionProvider"])
    # head_session = ort.InferenceSession(head_onnx_path, providers=["CUDAExecutionProvider"])
    run_onnx(template_session, search_session)






    
