import argparse
from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
import cv2
import onnxruntime as ort
import torch.nn.functional as F
import torch.quantization.quantize_fx as quantize_fx
from tqdm import tqdm

from videoanalyst.config.config import cfg, specify_task
from videoanalyst.model import builder as model_builder

# model parameters
TEMPLATE_INPUT_SIZE = 127
SEARCH_INPUT_SIZE = 303
FAKE_TEMPLATE_INPUT_SIZE = 128  # (127 // 16 + 1)
FAKE_SEARCH_INPUT_SIZE = 304  # (303 // 16 + 1)
MODEL_OUTPUT_SIZE = 17

# test data(images)
IMAGE_DIR = Path("/home/tengjunwan/project/ObjectTracking/SiamFC++/video_analyst-master/test_images/polo")
IMAGE_PATHS = sorted(list(IMAGE_DIR.glob("*.jpg")))
VIS_SAVE_DIR = Path("/home/tengjunwan/project/ObjectTracking/SiamFC++/video_analyst-master/test_images/polo_output")

# hyper params for post-process
# search_area_factor = 4.0
CONTEXT_AMOUNT = 0.5
penalty_k = 0.08
window = np.outer(np.hanning(MODEL_OUTPUT_SIZE), np.hanning(MODEL_OUTPUT_SIZE)).flatten()  # (289=17*17,) 
window_influence = 0.21
test_lr = 0.58


def make_parser():
    parser = argparse.ArgumentParser(
        description="press s to select the target box,\n \
                        then press enter or space to confirm it or press c to cancel it,\n \
                        press c to stop track and press q to exit program")
    parser.add_argument(
        "-cfg",
        "--config",
        default="experiments/siamfcpp/test/vot/siamfcpp_alexnet.yaml",
        type=str,
        help='experiment configuration')

    return parser


def get_model(args):  # reference to demo/main/video/sot_video.py
    root_cfg = cfg
    root_cfg.merge_from_file(args.config)

    # resolve config
    root_cfg = root_cfg.test
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()
    # build model
    model = model_builder.build(task, task_cfg.model)

    return model


def xcorr_depthwise(x, kernel):
    r"""
    Depthwise cross correlation. e.g. used for template matching in Siamese tracking network

    Arguments
    ---------
    x: torch.Tensor
        feature_x (e.g. search region feature in SOT)
    kernel: torch.Tensor
        feature_z (e.g. template feature in SOT)

    Returns
    -------
    torch.Tensor
        cross-correlation result
    """
    batch = int(kernel.size(0))
    channel = int(kernel.size(1))
    x = x.view(1, int(batch * channel), int(x.size(2)), int(x.size(3)))
    kernel = kernel.view(batch * channel, 1, int(kernel.size(2)),
                         int(kernel.size(3)))
    out = F.conv2d(x, kernel, groups=batch * channel)
    out = out.view(batch, channel, int(out.size(2)), int(out.size(3)))
    return out


class TemplateEmbedding(nn.Module):
    """wrapper of siamfc++: template imbedding"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, template_img):  # true 127 as input size
        template_img = template_img.float()
        f_z = self.model.basemodel(template_img)
        # template as kernel
        c_z_k = self.model.c_z_k(f_z)
        r_z_k = self.model.r_z_k(f_z)
        return c_z_k, r_z_k
    

class TemplateEmbeddingV2(nn.Module):
    """wrapper of siamfc++: template imbedding"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, template_img):  # fake 128 as input size
        template_img = template_img.float()
        template_img = template_img[:, :, :127, :127]
        f_z = self.model.basemodel(template_img)
        # template as kernel
        c_z_k = self.model.c_z_k(f_z)
        r_z_k = self.model.r_z_k(f_z)
        return c_z_k, r_z_k
    
class TemplateEmbeddingV3(nn.Module):
    """wrapper of siamfc++: template imbedding"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, template_img):  # fake 128 as input size, (1, 128, 128, 3)
        # template_img = template_img.permute(0, 3, 1, 2)
        template_img = template_img[:, :, :127, :127]
        f_z = self.model.basemodel(template_img)
        # template as kernel
        c_z_k = self.model.c_z_k(f_z)
        r_z_k = self.model.r_z_k(f_z)
        return c_z_k, r_z_k
        

class SearchTracing(nn.Module):
    """wrapper of siamfc++: searching"""
    def __init__(self, model):
        super().__init__()
        self.model = model
  
    def forward(self, search_img, c_z_k, r_z_k):  # true 303 as input size
        search_img = search_img.float()

        # backbone feature
        f_x = self.model.basemodel(search_img)

        # feature adjustment
        c_x = self.model.c_x(f_x)
        r_x = self.model.r_x(f_x)

        # feature matching
        r_out = xcorr_depthwise(r_x, r_z_k)
        c_out = xcorr_depthwise(c_x, c_z_k)

        # head
        fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.model.head(
            c_out, r_out, search_img.size(-1))
        
        # apply sigmoid
        fcos_cls_prob_final = torch.sigmoid(fcos_cls_score_final)
        fcos_ctr_prob_final = torch.sigmoid(fcos_ctr_score_final)

        # apply centerness correction
        fcos_score_final = fcos_cls_prob_final * fcos_ctr_prob_final
  
        return fcos_score_final, fcos_bbox_final
    

class SearchTracingV2(nn.Module):
    """wrapper of siamfc++: searching"""
    def __init__(self, model):
        super().__init__()
        self.model = model
  
    def forward(self, search_img, c_z_k, r_z_k):  # fake 304 as input size
        search_img = search_img.float()

        # search_img = F.interpolate(search_img, size=(303, 303), mode='nearest')
        search_img = search_img[:, :, :303, :303]

        # backbone feature
        f_x = self.model.basemodel(search_img)

        # feature adjustment
        c_x = self.model.c_x(f_x)
        r_x = self.model.r_x(f_x)

        # feature matching
        r_out = xcorr_depthwise(r_x, r_z_k)
        c_out = xcorr_depthwise(c_x, c_z_k)

        # head
        fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.model.head(
            c_out, r_out, search_img.size(-1))
        
        # apply sigmoid
        fcos_cls_prob_final = torch.sigmoid(fcos_cls_score_final)
        fcos_ctr_prob_final = torch.sigmoid(fcos_ctr_score_final)

        # apply centerness correction
        fcos_score_final = fcos_cls_prob_final * fcos_ctr_prob_final
  
        return fcos_score_final, fcos_bbox_final
    

class SearchTracingV3(nn.Module):
    """wrapper of siamfc++: searching"""
    def __init__(self, model):
        super().__init__()
        self.model = model
  
    def forward(self, search_img, c_z_k, r_z_k):  # fake 304 as input size
        # search_img = search_img.float()
        # search_img = search_img.permute(0, 3, 1, 2)

        # search_img = F.interpolate(search_img, size=(303, 303), mode='nearest')
        search_img = search_img[:, :, :303, :303]

        # backbone feature
        f_x = self.model.basemodel(search_img)

        # feature adjustment
        c_x = self.model.c_x(f_x)
        r_x = self.model.r_x(f_x)

        # feature matching
        r_out = xcorr_depthwise(r_x, r_z_k)
        c_out = xcorr_depthwise(c_x, c_z_k)

        # head
        fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.model.head(
            c_out, r_out, search_img.size(-1))
        
        # apply sigmoid
        fcos_cls_prob_final = torch.sigmoid(fcos_cls_score_final)
        fcos_ctr_prob_final = torch.sigmoid(fcos_ctr_score_final)

        # apply centerness correction
        fcos_score_final = fcos_cls_prob_final * fcos_ctr_prob_final
  
        return fcos_score_final, fcos_bbox_final



def next_multiple_of_16(x):
    return (x // 16 + 1) *  16


def get_template_crop(state, context_amount):
    w, h = state["w"], state["h"]
    wc = w + context_amount * (w + h)
    hc = h + context_amount * (w + h)
    size_template_crop = np.sqrt(wc * hc)
    # # maybe it must be multiples of 16
    size_template_crop = next_multiple_of_16(size_template_crop)
    state["scale"] = size_template_crop / TEMPLATE_INPUT_SIZE  

    crop_x = int(state["cx"] - size_template_crop * 0.5)
    crop_y = int(state["cy"] - size_template_crop * 0.5)
    crop_w = int(size_template_crop)
    crop_h = int(size_template_crop)
    return [crop_x, crop_y, crop_w, crop_h]

def get_search_crop(state, context_amount):
    w, h = state["w"], state["h"]
    wc = w + context_amount * (w + h)
    hc = h + context_amount * (w + h)
    size_template_crop = np.sqrt(wc * hc) 
    state["scale"] = size_template_crop / TEMPLATE_INPUT_SIZE  
    size_search_crop = SEARCH_INPUT_SIZE * state["scale"]
    # maybe it must be multiples of 16
    size_search_crop = next_multiple_of_16(size_search_crop) 

    crop_x = int(state["cx"] - size_search_crop * 0.5)
    crop_y = int(state["cy"] - size_search_crop * 0.5)
    crop_w = int(size_search_crop)
    crop_h = int(size_search_crop)
    return [crop_x, crop_y, crop_w, crop_h]



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








# state used in stmtrack_tracker
state = {
    "cx": 0,
    "cy": 0,
    "w": 0,
    "h": 0,
    "scale": 0}



def logic_in_one_piece(model):
    template_embedding = TemplateEmbeddingV2(model)
    search_tracing = SearchTracingV2(model)  

    # ============================model inference logics in one piece=======================

    # ====part 0: crop resize and preprocess====
    # init setting(only called once)
    init_frame = cv2.imread(str(IMAGE_PATHS[0]))
    init_xywh = (670, 234, 145, 113) # xy means top left
    state["cx"] = init_xywh[0] + 0.5 * init_xywh[2]
    state["cy"] = init_xywh[1] + 0.5 * init_xywh[3]
    state["w"] = init_xywh[2]
    state["h"] = init_xywh[3]
    
    
    template_crop = get_template_crop(state, context_amount=CONTEXT_AMOUNT)  # expand crop area according to bbox
    template_image = safe_crop(init_frame, template_crop)  # actual crop 
    
    # memory image resize
    template_image_resize = resize(template_image, FAKE_TEMPLATE_INPUT_SIZE)

    # run tempate embedding model
    with torch.no_grad():
        template_image_input = torch.from_numpy(template_image_resize).cuda()  # (128, 128, 3)
        template_image_input = template_image_input.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 128, 128)
        c_z_k, r_z_k = template_embedding(template_image_input)  # (1, 256, 4, 4), (1, 256, 4, 4)

    # query image
    for i in tqdm(range(1, len(IMAGE_PATHS)), total=len(IMAGE_PATHS) - 1):
    # for i in range(1, 5):
        frame = cv2.imread(str(IMAGE_PATHS[i]))
        search_crop = get_search_crop(state, context_amount=CONTEXT_AMOUNT)
        search_image = safe_crop(frame, search_crop)
        search_image_resize = resize(search_image, FAKE_SEARCH_INPUT_SIZE)

        # run search model
        with torch.no_grad():
            search_image_input = torch.from_numpy(search_image_resize).cuda()  # (304, 304, 3)
            search_image_input = search_image_input.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 304, 304)
            score, bbox = search_tracing(search_image_input, c_z_k, r_z_k)  # (1, 289, 1), (1, 289, 4)

        
        # reshape
        score = score[0, :, 0]  # (289,)
        bbox = bbox[0]  # (289, 4)

        # back to numpy
        score = score.detach().cpu().numpy()
        bbox = bbox.detach().cpu().numpy()

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
        pred_cy = pred_cy + state["cy"] - (SEARCH_INPUT_SIZE / 2) * state["scale"]
        pred_w = bbox_best[2] - bbox_best[0]
        pred_h = bbox_best[3] - bbox_best[1]

        # update wh by EMA
        lr = penalty[best_pscore_id] * score[best_pscore_id] * test_lr
        pred_w = state["w"] * (1 - lr) + pred_w * lr
        pred_h = state["h"] * (1 - lr) + pred_h * lr
        
        # =======final results=======
        final_score = pscore[best_pscore_id]
        final_bbox = [pred_cx, pred_cy, pred_w, pred_h]

        # udpate state for next frame tracking
        state["cx"] = pred_cx
        state["cy"] = pred_cy
        state["w"] = pred_w
        state["h"] = pred_h


        # visualization
        save_path = VIS_SAVE_DIR / IMAGE_PATHS[i].name
        frame_disp = frame.copy()
        
        cv2.rectangle(frame_disp, 
                      (int(state["cx"] - state["w"] * 0.5), int(state["cy"] - state["h"] * 0.5)),
                      (int(state["cx"] + state["w"] * 0.5), int(state["cy"] + state["h"] * 0.5)), 
                      (0, 0, 255), thickness=3)
        cv2.putText(frame_disp,  f"{final_score:.2f}", 
                    (int(state["cx"] - state["w"] * 0.5), int(state["cy"] - state["h"] * 0.5)), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                    (0, 0, 255), thickness=3)
        cv2.imwrite(str(save_path), frame_disp)
        # print("done")

    print("logic in one piece done!")


def convertToONNX_templateV2(model):
    template_embedding = TemplateEmbeddingV2(model)
    input_names = ["template"]
    output_names = ["c_z_k", "r_z_k"]
    dummy_template = torch.randint(0, 256, 
            (1, 3, FAKE_TEMPLATE_INPUT_SIZE, FAKE_TEMPLATE_INPUT_SIZE)).to(torch.uint8).cuda()
    torch.onnx.export(
        template_embedding,
        (dummy_template),
        "./onnx/direct/siamfcpp_template_direct.onnx",
        export_params=True,
        opset_version=12,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
    )


def convertToONNX_searchV2(model):
    template_embedding = SearchTracingV2(model)
    input_names = ["search", "c_z_k", "r_z_k"]
    output_names = ["score", "bbox"]
    dummy_search = torch.randint(0, 256, 
            (1, 3, FAKE_SEARCH_INPUT_SIZE, FAKE_SEARCH_INPUT_SIZE)).to(torch.uint8).cuda()
    dummy_c_z_k = torch.randn(1, 256, 4, 4).cuda()
    dummy_r_z_k = torch.randn(1, 256, 4, 4).cuda()
    torch.onnx.export(
        template_embedding,
        (dummy_search, dummy_c_z_k, dummy_r_z_k),
        "./onnx/direct/siamfcpp_search_direct.onnx",
        export_params=True,
        opset_version=12,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
    )

def convertToONNX_templateV3(model):
    template_embedding = TemplateEmbeddingV3(model)
    input_names = ["template"]
    output_names = ["c_z_k", "r_z_k"]
    dummy_template = torch.randint(0, 256, 
            (1, 3, FAKE_TEMPLATE_INPUT_SIZE, FAKE_TEMPLATE_INPUT_SIZE)).to(torch.float32).cuda()
    torch.onnx.export(
        template_embedding,
        (dummy_template),
        "./onnx/direct/siamfcpp_template_direct_aipp.onnx",
        export_params=True,
        opset_version=12,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
    )


def convertToONNX_searchV3(model):
    template_embedding = SearchTracingV3(model)
    input_names = ["search", "c_z_k", "r_z_k"]
    output_names = ["score", "bbox"]
    dummy_search = torch.randint(0, 256, 
            (1, 3, FAKE_SEARCH_INPUT_SIZE, FAKE_SEARCH_INPUT_SIZE)).to(torch.float32).cuda()
    dummy_c_z_k = torch.randn(1, 256, 4, 4).to(torch.float16).cuda()
    dummy_r_z_k = torch.randn(1, 256, 4, 4).to(torch.float16).cuda()
    torch.onnx.export(
        template_embedding,
        (dummy_search, dummy_c_z_k, dummy_r_z_k),
        "./onnx/direct/siamfcpp_search_direct_aipp_half.onnx",
        export_params=True,
        opset_version=12,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
    )


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    model = get_model(args)
    model.eval()
    model.cuda()

    # logic_in_one_piece(model)

    convert_flag = True
    if convert_flag:
        # print("converting siamfcpp_template.onnx...")
        # convertToONNX_templateV2(model)
        # print("done.")

        # print("converting siamfcpp_search.onnx...")
        # convertToONNX_searchV2(model)
        # print("done.")

        print("converting siamfcpp_template.onnx...")
        convertToONNX_templateV3(model)
        print("done.")

        print("converting siamfcpp_search.onnx...")
        convertToONNX_searchV3(model)
        print("done.")

        pass




    print("done")