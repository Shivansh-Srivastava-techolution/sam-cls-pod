from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys
import json
import yaml

current_directory = os.getcwd()
paddle_ocr_directory = os.path.join(current_directory, "PaddleOCR")
sys.path.append(paddle_ocr_directory)


os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list

def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in [".yml", ".yaml"], "only support yaml files for now"
    config = yaml.load(open(file_path, "rb"), Loader=yaml.Loader)
    return config

def merge_config(config, opts):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    for key, value in opts.items():
        if "." not in key:
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        else:
            sub_keys = key.split(".")
            assert sub_keys[0] in config, (
                "the sub_keys can only be one of global_config: {}, but get: "
                "{}, please check your running command".format(
                    config.keys(), sub_keys[0]
                )
            )
            cur = config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]
    return config

def load_rec_model(config_path, rec_model_dir, image_dir=None):
    FLAGS = {"config": config_path, "opt":{'Global.pretrained_model': rec_model_dir, 'Global.infer_img': image_dir}, "profiler_options": None}
    device = 'gpu:0'
    config = load_config(config_path)
    config = config = merge_config(config, FLAGS["opt"])
    profile_dic = {"profiler_options": None}
    config = merge_config(config, profile_dic)
    
    global_config = config["Global"]

    # build post process
    post_process_class = build_post_process(config["PostProcess"], global_config)
    
    # build model
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if config["Architecture"]["algorithm"] in [
            "Distillation",
        ]:  # distillation model
            for key in config["Architecture"]["Models"]:
                if (
                    config["Architecture"]["Models"][key]["Head"]["name"] == "MultiHead"
                ):  # multi head
                    out_channels_list = {}
                    if config["PostProcess"]["name"] == "DistillationSARLabelDecode":
                        char_num = char_num - 2
                    if config["PostProcess"]["name"] == "DistillationNRTRLabelDecode":
                        char_num = char_num - 3
                    out_channels_list["CTCLabelDecode"] = char_num
                    out_channels_list["SARLabelDecode"] = char_num + 2
                    out_channels_list["NRTRLabelDecode"] = char_num + 3
                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels_list"
                    ] = out_channels_list
                else:
                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels"
                    ] = char_num
        elif config["Architecture"]["Head"]["name"] == "MultiHead":  # multi head
            out_channels_list = {}
            char_num = len(getattr(post_process_class, "character"))
            if config["PostProcess"]["name"] == "SARLabelDecode":
                char_num = char_num - 2
            if config["PostProcess"]["name"] == "NRTRLabelDecode":
                char_num = char_num - 3
            out_channels_list["CTCLabelDecode"] = char_num
            out_channels_list["SARLabelDecode"] = char_num + 2
            out_channels_list["NRTRLabelDecode"] = char_num + 3
            config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
        else:  # base rec model
            config["Architecture"]["Head"]["out_channels"] = char_num
    model = build_model(config["Architecture"])
    
    load_model(config, model)
    
    # create data ops
    transforms = []
    for op in config["Eval"]["dataset"]["transforms"]:
        op_name = list(op)[0]
        if "Label" in op_name:
            continue
        elif op_name in ["RecResizeImg"]:
            op[op_name]["infer_mode"] = True
        elif op_name == "KeepKeys":
            if config["Architecture"]["algorithm"] == "SRN":
                op[op_name]["keep_keys"] = [
                    "image",
                    "encoder_word_pos",
                    "gsrm_word_pos",
                    "gsrm_slf_attn_bias1",
                    "gsrm_slf_attn_bias2",
                ]
            elif config["Architecture"]["algorithm"] == "SAR":
                op[op_name]["keep_keys"] = ["image", "valid_ratio"]
            elif config["Architecture"]["algorithm"] == "RobustScanner":
                op[op_name]["keep_keys"] = ["image", "valid_ratio", "word_positons"]
            else:
                op[op_name]["keep_keys"] = ["image"]
        transforms.append(op)
    global_config["infer_mode"] = True
    ops = create_operators(transforms, global_config)
    
    model.eval()
    
    print("Inference on: ", device)
    
    return model, ops, config, post_process_class

def inference(model, ops, config, post_process_class, image_dir):
    with open(image_dir, "rb") as f:
        img = f.read()
        data = {"image": img}
    batch = transform(data, ops)
    if config["Architecture"]["algorithm"] == "SRN":
        encoder_word_pos_list = np.expand_dims(batch[1], axis=0)
        gsrm_word_pos_list = np.expand_dims(batch[2], axis=0)
        gsrm_slf_attn_bias1_list = np.expand_dims(batch[3], axis=0)
        gsrm_slf_attn_bias2_list = np.expand_dims(batch[4], axis=0)

        others = [
            paddle.to_tensor(encoder_word_pos_list),
            paddle.to_tensor(gsrm_word_pos_list),
            paddle.to_tensor(gsrm_slf_attn_bias1_list),
            paddle.to_tensor(gsrm_slf_attn_bias2_list),
        ]
    if config["Architecture"]["algorithm"] == "SAR":
        valid_ratio = np.expand_dims(batch[-1], axis=0)
        img_metas = [paddle.to_tensor(valid_ratio)]
    if config["Architecture"]["algorithm"] == "RobustScanner":
        valid_ratio = np.expand_dims(batch[1], axis=0)
        word_positons = np.expand_dims(batch[2], axis=0)
        img_metas = [
            paddle.to_tensor(valid_ratio),
            paddle.to_tensor(word_positons),
        ]
    if config["Architecture"]["algorithm"] == "CAN":
        image_mask = paddle.ones(
            (np.expand_dims(batch[0], axis=0).shape), dtype="float32"
        )
        label = paddle.ones((1, 36), dtype="int64")
    images = np.expand_dims(batch[0], axis=0)
    images = paddle.to_tensor(images)
    if config["Architecture"]["algorithm"] == "SRN":
        preds = model(images, others)
    elif config["Architecture"]["algorithm"] == "SAR":
        preds = model(images, img_metas)
    elif config["Architecture"]["algorithm"] == "RobustScanner":
        preds = model(images, img_metas)
    elif config["Architecture"]["algorithm"] == "CAN":
        preds = model([images, image_mask, label])
    else:
        preds = model(images)
    post_result = post_process_class(preds)
    info = None
    if isinstance(post_result, dict):
        rec_info = dict()
        for key in post_result:
            if len(post_result[key][0]) >= 2:
                rec_info[key] = {
                    "label": post_result[key][0][0],
                    "score": float(post_result[key][0][1]),
                }
        info = json.dumps(rec_info, ensure_ascii=False)
    elif isinstance(post_result, list) and isinstance(post_result[0], int):
        # for RFLearning CNT branch
        info = str(post_result[0])
    else:
        if len(post_result[0]) >= 2:
            info = post_result[0][0] + "\t" + str(post_result[0][1])
    return info
    



