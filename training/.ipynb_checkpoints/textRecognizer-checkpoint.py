import os
import sys
import time

s1 = time.time()
current_directory = os.getcwd()
paddle_ocr_directory = os.path.join(current_directory, "PaddleOCR")
sys.path.append(paddle_ocr_directory)

# from PaddleOCR.tools.infer_rec import main as Recog

img_path = '/home/jupyter/vamshi/pocr-pod-copy/training/6.png'
rec_model_dir = '/home/jupyter/vamshi/pocr-pod-copy/models/662ac3b0a76aebd5c9a2cdca/inference_model/model'
config_path = '/home/jupyter/vamshi/pocr-pod-copy/config.yml'

# s = time.time()
# text_results = Recog(img_path, rec_model_dir, config_path)
# e = time.time()
# print("Time", e-s, e-s1)
# print(text_results)

import gpu_infer

ls = time.time()
model, ops, config, post_process_class = gpu_infer.load_rec_model(config_path, rec_model_dir)
es =time.time()
print("Model Loaded", es - ls)
print()
print("Starting Inference")
print()
print()
si = time.time()
info = gpu_infer.inference(model, ops, config, post_process_class, img_path)
ei = time.time()
print("Inference Time:", (ei - si) * 1000)
print(info)
print()

avg_time = 0
for i in range(10):
    si = time.time()
    info = gpu_infer.inference(model, ops, config, post_process_class, img_path)
    ei = time.time()
    print("Inference Time:", (ei - si) * 1000)
    print(info)
    avg_time += (ei - si)
    print()
print("Average Inference Time:", (avg_time/10) * 1000)



