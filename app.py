import gradio as gr
import pandas as pd
import numpy as np
import os
import shutil
from urllib.request import urlretrieve
from tqdm.auto import tqdm
import cv2
import subprocess
from PIL import Image

def process_file(file):
    filename = os.path.basename(file.name)
    os.makedirs('1/', exist_ok=True)
    shutil.copyfile(file.name, f'1/{filename}')
    df = pd.read_excel(f'1/{filename}')

    subprocess.run(["touch", "res10_300x300_ssd_iter_140000_fp16.caffemodel"])
    proc = subprocess.Popen(["curl", "https://storage.googleapis.com/audio-transcription-saral/image-models/res10_300x300_ssd_iter_140000_fp16.caffemodel", "--output", "res10_300x300_ssd_iter_140000_fp16.caffemodel"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    proc.wait()

    subprocess.run(["touch", "deploy.prototxt"])
    proc = subprocess.Popen(["curl", "https://storage.googleapis.com/audio-transcription-saral/image-models/deploy.prototxt", "--output", "deploy.prototxt"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    proc.wait()

    detector = cv2.dnn_DetectionModel("res10_300x300_ssd_iter_140000_fp16.caffemodel", "deploy.prototxt")
    l = []
    l1 = []
    l2 = []
    for n in tqdm(range(len(df))):
        if(df['Photo'][n]!=df['Photo'][n]):
            l.append('not ok')
            l1.append('no url')
            l2.append('no')
            continue
        file = df['Photo'][n]
        try:
            urlretrieve(file,"img.png")
        except:
            l.append('not ok')
            l1.append('invalid url')
            l2.append('no')
            continue
        try:
            pil_img = Image.open("img.png")
            np_img = np.asarray(pil_img)
            detections = detector.detect(np_img)
        except:
            l.append('not ok')
            l1.append('img error')
            l2.append('yes')
            continue
        if len(detections[2]) == 0:
            l.append('not ok')
            l1.append('no face found')
            l2.append('yes')
        elif(len(detections[2])>1):
            l.append('ok')
            l1.append(f'faces found:{len(detections[2])}')
            l2.append('yes')
        else:
            l.append('ok')
            l1.append(f'score:{round(float(detections[1][0]),2)}')
            if(detections[1][0]>=0.8):
                l2.append('optional')
            else:
                l2.append('yes')
    
    df['Photo_check'] = l
    df['Photo_check_reason'] = l1
    df['qc'] = l2
    df.to_excel("output.xlsx", index=False)
    return("output.xlsx")



with gr.Blocks() as my_demo:
    gr.Markdown("Demo app to check the validation of person's profile image")
    with gr.Tab("Excel file"):
        with gr.Row():
            input = gr.File()
            output = gr.File()
        image_button = gr.Button("Process")

    image_button.click(process_file, inputs=input, outputs=output)

if __name__ == "__main__":
    my_demo.launch()