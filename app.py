from pathlib import Path
import json
try:
    import gradio as gr
except ImportError:
    gr=None
from src.dram_diag.inference import Predictor

predictor=Predictor()
def diagnose(image):
    if image is None: return {"status":"请上传图像"}
    return predictor.predict(image)
if gr:
    with gr.Blocks(title="HPOD-Classifier") as demo:
        gr.Markdown("# HPOD-Classifier\nDRAM 晶圆缺陷图像诊断原型")
        with gr.Tab("单图诊断"):
            inp=gr.Image(type="filepath",label="ROI 图像"); out=gr.JSON(label="诊断结果"); gr.Button("开始诊断").click(diagnose,inp,out)
        with gr.Tab("批量导入"):
            gr.Markdown("上传多张图像后逐张诊断；结果可在后续版本导出。")
else: demo=None
if __name__=="__main__":
    if demo is None: raise SystemExit("请先安装 gradio")
    demo.launch()
