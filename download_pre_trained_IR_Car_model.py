from roboflow import Roboflow
rf = Roboflow(api_key="7RlN0iyDFLAA6YoJvETh")  # Get API key from RoboFlow account
project = rf.workspace("irpeople").project("ir-car-v5ctp")
dataset = project.version(2).download("yolov11")  # or "yolov11" if available