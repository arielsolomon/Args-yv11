from ultralytics import YOLO
import argparse
from datetime import datetime
import wandb
from wandb.integration.ultralytics import add_wandb_callback


# Get the current date and format it as "dd_m_yy"
current_date = datetime.now().strftime("%d_%-m_%y")
wandb.init(project='test_track',job_type='Tracking')


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolo11n.pt",help="initial weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence thr")
    parser.add_argument("--source", type=str, default="../MOT17/MOT17/test/MOT17-07-DPM/img1/", help="infrerence source")
    parser.add_argument("--iou", type=float, default=0.4, help="iou thr")
    parser.add_argument("--batch", type=int, default=1, help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=1920, help="train, val image size (pixels)")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default="test_MOT17_07_DPM", help="save to project/name")
    parser.add_argument("--name", default="test_track_params", help="save to project/name")
    parser.add_argument("--save", type=bool, default='True')
    parser.add_argument("--save_txt", type=bool, default='True', help='saving predictions labels')
    parser.add_argument("--save_conf", type=bool, default='True', help='saving predictions confidence score')
    parser.add_argument("--line_width", type=int, default=1, help='bbox line width')
    parser.add_argument("--show", type=bool, default=True)
    parser.add_argument("--visualize", type=bool, default=True)
        # Logger arguments
    # parser.add_argument("--entity", default=None, help="Entity")
    # parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='Upload data, "val" option')
    # parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")
    # parser.add_argument("--artifact_alias", type=str, default="latest", help="Version of dataset artifact to use")

    # # NDJSON logging
    # parser.add_argument("--ndjson-console", action="store_true", help="Log ndjson to console")
    # parser.add_argument("--ndjson-file", action="store_true", help="Log ndjson to file")

    return parser.parse_known_args()[0] if known else parser.parse_args()

# define params
opt = parse_opt()
weights = opt.weights
batch = opt.batch
imgsz = opt.imgsz
name = opt.name+'_'+current_date
project = opt.project
source = opt.source
conf = opt.conf
iou =opt.iou
save = opt.save
save_txt = opt.save_txt
save_conf = opt.save_conf
line_width = opt.line_width
show_img = opt.show
visualize = opt.visualize
# Load a model
model = YOLO(weights)
if __name__=='__main__':
    add_wandb_callback(model, enable_model_checkpointing=True)
    model.track(project=project, batch=batch, imgsz=imgsz, iou=iou,conf=conf, name=name, source=source, save=save
                  , save_txt=save_txt, save_conf=save_conf,line_width=line_width, show=show_img)
    wandb.finish()
