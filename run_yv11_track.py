from ultralytics import YOLO
import argparse
from datetime import datetime
import wandb
from wandb.integration.ultralytics import add_wandb_callback
wandb.login(key="dac846d6e84dafb1a9a54a40976f97adda480161")
os.environ["WANDB_API_KEY"] = "dac846d6e84dafb1a9a54a40976f97adda480161"
# Get the current date and format it as "dd_m_yy"
current_date = datetime.now().strftime("%d_%-m_%y")
wandb.init(project='test_track',job_type='Tracking')


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolo11n.pt",help="initial weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence thr")
    parser.add_argument("--data", type=str, default='MOT17.yaml', help='datafile name')
    parser.add_argument("--source", type=str, default="../MOT17/test/MOT17-01-DPM/img1/", help="infrerence source")
    parser.add_argument("--val", type=bool, default=True, help="task is tracking with stats")
    parser.add_argument("--iou", type=float, default=0.4, help="iou thr")
    parser.add_argument("--batch", type=int, default=1, help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=1920, help="train, val image size (pixels)")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default="MOT17", help="save to project/name")
    parser.add_argument("--name", default="test_track_params", help="save to project/name")
    parser.add_argument("--save", type=bool, default='True')
    parser.add_argument("--split", type=str, default='val')
    parser.add_argument("--save_txt", type=bool, default='True', help='saving predictions labels')
    parser.add_argument("--save_conf", type=bool, default='True', help='saving predictions confidence score')
    parser.add_argument("--line_width", type=int, default=1, help='bbox line width')
    parser.add_argument("--show", type=bool, default=True)
    parser.add_argument("--tracker", type=str, default='bytetrack.yaml', help='Selecting tracker')
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
val = opt.val
batch = opt.batch
imgsz = opt.imgsz
name = opt.name+'_'+current_date
project = opt.project
source = opt.source
conf = opt.conf
split = opt.split
iou = opt.iou
data = opt.data
tracker = opt.tracker
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
    results = model.val(project=project, batch=batch, data=data, split=split, tracker=tracker, imgsz=imgsz, iou=iou,conf=conf, name=name, source=source, save=save
                  , save_txt=save_txt, save_conf=save_conf,line_width=line_width, val=val,show=show_img)
    # Access metrics
    print("\n===== Tracking Metrics =====")
    # Access metrics
    try:
        print(f"MOTA: {results.metrics.mota:.2f}")
        print(f"IDF1: {results.metrics.idf1:.2f}")
    except AttributeError:
        # Method 2: Read from saved JSON (universal fallback)
        results_file = Path("runs/detect/val/results.json")
        if results_file.exists():
            with open(results_file) as f:
                metrics = json.load(f)
            print(f"MOTA: {metrics.get('metrics/mota', 'N/A'):.2f}")
            print(f"IDF1: {metrics.get('metrics/idf1', 'N/A'):.2f}")
        else:
            print("Metrics not found. Please verify:")
            print("1. Ground truth file (gt.txt) exists and is properly formatted")
            print("2. Your MOT17.yaml points to the correct paths")
            print("3. You're using the latest Ultralytics version")
            print("4. The dataset contains enough frames for evaluation")
