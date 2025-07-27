from ultralytics import YOLO
import argparse
from datetime import datetime
import os
import warnings
import wandb
#from wandb.integration.ultralytics import add_wandb_callback
#wandb.login(key="dac846d6e84dafb1a9a54a40976f97adda480161")
#os.environ["WANDB_API_KEY"] = "dac846d6e84dafb1a9a54a40976f97adda480161"
# Get the current date and format it as "dd_m_yy"
current_date = datetime.now().strftime("%d_%-m_%y")
#wandb.init(project='test_track',job_type='Tracking')
warnings.filterwarnings("ignore")

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="/work/Arg_lightly_train/finetune/Dino_3percent_500epochs_finetune_1000_epochs/Dino_3percent_1000_finetune_25_07_25/weights/best.pt",help="initial weights path")
    parser.add_argument("--cfg", type=str, default="yolo11.yaml", help="train from scratch")
    parser.add_argument("--data", type=str, default='chin_env_3_dataset.yaml', help='datafile name')
    parser.add_argument("--batch", type=int, default=1, help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=16, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default="smi_supervised_train", help="save to project/name")
    parser.add_argument("--name", default="self_supervised_Dino_3_97_best_from_3percent_model_as_started_model_Validation_on_3percent_data_test_set_", help="save to project/name")
    parser.add_argument("--save", type=bool, default='True')
    parser.add_argument("--pretrained", type=bool, default=False, help='start from pretrained or not')


    return parser.parse_known_args()[0] if known else parser.parse_args()

# define params
opt = parse_opt()
weights = opt.weights
batch = opt.batch
imgsz = opt.imgsz
#epochs = opt.epochs
name = opt.name+current_date
project = opt.project
data = opt.data
cfg = opt.cfg
device = opt.device
save = opt.save
pretrained = opt.pretrained
# Load a model
model = YOLO(weights)
print(f"\nModel loaded\n")
if __name__=='__main__':
#    add_wandb_callback(model, enable_model_checkpointing=True)
    results = model.val(project=project, batch=batch, data=data,imgsz=imgsz, name=name)#, epochs=epochs, device=device)
    # Access metrics
    print(f"\nresults: {results.results_dict}")
    print(f"\ntype of results: {type(results)}")
