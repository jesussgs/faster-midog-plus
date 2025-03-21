import sys
from training import training
from inference import inference
from inference_FasterRCNN import inference_FasterRCNN
from inference_RetinaNet import inference_RetinaNet
from evaluation import evaluate
from aggregate import aggregate
from hydra import compose, initialize
from trainingFasterRCNN import training_FasterRCNN
from training_torch_RetinaNet import training_torch_RetinaNet

if __name__ == '__main__':
    initialize(version_base=None, config_path="configs/")
    cfg = compose(config_name="all")
    # Training
    training(cfg) # For FastAI RetinaNet
    #training_FasterRCNN(cfg) # For FasterRCNN Pytorch
    #training_torch_RetinaNet(cfg) # For RetinaNet Pytorch
    
    # Inference
    inference("/app/wandb")
    #inference_FasterRCNN("/app/wandb")
    #inference_RetinaNet("/app/wandb")
    
    # Evaluation and aggregation
    evaluate("/app/wandb")
    aggregate("/app/wandb")
