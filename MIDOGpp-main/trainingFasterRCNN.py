import os
from torch.utils.data import SubsetRandomSampler
from slide.slide_helper import *
from slide.data_loader import *
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import json
import pickle
import torchvision
import torch
from fastai.vision import *
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
import time
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from typing import Tuple, List, Dict, Optional
from torch import Tensor
from collections import OrderedDict
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers
import albumentations as A
import utils.utils_ObjectDetection as utilsObjectDetection
from utils.stain_norm import *
from tiatoolbox.tools.stainaugment import StainAugmentor

def get_y_func(x):
    return x.y

    
    
# Modify the initialization of the model to use Faster R-CNN
def initialize_faster_rcnn_model_original(num_classes, feature_extraction = True):
    # Load the pretrained faster r-cnn model
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
                                                    
    return model
       
    
# Define your dataset class
class SlideDataset(torch.utils.data.Dataset):
    def __init__(self, item_list, patch_size, mean,std, data_transform=None, norm_transform = None, torch_transform = None):
        self.item_list = item_list
        self.patch_size = patch_size
        self.data_transform = data_transform
        self.norm_transform = norm_transform
        self.torch_transform = torch_transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        # Get Slidecontainer
        containerSlide = self.item_list[idx]
        
        # Get patch from the slide
        x,y,patch = self.extract_patch(containerSlide)
        
        # Generate target for the patch
        target = self.generate_target(x,y,patch, containerSlide)
        
        # Transform the patch if is needed
        if self.data_transform is not None:
            # Transform the patch
            transformed = self.data_transform(image=patch, bboxes=target['boxes'], labels=target['labels'])
            # Update patch and target with transformed values
            patch = transformed['image']
            target = {'boxes': transformed['bboxes'], 'labels': transformed['labels']}
        
        # Convert boxes and labels to tensors
        target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32) 
        if len(target["labels"]) == 0:
          target["labels"] = torch.tensor([0], dtype=torch.int64)
          target["boxes"] = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
          target["area"] = torch.tensor([0], dtype=torch.float32)
        else:
          target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)
          target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
            
        
        if self.torch_transform is not None:
            patch = self.torch_transform(patch)
        else:
            # Convert patch to float tensor
            patch = patch.float()
            # Normalize using mean and std
            patch = transforms.Normalize(mean=self.mean, std=self.std)(patch)
            
        # no crowd instances
        target["iscrowd"] = torch.zeros((target["boxes"].shape[0],), dtype=torch.int64)
        # id
        target["image_id"] = containerSlide.file.name           
        # Return a tuple containing a list of transformed patches and a list of targets
        return patch, target

    def extract_patch(self, containerSlide):
      # Logic to extract a patch from the slide, each time is a different patch
      x,y = containerSlide.get_new_train_coordinates()
      patch = containerSlide.get_patch(x, y)  
      return x,y,patch

    def generate_target(self, x,y, patch, containerSlide):
      # Extract bounding boxes
      bboxes, labels = containerSlide.y
      
      # Width and Height of the patch
      h, w = containerSlide.shape
  
      # Extract bounding boxes and labels
      bboxes = np.array([box for box in bboxes]) if len(np.array(bboxes).shape) == 1 else np.array(bboxes)
      # Map labels to numeric values
      label_mapping = {'mitotic figure': 1, 'hard negative': 2}
      labels = np.array([label_mapping[label] for label in labels])

      # Adjust labels and bounding boxes
      if len(labels) > 0:
          bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - x
          bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - y

          bb_widths = (bboxes[:, 2] - bboxes[:, 0]) / 2
          bb_heights = (bboxes[:, 3] - bboxes[:, 1]) / 2

          ids = ((bboxes[:, 0] + bb_widths) > 0) \
                & ((bboxes[:, 1] + bb_heights) > 0) \
                & ((bboxes[:, 2] - bb_widths) < w) \
                & ((bboxes[:, 3] - bb_heights) < h)

          bboxes = bboxes[ids]
          bboxes = np.clip(bboxes, 0, max(h, w))
          #bboxes = bboxes[:, [1, 0, 3, 2]]
          bboxes = bboxes[:, [0, 1, 2, 3]]  # to pascal_voc format

          labels = labels[ids]
      
      # In case we do not have labels on the selected patch
      if len(labels) == 0:
          labels = np.array([0])
          bboxes = np.array([[0, 0, 1, 1]])
      
      # Returning values
      target = {}
      target["boxes"] = bboxes
      target["labels"] = labels
      return target

# Ajustar collate_fn_torch
def collate_fn_torch(batch):
    return list(zip(*batch))

# Albumentations transforms    
def get_a_transforms(mean,std):
  # Define Albumentations pipeline
  tfms = A.Compose([
    A.OneOf([
        A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1),
        A.RandomRotate90(p=1),
    ], p=0.5),
    StainAugmentor(stain_matrix=get_target_stain_matrix()),
    A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.5, 0.5), contrast_limit=(-0.5, 0.5)),
    A.Affine(p=0.5, scale=(1.0, 2.0), shear=(-20, 20)),
  ], p=1, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),)

#  tfms = A.Compose([
#      A.HorizontalFlip(p=0.5),
#      A.VerticalFlip(p=0.5),
#      A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.5, 0.5), contrast_limit=(-0.5, 0.5)), 
#      A.Affine(p=0.5, scale=(1.0, 2.0), shear=(-20, 20)),
#    ], p=1, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),)
    
  return tfms
    
# Only normalized transformations    
def get_normalized_transforms(mean,std):
    tfms_pytorch = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return tfms_pytorch
  

def eval_forward(model, images, targets):
    # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            It returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).
    """
    model.eval()

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, targets)

    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                raise ValueError(
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}."
                )

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    model.rpn.training=True
    #model.roi_heads.training=True


    #####proposals, proposal_losses = model.rpn(images, features, targets)
    features_rpn = list(features.values())
    objectness, pred_bbox_deltas = model.rpn.head(features_rpn)
    anchors = model.rpn.anchor_generator(images, features_rpn)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    proposals, scores = model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

    proposal_losses = {}
    assert targets is not None
    labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors, targets)
    regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)
    loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
        objectness, pred_bbox_deltas, labels, regression_targets
    )
    proposal_losses = {
        "loss_objectness": loss_objectness,
        "loss_rpn_box_reg": loss_rpn_box_reg,
    }

    #####detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
    image_shapes = images.image_sizes
    proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(proposals, targets)
    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    result: List[Dict[str, torch.Tensor]] = []
    detector_losses = {}
    loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
    detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
    boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
    num_images = len(boxes)
    for i in range(num_images):
        result.append(
            {
                "boxes": boxes[i],
                "labels": labels[i],
                "scores": scores[i],
            }
        )
    detections = result
    detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]
    model.rpn.training=False
    model.roi_heads.training=False
    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)
    return losses, detections
    
    
# Function to obntain the validation loss of the epoch
def evaluate_loss(model, data_loader, device):
    val_loss = 0
    with torch.no_grad():
      for images, targets in data_loader:
          images = list(image.to(device) for image in images)
          targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
          losses_dict, detections = eval_forward(model, images, targets)
         
          losses = sum(loss for loss in losses_dict.values())

          val_loss += losses
          
    validation_loss = val_loss/ len(data_loader)    
    return validation_loss
    
    
# Make predictions
def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    for id in range(len(preds)) :
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']) :
            if score > threshold : #select idx which meets the threshold
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]


    return preds
    

# Calculate AP of the mitotic figures    
def calculate_AP_mitotic(model,data_loader, device):
    labels = []
    preds_adj_all = []
    annot_all = []
    
    for images, targets in data_loader:
      images = list(image.to(device) for image in images)
      for t in targets:
        labels += t['labels']

      with torch.no_grad():
          preds_adj = make_prediction(model, images, 0.3) # Detection threshold = 0.3 as MIDOG++
          preds_adj = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in preds_adj]
          preds_adj_all.append(preds_adj)
          annot_all.append(targets)
          
          
    sample_metrics = []
    for batch_i in range(len(preds_adj_all)):
        sample_metrics += utilsObjectDetection.get_batch_statistics(preds_adj_all[batch_i], annot_all[batch_i], iou_threshold=0.5)
        
    true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = utilsObjectDetection.ap_per_class(true_positives, pred_scores, pred_labels, torch.tensor(labels))
    AP = AP.float()
    mAP = torch.mean(AP)
    ap_mitotic = AP[1].item()
    print(f'mAP : {mAP}')
    print(f'AP : {AP}')
    print(f'AP-mitotic : {ap_mitotic}')
    
    return mAP, AP, ap_mitotic
    
    
def training_FasterRCNN(cfg: DictConfig):
    if torch.cuda.is_available():
        print("CUDA IS AVAILABLE")
    
    # Confirm that you have a GPU!
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tumortypes =  [tumortype for tumortype in cfg.data.tumortypes.split(",")]
    sizes = [(cfg.retinanet.sizes, cfg.retinanet.sizes)]
    ratios = [cfg.retinanet.ratios]
    scales = [float(s) for s in cfg.retinanet.scales.split(",")]

    # Load the images
    all_files, _, test_files = load_images(Path(cfg.files.image_path), cfg.files.annotation_file, level=cfg.data.level, patch_size=cfg.data.patch_size, categories=[1], tumortypes=tumortypes)
    # For specific tumortype for training
    #files = [file for file in all_files if file.tumortype == "canine lung cancer"]
    # For all tumortypes for training
    files = all_files
    test_files_names = [file.file.name for file in test_files]
    with open('/app/MIDOGpp-main/statistics_sdata.pickle', 'rb') as handle:
        statistics = pickle.load(handle)
    mean = np.array(np.mean(np.array([value for key,value in statistics['mean'].items() if tumortypes.__contains__(key)]), axis=(0,1)), dtype=np.float32)
    std = np.array(np.mean(np.array([value for key,value in statistics['std'].items() if tumortypes.__contains__(key)]), axis=(0,1)), dtype=np.float32)
                          
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    bins = pd.qcut(pd.Series([len(t.y[1]) for t in files]), 3, labels=False)
    index_kfold=1
    for train_index, val_index in skf.split(files, bins):
        cfg.update({'x-validation': {'train': json.dumps([files[i].file.name for i in train_index]), 'valid': json.dumps([files[i].file.name for i in val_index])}})
        # Log in on wandb with the API key
        wandb.login(key='use_your_key')
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                         config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True), reinit=True, name= "Kfold_FasterRCNN_"+str(index_kfold))
        train_files = [files[i] for i in train_index]
        valid_files = [files[i] for i in val_index]
        index_kfold+=1
        
        # Create transformations
        image_transforms_norm = get_normalized_transforms(mean,std)
        a_transforms = get_a_transforms(mean,std)
        
        # Create PyTorch datasets
        train_dataset = SlideDataset(train_files, patch_size= cfg.data.patch_size, mean = mean, std = std, data_transform=a_transforms, torch_transform = image_transforms_norm)
        valid_dataset = SlideDataset(valid_files, patch_size= cfg.data.patch_size, mean = mean, std = std, data_transform=None, torch_transform = image_transforms_norm)
        
        # Create PyTorch data loaders
        train_sampler = SubsetRandomSampler(indices=create_indices(train_files, cfg.data.train_patches))
        valid_sampler = SubsetRandomSampler(indices=create_indices(valid_files, cfg.data.valid_patches))
        
        train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, collate_fn=collate_fn_torch,
                                   sampler=train_sampler, num_workers=0)
        valid_loader = DataLoader(valid_dataset, batch_size=cfg.data.batch_size, collate_fn=collate_fn_torch,
                                   sampler=valid_sampler, num_workers=0)

        # TRAINING
        # Model Summary
        print("Input Size: {} x {}".format(cfg.data.patch_size, cfg.data.patch_size))
        print("Resolution Level: {}".format(cfg.data.level))
        print("Batch Size: {}".format(cfg.data.batch_size))
        print("Training Set: {} Slides with {} Patches".format(len(train_files), len(train_loader) * cfg.data.batch_size))
        print("Validation Set: {} Slides with {} Patches".format(len(valid_files), len(valid_loader) * cfg.data.batch_size))
        
        
        # Define the number of output classes based on your data
        num_classes = 2 # 0.Background + 1.Mitotic figure 
        
        # Initialize Faster R-CNN model
        model = initialize_faster_rcnn_model_original(num_classes)
        
        # Model to CUDA
        model.to(device)

        # Initialize the optimizer and scheduler as needed
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=cfg.training.lr)
                
        # Antes del bucle de entrenamiento
        best_valid_loss = float('inf')
        best_ap_mitotic = 0
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True, min_lr=cfg.training.lr)
        
        # Number of epochs
        num_epochs = 80
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            total_batches = len(train_loader)
            epoch_loss = []
            scaler=None
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
              inputs = [item.to(device) for item in inputs]
              targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
              
              with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_dict = model(inputs, targets)
                losses = sum(loss for loss in loss_dict.values())
            
              # Update values
              optimizer.zero_grad()
              losses.backward()
              optimizer.step() 
              epoch_loss.append(float(losses))
      
        
            average_loss = np.mean(epoch_loss)
            print(f"Epoch [{epoch + 1}/{cfg.training.num_epochs}], Average Loss: {average_loss:.4f}")
            
            # Validation
            validation_loss  = evaluate_loss(model, valid_loader, device=device)
            print(f"Validation Loss: {validation_loss:.4f}")
            mAP, AP, ap_mitotic = calculate_AP_mitotic(model,valid_loader,device)
            scheduler.step(np.mean(epoch_loss))
            
        
            # Log metrics to WandB
            wandb.run.summary.update({
                "epoch": epoch + 1,
                "training_loss": average_loss,
                "valid_loss": validation_loss,
                "AP-mitotic figure": ap_mitotic,
                "_runtime": int(time.time()) - run.start_time,
                "_timestamp": int(time.time()),
                "_step": epoch + 1,
                "_wandb": {"runtime": int(time.time()) - run.start_time}
            })
            
            # Get output directory
            output_dir = run.dir
            
            # In case we found new best model
            if best_ap_mitotic < ap_mitotic:
                best_ap_mitotic = ap_mitotic
                print(f"New best model obtained on epoch : {epoch+1:.1f}")
                torch.save(model.state_dict(), os.path.join(output_dir, "bestmodel.pth"))
            
            # Empty Line            
            print()

        # Finish wandb
        wandb.run.finish()
    
    # Create json file with the test files
    with open(os.path.join("/app/wandb", 'test_files.json'), 'w') as test_file_json:
        json.dump(test_files_names, test_file_json)
