from fastai.vision import *
import torch


def bbox_to_activ(bboxes, anchors, flatten=True):
    """Return the target of the model on `anchors` for the `bboxes`."""
    if flatten:
        # x and y offsets are normalized by radius
        t_centers = (bboxes[..., :2] - anchors[..., :2]) / anchors[..., 2:]
        # Radius is given in log scale, relative to anchor radius
        t_sizes = torch.log(bboxes[..., 2:] / anchors[..., 2:] + 1e-8)
        # Finally, everything is divided by 0.1 (radii by 0.2)
        if bboxes.shape[-1] == 4:
            return torch.cat([t_centers, t_sizes], -1).div_(bboxes.new_tensor([[0.1, 0.1, 0.2, 0.2]]))
        else:
            return torch.cat([t_centers, t_sizes], -1).div_(bboxes.new_tensor([[0.1, 0.1, 0.2]]))


def create_grid(size):
    """Create a grid of a given `size`."""
    H, W = size if isinstance(size, tuple) else (size, size)
    grid = torch.FloatTensor(H, W, 2)
    linear_points = torch.linspace(-1+1/W, 1-1/W, W) if W > 1 else torch.as_tensor([0.])
    grid[:, :, 1] = torch.ger(torch.ones(H), linear_points).expand_as(grid[:, :, 0])
    linear_points = torch.linspace(-1+1/H, 1-1/H, H) if H > 1 else torch.as_tensor([0.])
    grid[:, :, 0] = torch.ger(linear_points, torch.ones(W)).expand_as(grid[:, :, 1])
    return grid.view(-1, 2)


def create_anchors(sizes, ratios, scales, flatten=True):
    """Create anchor of `sizes`, `ratios` and `scales`."""
    aspects = [[[s*math.sqrt(r), s*math.sqrt(1/r)] for s in scales] for r in ratios]
    aspects = torch.tensor(aspects).view(-1, 2)
    anchors = []
    for h, w in sizes:
        # 4 here to have the anchors overlap.
        sized_aspects = 4 * (aspects * torch.tensor([2/h, 2/w])).unsqueeze(0)
        base_grid = create_grid((h, w)).unsqueeze(1)
        n, a = base_grid.size(0), aspects.size(0)
        ancs = torch.cat([base_grid.expand(n, a, 2), sized_aspects.expand(n, a, 2)], 2)
        anchors.append(ancs.view(h, w, a, 4))
    return torch.cat([anc.view(-1, 4) for anc in anchors], 0) if flatten else anchors
    
# ADDED
def create_anchors_FasterRCNN(sizes, ratios, scales, image_size=(512, 512)):
    """Create anchors for Faster R-CNN."""
    anchors = []
    for size,size_2 in sizes:
        for ratio in ratios:
            w = size * math.sqrt(ratio)
            h = size / math.sqrt(ratio)
            for scale in scales:
                anchor_w = w * scale
                anchor_h = h * scale
                stride_w = image_size[1] // math.ceil(image_size[1] / anchor_w)
                stride_h = image_size[0] // math.ceil(image_size[0] / anchor_h)
                x_center = torch.arange(0, image_size[1], stride_w) + stride_w // 2
                y_center = torch.arange(0, image_size[0], stride_h) + stride_h // 2
                grid_x, grid_y = torch.meshgrid(x_center, y_center)
                grid_x = grid_x.reshape(-1)
                grid_y = grid_y.reshape(-1)
                anchor_x1 = grid_x - anchor_w / 2
                anchor_y1 = grid_y - anchor_h / 2
                anchor_x2 = grid_x + anchor_w / 2
                anchor_y2 = grid_y + anchor_h / 2
                anchors.append(torch.stack([anchor_x1, anchor_y1, anchor_x2, anchor_y2], dim=1))
    return torch.cat(anchors, dim=0)


def tlbr2cthw(boxes):
    """Convert top/left bottom/right format `boxes` to center/size corners."""
    center = (boxes[:, :2] + boxes[:, 2:])/2
    sizes = boxes[:, 2:] - boxes[:, :2]
    return torch.cat([center, sizes], 1)


def encode_class(idxs, n_classes):
    target = idxs.new_zeros(len(idxs), n_classes).float()
    mask = idxs != 0
    i1s = torch.LongTensor(list(range(len(idxs))))
    target[i1s[mask], idxs[mask]-1] = 1
    return target


def cthw2tlbr(boxes):
    """Convert center/size format `boxes` to top/left bottom/right corners."""
    top_left = boxes[:, :2] - boxes[:, 2:]/2
    bot_right = boxes[:, :2] + boxes[:, 2:]/2
    return torch.cat([top_left, bot_right], 1)


def intersection(anchors, targets):
    """Compute the sizes of the intersections of `anchors` by `targets`."""
    ancs, tgts = cthw2tlbr(anchors), cthw2tlbr(targets)
    a, t = ancs.size(0), tgts.size(0)
    ancs, tgts = ancs.unsqueeze(1).expand(a, t, 4), tgts.unsqueeze(0).expand(a, t, 4)
    top_left_i = torch.max(ancs[..., :2], tgts[..., :2])
    bot_right_i = torch.min(ancs[..., 2:], tgts[..., 2:])
    sizes = torch.clamp(bot_right_i - top_left_i, min=0)
    return sizes[..., 0] * sizes[..., 1]


def IoU_values(anchors, targets):
    """Compute the IoU values of `anchors` by `targets`."""
    if anchors.shape[-1] == 4:

        inter = intersection(anchors, targets)
        anc_sz, tgt_sz = anchors[:, 2] * anchors[:, 3], targets[:, 2] * targets[:, 3]
        union = anc_sz.unsqueeze(1) + tgt_sz.unsqueeze(0) - inter

        return inter / (union + 1e-8)

    else:  # circular anchors
        a, t = anchors.size(0), targets.size(0)
        ancs = anchors.unsqueeze(1).expand(a, t, 3)
        tgts = targets.unsqueeze(0).expand(a, t, 3)
        diff = (ancs[:, :, 0:2] - tgts[:, :, 0:2])
        distances = (diff ** 2).sum(dim=2).sqrt()
        radius1 = ancs[..., 2]
        radius2 = tgts[..., 2]
        acosterm1 = (((distances ** 2) + (radius1 ** 2) - (radius2 ** 2)) / (2 * distances * radius1)).clamp(-1,
                                                                                                             1).acos()
        acosterm2 = (((distances ** 2) - (radius1 ** 2) + (radius2 ** 2)) / (2 * distances * radius2)).clamp(-1,
                                                                                                             1).acos()
        secondterm = ((radius1 + radius2 - distances) * (distances + radius1 - radius2) * (
                    distances + radius1 + radius2) * (distances - radius1 + radius2)).clamp(min=0).sqrt()

        intersec = (radius1 ** 2 * acosterm1) + (radius2 ** 2 * acosterm2) - (0.5 * secondterm)

        union = np.pi * ((radius1 ** 2) + (radius2 ** 2)) - intersec

        return intersec / (union + 1e-8)


def match_anchors(anchors, targets, match_thr=0.5, bkg_thr=0.4):
    """Match `anchors` to targets. -1 is match to background, -2 is ignore."""
    ious = IoU_values(anchors, targets)
    matches = anchors.new(anchors.size(0)).zero_().long() - 2

    if ious.shape[1] > 0:
        vals, idxs = torch.max(ious, 1)
        matches[vals < bkg_thr] = -1
        matches[vals > match_thr] = idxs[vals > match_thr]
    return matches


def process_output(clas_pred, bbox_pred, anchors, detect_thresh=0.25, use_sigmoid=True):
    """Transform predictions to bounding boxes and filter results"""
    bbox_pred = activ_to_bbox(bbox_pred, anchors.to(clas_pred.device))

    if use_sigmoid:
        clas_pred = torch.sigmoid(clas_pred)

    clas_pred_orig = clas_pred.clone()
    detect_mask = clas_pred.max(1)[0] > detect_thresh
    if np.array(detect_mask.cpu()).max() == 0:
        return {'bbox_pred': None, 'scores': None, 'preds': None, 'clas_pred': clas_pred,
                'clas_pred_orig': clas_pred_orig, 'detect_mask': detect_mask}

    bbox_pred, clas_pred = bbox_pred[detect_mask], clas_pred[detect_mask]
    if bbox_pred.shape[-1] == 4:
        bbox_pred = tlbr2cthw(torch.clamp(cthw2tlbr(bbox_pred), min=-1, max=1))
    else:
        bbox_pred = bbox_pred  # torch.clamp(bbox_pred, min=-1, max=1)

    scores, preds = clas_pred.max(1)
    return {'bbox_pred': bbox_pred, 'scores': scores, 'preds': preds, 'clas_pred': clas_pred,
            'clas_pred_orig': clas_pred_orig, 'detect_mask': detect_mask}

# ADDED
def process_output_models(clas_pred, bbox_pred, scores_pred, detect_thresh=0.25, use_sigmoid=True):
    """Transform predictions to bounding boxes and filter results"""
    
    clas_pred_orig = clas_pred.clone()
    detect_mask = scores_pred > detect_thresh  

    if not detect_mask.any():
        return {'bbox_pred': None, 'scores': None, 'preds': None, 'clas_pred': clas_pred,
                'clas_pred_orig': clas_pred_orig, 'detect_mask': detect_mask}

    # Concatenate the tensors in the bbox_pred list into a single tensor
    if isinstance(bbox_pred, list):
        bbox_pred = torch.cat(bbox_pred, dim=0)
    
    # Index tensor with detect_mask
    bbox_pred = bbox_pred[detect_mask]
    
    # Repeat process for clas_pred and scores_pred
    clas_pred_filter = clas_pred[detect_mask]
    scores_pred_filter = scores_pred[detect_mask]

    scores, preds = scores_pred_filter, clas_pred_filter
    return {'bbox_pred': bbox_pred, 'scores': scores, 'preds': preds, 'clas_pred': clas_pred,
            'clas_pred_orig': clas_pred_orig, 'detect_mask': detect_mask}


def activ_to_bbox(acts, anchors, flatten=True):
    """Extrapolate bounding boxes on anchors from the model activations."""
    if flatten:
        if anchors.shape[-1] == 4:
            acts = torch.mul(acts, acts.new_tensor([[0.1, 0.1, 0.2, 0.2]]))
            centers = anchors[..., 2:] * acts[..., :2] + anchors[..., :2]
            sizes = anchors[..., 2:] * torch.exp(acts[..., 2:])
        else:
            acts = torch.mul(acts, acts.new_tensor([[0.1, 0.1, 0.2]]))
            centers = anchors[..., 2:] * acts[..., :2] + anchors[..., :2]
            sizes = anchors[..., 2:] * torch.exp(acts[..., 2:])
        return torch.cat([centers, sizes], -1)
    else:
        return [activ_to_bbox(act, anc) for act, anc in zip(acts, anchors)]

def show_anchors_on_images(data, anchors, figsize=(15,15), save=False):
    all_boxes = []
    all_labels = []
    x, y = data.one_batch(DatasetType.Train, True, True)
    for i, (image, bboxes, labels) in enumerate(zip(x, y[0], y[1])):
        image = Image(image.float().clamp(min=0, max=1))

        # 0=not found; 1=found; found 2=anchor
        processed_boxes = []
        processed_labels = []
        for gt_box in tlbr2cthw(bboxes[labels > 0]):
            matches = match_anchors(anchors, gt_box[None, :])
            bbox_mask = matches >= 0
            if bbox_mask.sum() != 0:
                bbox_tgt = anchors[bbox_mask]

                processed_boxes.append(to_np(gt_box))
                processed_labels.append(2)
                for bb in bbox_tgt:
                    processed_boxes.append(to_np(bb))
                    processed_labels.append(3)
            else:
                processed_boxes.append(to_np(gt_box))
                processed_labels.append(0)
                val, idx = torch.max(IoU_values(anchors, gt_box[None, :]), 0)
                best_fitting_anchor = anchors[idx][0]
                processed_boxes.append(to_np(best_fitting_anchor))
                processed_labels.append(1)

        all_boxes.extend(processed_boxes)
        all_labels.extend(processed_labels)

        processed_boxes = np.array(processed_boxes)
        processed_labels = np.array(processed_labels)

        _, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        ax[0].set_title("Matched Anchors")
        ax[1].set_title("No match")

        if sum(processed_labels == 2) > 0:
            imageBB = ImageBBox.create(*image.size, cthw2tlbr(tensor(processed_boxes[processed_labels > 1])),
                                           labels=processed_labels[processed_labels > 1],
                                           classes=["", "", "Match", "Anchor"], scale=False)

            image.show(ax=ax[0], y=imageBB)
        else:
            image.show(ax=ax[0])

        if sum(processed_labels == 0) > 0:
            imageBBNoMatch = ImageBBox.create(*image.size, cthw2tlbr(tensor(processed_boxes[processed_labels <= 1])),
                                                  labels=processed_labels[processed_labels <= 1],
                                                  classes=["No Match", "Anchor"], scale=False)
            image.show(ax=ax[1], y=imageBBNoMatch)
        else:
            image.show(ax=ax[1])
        if save:
            plt.savefig('anchors_{}.png'.format(i))


    return np.array(all_boxes), np.array(all_labels)


def rescale_boxes(bboxes, t_sz: torch.Tensor):
    bboxes[:, 2:] = bboxes[:, 2:] * t_sz / 2
    bboxes[:, :2] = (bboxes[:, :2] + 1) * t_sz / 2
    return bboxes


