import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from utils.iotools import save_checkpoint
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
from loss.softmax_loss import CrossEntropyLabelSmooth
from tqdm import tqdm
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io

import concurrent.futures
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pandas as pd


def do_train_stage2(cfg,
             model,
             center_criterion,
             train_loader_stage1,
             train_loader_stage2,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank,num_classes):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("TFCLIP.train")
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info('start training')
    
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    acc_meter_id1 = AverageMeter()
    acc_meter_id2 = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent_frame = CrossEntropyLabelSmooth(num_classes=num_classes)

    @torch.no_grad()
    def generate_cluster_features(labels, features):
        import collections
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0)
        return centers

    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    #######   1.CLIP-Memory module ####################
    print("=> Automatically generating CLIP-Memory (might take a while, have a coffe)")
    image_features = []
    labels = []
    
    feature_dir = os.path.join(cfg.OUTPUT_DIR, 'clip_memory_features')
    os.makedirs(feature_dir, exist_ok=True)

    with torch.no_grad():
        for n_iter, (img, vid, target_cam, target_view) in enumerate(tqdm(train_loader_stage1, desc="Processing", unit="batch")):
            img = img.to(device)  # torch.Size([64, 4, 3, 256, 128])
            target = vid.to(device)  # torch.Size([64])
            
            feature_file = os.path.join(feature_dir, f'feature_{n_iter}.pt')
            
            if os.path.exists(feature_file):
                # Load existing features
                batch_features = torch.load(feature_file)
                print(f"Loading feature {feature_file}")
            else:
                # Generate and save new features
                if len(img.size()) == 6:
                    # method = 'dense'
                    b, n, s, c, h, w = img.size()
                    assert (b == 1)
                    img = img.view(b * n, s, c, h, w)  # torch.Size([5, 8, 3, 256, 128])
                    with amp.autocast(enabled=True):
                        image_feature = model(img, get_image = True)
                        image_feature = image_feature.view(-1, image_feature.size(1))
                        image_feature = torch.mean(image_feature, 0, keepdim=True)  # 1,512
                        batch_features = image_feature
                else:
                    with amp.autocast(enabled=True):
                        batch_features = model(img, get_image = True)
                
                # Save features
                torch.save(batch_features, feature_file)
                print(f"Saving feature {feature_file}")
            
            # Append features and labels
            for i, img_feat in zip(target, batch_features):
                labels.append(i)
                image_features.append(img_feat.cpu())

        labels_list = torch.stack(labels, dim=0).cuda()  # N torch.Size([8256])
        image_features_list = torch.stack(image_features, dim=0).cuda()  # torch.Size([8256, 512])

    cluster_features = generate_cluster_features(labels_list.cpu().numpy(), image_features_list).detach()
    best_performance = 0.0
    best_cmc = 0.0
    best_mAP = 0.0
    best_epoch = 1

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        acc_meter_id1.reset()
        acc_meter_id2.reset()
        evaluator.reset()

        model.train()

        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else:
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            with amp.autocast(enabled=True):
                B, T, C, H, W = img.shape  # B=64, T=4.C=3 H=256,W=128
                score, feat, logits1 = model(x = img, cam_label=target_cam, view_label=target_view, text_features2=cluster_features)
                score1 = score[0:3]
                score2 = score[3]

                if (n_iter + 1) % log_period == 0:
                    loss1 = loss_fn(score1, feat, target, target_cam, logits1, isprint=True)
                else:
                    loss1 = loss_fn(score1, feat, target, target_cam, logits1)

                targetX = target.unsqueeze(1)  # 12,1   => [94 94 10 10 15 15 16 16 75 75 39 39]
                targetX = targetX.expand(B, T)
                # 12,8  => [ [94...94][94...94][10...10][10...10] ... [39...39] [39...39]]
                targetX = targetX.contiguous()
                targetX = targetX.view(B * T,
                                       -1)  # 96  => [94...94 10...10 15...15 16...16 75...75 39...39]
                targetX = targetX.squeeze(1)
                loss_frame = xent_frame(score2, targetX)
                loss = loss1 + loss_frame / T


            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc1 = (logits1.max(1)[1] == target).float().mean()
            acc_id1 = (score[0].max(1)[1] == target).float().mean()
            acc_id2 = (score[3].max(1)[1] == targetX).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc1, 1)
            acc_meter_id1.update(acc_id1, 1)
            acc_meter_id2.update(acc_id2, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc_clip: {:.3f}, Acc_id1: {:.3f}, Acc_id2: {:.3f}, Base Lr: {:.2e}"
                    .format(epoch, (n_iter + 1), len(train_loader_stage2),
                            loss_meter.avg, acc_meter.avg, acc_meter_id1.avg, acc_meter_id2.avg, scheduler.get_lr()[0]))

        scheduler.step()

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch))


        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else:
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else:
                                target_view = None
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10, 20]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        if cfg.MODEL.SIE_CAMERA:
                            camids = camids.to(device)
                        else:
                            camids = None
                        if cfg.MODEL.SIE_VIEW:
                            target_view = target_view.to(device)
                        else:
                            target_view = None
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10, 20]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()
            prec1 = cmc[0] + mAP
            is_best = prec1 > best_performance
            best_performance = max(prec1, best_performance)
            if is_best:
                best_epoch = epoch
                best_cmc = cmc
                best_mAP = mAP
                
            save_checkpoint(model.state_dict(), is_best, os.path.join(cfg.OUTPUT_DIR, 'checkpoint_ep.pth.tar'))

    logger.info("==> Best Perform {:.1%}, achieved at epoch {}".format(best_performance, best_epoch))
    logger.info("==> Best mAP {:.1%}, achieved at epoch {}".format(best_mAP, best_epoch))
    logger.info("Best CMC curve:")
    for r in [1, 5, 10, 20]:
        logger.info("Rank-{:<3}:{:.1%}".format(r, best_cmc[r - 1]))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)


def do_inference_dense(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("TFCLIP.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        img = img.to(device)  # torch.Size([64, 4, 3, 256, 128])
        if len(img.size()) == 6:
            # method = 'dense'
            b, n, s, c, h, w = img.size()
            assert (b == 1)
            img = img.view(b * n, s, c, h, w)  # torch.Size([5, 8, 3, 256, 128])

        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else:
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            feat = feat.view(-1, feat.size(1))
            feat = torch.mean(feat, 0, keepdim=True)  # 1,512
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)


    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]

def do_inference_rrs(cfg,
                     model,
                     val_loader,
                     num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")
    
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    
    evaluator.reset()
    
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            # model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        img = img.to(device)  # torch.Size([64, 4, 3, 256, 128])
        if len(img.size()) == 6:
            # method = 'dense'
            b, n, s, c, h, w = img.size()
            assert (b == 1)
            img = img.view(b * n, s, c, h, w)  # torch.Size([5, 8, 3, 256, 128])
        
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else:
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            # feat = feat.view(-1, feat.size(1))
            # feat = torch.mean(feat, 0, keepdim=True)  # 1,512
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)
    
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]

def do_inference_rrs_visualize(cfg, model, val_loader, num_query, 
                               visualize=True, sort="ascending", num_vis=10, rank_vis=10, output_dir="./vis_results"):
 
    output_feat = os.path.join(cfg.OUTPUT_DIR, output_dir)+"_feats"
    
    os.makedirs(output_feat, exist_ok=True)

    output_mat = os.path.join(cfg.OUTPUT_DIR, output_dir)+"_mat"
    os.makedirs(output_feat, exist_ok=True)

    output_dir = os.path.join(cfg.OUTPUT_DIR, output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")
    
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    
    model = model.to(device)
    model.eval()

    # Lists to store tracklet folder names
    query_tracklet_names = []
    gallery_tracklet_names = []

    for n_iter, (img, pids, camids, camids_batch, viewids, img_paths) in enumerate(tqdm(val_loader)):
    
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids_batch = camids_batch.to(device)
            else:
                camids_batch = None
            if cfg.MODEL.SIE_VIEW:
                viewids = viewids.to(device)
            else:
                viewids = None
            
            feat_path= os.path.join(output_feat, f'feat_{n_iter}.pt')
            if os.path.exists(feat_path):
                feat = torch.load(feat_path)
                # print(f"Loading feature {feat_path}")
            else:
                feat = model(img, cam_label=camids_batch, view_label=viewids)
                torch.save(feat, feat_path)
                # print(f"Saving feature {feat_path}")
            # feat = model(img, cam_label=camids_batch, view_label=viewids)
            
            
            # Extract tracklet folder names from img_paths
            for img_path in img_paths:
                # Extract tracklet folder name from the first image path in each sequence
                tracklet_name = os.path.basename(os.path.dirname(img_path[0]))
                
                # If n_iter < num_query, it's a query tracklet, otherwise it's a gallery tracklet
                if n_iter < num_query:
                    query_tracklet_names.append(tracklet_name)
                else:
                    gallery_tracklet_names.append(tracklet_name)
            
            
            evaluator.update_v2((feat, pids, camids, img_paths))

    cmc, mAP,  all_AP, distmat, pids, camids, qf, gf, q_pids, g_pids, q_camids, g_camids, q_img_paths, g_img_paths  = evaluator.compute_v2()

    print("Shape of distmat:", distmat.shape)
    print("Length of pids:", len(pids))
    print("Length of camids:", len(camids))
    print("Shape of qf:", qf.shape)
    print("Shape of gf:", gf.shape)
    print("Length of q_pids:", len(q_pids))
    print("Length of g_pids:", len(g_pids))
    print("Length of q_camids:", len(q_camids))
    print("Length of g_camids:", len(g_camids))
    print("Length of q_img_paths:", len(q_img_paths))
    print("Length of g_img_paths:", len(g_img_paths))
    print("Length of query_tracklet_names:", len(query_tracklet_names))
    print("Length of gallery_tracklet_names:", len(gallery_tracklet_names))

    # Extract tracklet folder names from image paths
    query_tracklet_names = []
    gallery_tracklet_names = []
    
    for paths in q_img_paths:
        # Extract folder name from the first path in the sequence
        folder_name = os.path.basename(os.path.dirname(paths[0]))
        query_tracklet_names.append(folder_name)
    
    for paths in g_img_paths:
        # Extract folder name from the first path in the sequence
        folder_name = os.path.basename(os.path.dirname(paths[0]))
        gallery_tracklet_names.append(folder_name)

    # Save distmat, pids, and camids
    save_path = os.path.join(output_mat, "evaluation_results.mat")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results = {
        'distance_matrix': distmat,
        'query_pids': q_pids,
        'gallery_pids': g_pids,
        'query_camids': q_camids,
        'gallery_camids': g_camids,
        'query_tracklet_names': np.array(query_tracklet_names, dtype=np.object_),
        'gallery_tracklet_names': np.array(gallery_tracklet_names, dtype=np.object_)
    }
    scipy.io.savemat(save_path, results)
    print(f"Results saved to {save_path}")

    # Create a separate CSV file directly
    csv_path = os.path.join(output_mat, "tracklet_rankings.csv")
    
    # Create DataFrame
    results = []
    
    # For each query, find the top-10 gallery tracklets
    for query_idx in range(len(query_tracklet_names)):
        # Get distances for this query
        distances = distmat[query_idx]
        
        # Sort indices by distance (ascending)
        sorted_indices = np.argsort(distances)
        
        # Get top-10 gallery tracklets
        top_gallery_indices = sorted_indices[:10]
        
        # Get the actual tracklet names
        query_tracklet = query_tracklet_names[query_idx]
        gallery_tracklets = " ".join([gallery_tracklet_names[idx] for idx in top_gallery_indices])
        
        # Add to results
        results.append({
            "row_id": query_idx + 1,  # 1-indexed for row_id
            "query_tracklet": query_tracklet,
            "gallery_tracklets": gallery_tracklets
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"CSV results saved to {csv_path}")

    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))

    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "result.txt")

    with open(output_file_path, "w") as f:
        # Write mAP to the result file
        map_str = "mAP: {:.1%}".format(mAP)
        logger.info(map_str)
        f.write(map_str + "\n")
        
        # Write CMC curve results to the result file
        for r in [1, 5, 10, 20]:
            result_str = "CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1])
            logger.info(result_str)
            f.write(result_str + "\n")
    print(f"Results saved to {output_file_path}")

    return cmc[0], cmc[4]

def load_image(path):
    return cv2.imread(path)
