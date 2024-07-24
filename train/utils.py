from sklearn.utils import shuffle
import torch
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support,precision_recall_curve
from torch.nn.functional import relu


def get_random_pair(bags_ins_list):
    # Separate positive and negative samples
    pos_samples = [item for item in bags_ins_list if item[0] == 1]
    neg_samples = [item for item in bags_ins_list if item[0] == 0]

    # Shuffle their order
    pos_samples = shuffle(pos_samples)
    neg_samples = shuffle(neg_samples)

    # Yield pairs of positive and negative samples
    num_pairs = min(len(pos_samples), len(neg_samples))
    for i in range(num_pairs):
        pos_sample = pos_samples[i]
        neg_sample = neg_samples[i]
        yield i, pos_sample, neg_sample, pos_samples, neg_samples, num_pairs

# used for the MIL training
def epoch_train(bag_ins_list, optimizer, criterion, milnet, args):
    epoch_loss = 0
    
    if args.ranking_loss:
        
        for i, pos_sample, neg_sample, _, _, num_pairs in get_random_pair(bag_ins_list):
            optimizer.zero_grad()
            # Process positive sample
            pos_bag_label = torch.from_numpy(np.array(int(np.clip(pos_sample[0], 0, 1)))).float().cuda()
            pos_bag_feats = shuffle(pos_sample[1])
            pos_bag_feats = torch.from_numpy(np.stack(pos_bag_feats)).float().cuda()
            pos_bag_feats = pos_bag_feats[:, 0:args.num_feats]
            pos_ins_prediction, pos_bag_prediction, _, _ = milnet(pos_bag_feats)
            pos_max_prediction, _ = torch.max(pos_ins_prediction, 0)
            
            # Process negative sample
            neg_bag_label = torch.from_numpy(np.array(int(np.clip(neg_sample[0], 0, 1)))).float().cuda()
            neg_bag_feats = shuffle(neg_sample[1])
            neg_bag_feats = torch.from_numpy(np.stack(neg_bag_feats)).float().cuda()
            neg_bag_feats = neg_bag_feats[:, 0:args.num_feats]
            neg_ins_prediction, neg_bag_prediction, _, _ = milnet(neg_bag_feats)
            neg_max_prediction, _ = torch.max(neg_ins_prediction, 0)
            
            # Get the top-k predictions from pos_ins_prediction and neg_ins_prediction
            _, pos_topk_indices = torch.topk(pos_ins_prediction.view(-1), args.rank_k) 
            _, neg_topk_indices = torch.topk(neg_ins_prediction.view(-1), args.rank_k)
            
            positive_topk_ins_prediction = torch.sum(pos_ins_prediction[pos_topk_indices]) / args.rank_k
            negative_topk_ins_prediction = torch.sum(neg_ins_prediction[neg_topk_indices]) / args.rank_k
            
            # Calculate ranking loss
            loss_ranking = relu(1 + negative_topk_ins_prediction - positive_topk_ins_prediction)
            
            # Calculate other losses
            pos_bag_loss = criterion(pos_bag_prediction.view(1, -1), pos_bag_label.view(1, -1))
            pos_max_loss = criterion(pos_max_prediction.view(1, -1), pos_bag_label.view(1, -1))

            neg_bag_loss = criterion(neg_bag_prediction.view(1, -1), neg_bag_label.view(1, -1))
            neg_max_loss = criterion(neg_max_prediction.view(1, -1), neg_bag_label.view(1, -1))

            # Calculate total loss
            loss_total = args.loss_weight_bag * (pos_bag_loss + neg_bag_loss) / 2 + \
                         args.loss_weight_ins * (pos_max_loss + neg_max_loss) / 2 + \
                         args.loss_weight_ranking * loss_ranking

            loss_total.backward()
            optimizer.step()
            epoch_loss += loss_total.item()
        return epoch_loss / num_pairs
    
    else:
        for i, data in enumerate(bag_ins_list):
            optimizer.zero_grad()
            data_bag_list = shuffle(data[1])
            data_tensor = torch.from_numpy(np.stack(data_bag_list)).float().cuda()
            data_tensor = data_tensor[:, 0:args.num_feats]
            label = torch.from_numpy(np.array(int(np.clip(data[0], 0, 1)))).float().cuda()
            classes, bag_prediction, _, _ = milnet(data_tensor) # n X L
            max_prediction, index = torch.max(classes, 0)
            loss_bag = criterion(bag_prediction.view(1, -1), label.view(1, -1))
            loss_max = criterion(max_prediction.view(1, -1), label.view(1, -1))
            loss_total = args.loss_weight_bag*loss_bag + args.loss_weight_ins*loss_max
            loss_total = loss_total.mean()
            loss_total.backward()
            optimizer.step()
            epoch_loss = epoch_loss + loss_total.item()
        return epoch_loss / len(bag_ins_list)


def epoch_test(bag_ins_list, criterion, milnet, args):
    bag_labels = []
    bag_predictions = []
    epoch_loss = 0
    with torch.no_grad():
        for i, data in enumerate(bag_ins_list):
            bag_labels.append(np.clip(data[0], 0, 1))
            data_tensor = torch.from_numpy(np.stack(data[1])).float().cuda()
            data_tensor = data_tensor[:, 0:args.num_feats]
            label = torch.from_numpy(np.array(int(np.clip(data[0], 0, 1)))).float().cuda()
            classes, bag_prediction, _, _ = milnet(data_tensor) # n X L
            max_prediction, index = torch.max(classes, 0)
            loss_bag = criterion(bag_prediction.view(1, -1), label.view(1, -1))
            loss_max = criterion(max_prediction.view(1, -1), label.view(1, -1))
            loss_total = 0.5*loss_bag + 0.5*loss_max
            loss_total = loss_total.mean()
            bag_predictions.append(torch.sigmoid(bag_prediction).cpu().squeeze().numpy())
            epoch_loss = epoch_loss + loss_total.item()
    epoch_loss = epoch_loss / len(bag_ins_list)
    return epoch_loss, bag_labels, bag_predictions

def optimal_thresh(fpr, tpr, thresholds, p=0):
    """Optimal threshold for computing the possible max F1 score"""
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def five_scores(bag_labels, bag_predictions):
    """
    For evaluating the model performance
    """
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label>=threshold_optimal] = 1
    this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='binary')
    accuracy = 1- np.count_nonzero(np.array(bag_labels).astype(int)- bag_predictions.astype(int)) / len(bag_labels)
    return accuracy, auc_value, precision, recall, fscore

def compute_pos_weight(bags_list):
    "Compute the positive weight"
    pos_count = 0
    for item in bags_list:
        pos_count = pos_count + np.clip(item[0], 0, 1)
    return (len(bags_list)-pos_count)/pos_count