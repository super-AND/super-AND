import torch
from lib.utils import AverageMeter
import time


def kNN(net, npc, trainloader, testloader, K=200, sigma=0.1, recompute_memory=False, device='cpu'):
    # set the model to evaluation mode
    net.eval()

    # tracking variables
    total = 0

    trainFeatures = npc.memory
    trainLabels = torch.LongTensor(trainloader.dataset.targets).to(device)

    # recompute features for training samples
    if recompute_memory:
        trainFeatures, trainLabels = traverse(net, trainloader, 
                                    testloader.dataset.transform, device)
    trainFeatures = trainFeatures.t()
    C = trainLabels.max() + 1
    
    # start to evaluate
    top1 = 0.
    top5 = 0.
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C.item()).to(device)
        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):

            batchSize = inputs.size(0)
            targets, inputs = targets.to(device), inputs.to(device)

            # forward
            features = net(inputs)

            # cosine similarity
            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C),
                                        yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1,1))

            top1 = top1 + correct.narrow(1,0,1).sum().item()
            top5 = top5 + correct.narrow(1,0,5).sum().item()

            total += targets.size(0)

    return top1/total


