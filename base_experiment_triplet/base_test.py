import logging

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from experiment_tools import (
    DatasetSplitter,
    MobileNetV3Embedder,
    ResNet18Embedder,
    Verify,
    augumentation,
)
from pytorch_metric_learning import distances, losses, miners, reducers
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger("base_test")
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler("./base_test.log")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(levelname)s  %(asctime)s  [%(name)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def normalize_waveform(waveform):
    """
    Normalize waveform by its maximum value.
    """
    return waveform / np.abs(waveform).max()


def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()  # set the model to training mode
    for batch_idx, (data, labels) in enumerate(train_loader):  # for each batch
        data, labels = data.to(device), labels.to(device)  #
        optimizer.zero_grad()  # set the gradients to zero
        embeddings = model(data)  # forward pass
        indices_tuple = mining_func(embeddings, labels)  # get the indices of the triplets
        loss = loss_func(embeddings, labels, indices_tuple)  # compute the loss
        loss.backward()  # backward pass
        optimizer.step()  # update the parameters

        if batch_idx % 20 == 0:
            # print the loss every 20 batches
            print(
                "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                    epoch, batch_idx, loss, mining_func.num_triplets
                )
            )


def train_and_test(args):
    dataset, verify_user, gesture, enroll_N, test_id = args
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # split the dataset
    splitter = DatasetSplitter(dataset)
    X_train, y_train = splitter.get_train_data(verify_user, gesture)
    X_enroll, X_test, y_test, y_enroll = splitter.get_test_data(verify_user, gesture, enroll_N)

    # Normalize waveforms
    X_train = np.apply_along_axis(normalize_waveform, 1, X_train)
    X_enroll = np.apply_along_axis(normalize_waveform, 1, X_enroll)
    X_test = np.apply_along_axis(normalize_waveform, 1, X_test)

    # augumentation
    X_train, y_train = augumentation(X_train, y_train)

    # create the dataset
    X_train = torch.FloatTensor(X_train).unsqueeze(1)  # torch.Size([58656, 1, 512])
    y_train = torch.LongTensor(y_train)  # torch.Size([58656])
    X_enroll = torch.FloatTensor(X_enroll).unsqueeze(1)  # torch.Size([10, 1, 512])
    X_test = torch.FloatTensor(X_test).unsqueeze(1)  # torch.Size([346, 1, 512])
    X_enroll = X_enroll.to(device)
    X_test = X_test.to(device)
    Dataset = torch.utils.data.TensorDataset(X_train, y_train)

    # batch size and number of epochs
    batch_size = 1024
    num_epochs = 5

    # create the dataloader
    trainloader = DataLoader(Dataset, batch_size=batch_size, shuffle=True)

    # create the model
    # input_dim = X_test.shape[1]
    model = MobileNetV3Embedder().to(device)  # ResNet18Embedder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # create the loss function
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(
        margin=0.2, distance=distance, type_of_triplets="semihard"
    )

    # train the model
    for epoch in range(1, num_epochs + 1):
        train(model, loss_func, mining_func, device, trainloader, optimizer, epoch)
        # test the model
        model.eval()
        X_enroll_ = model(X_enroll).detach().numpy()
        X_test_ = model(X_test).detach().numpy()
        verificator = LogisticRegression(random_state=42, max_iter=5000)
        verificator.fit(X_enroll_, y_enroll)
        acc = verificator.score(X_test_, y_test)
        print("\033[31m" + "Epoch {} Accuracy: {}".format(epoch, acc) + "\033[0m")

    # test the model
    model.eval()
    X_enroll = model(X_enroll).detach().numpy()
    X_test = model(X_test).detach().numpy()
    verificator = LogisticRegression(random_state=42, max_iter=5000)
    verificator.fit(X_enroll, y_enroll)
    y_pred_sim = verificator.predict_proba(X_test)

    # ==========ログ出力==========
    logger.info("verify user: {}".format(verify_user))
    logger.info("gesture: {}".format(gesture))
    logger.info("train data shape: {}".format(X_train.shape))
    logger.info("test data shape: {}".format(X_test.shape))
    logger.info("enroll data shape: {}".format(X_enroll.shape))
    logger.info("test accuracy: {:.3f}".format(verificator.score(X_test_, y_test, optimize=True)))
    logger.info("threshold: {:.3f}".format(verificator.threshold))

    # save the result
    result = pd.DataFrame()
    result["y_pred_p"] = y_pred_sim
    result["verify_user"] = verify_user
    result["gesture"] = gesture
    result["label"] = y_test
    result["test_id"] = test_id

    del model, X_train, X_enroll, X_test
    torch.cuda.empty_cache()

    return result


if __name__ == "__main__":
    dataset = pd.read_csv("../dataset/dataset.csv")
    all_users = np.unique(dataset["user"])
    all_gestures = np.unique(dataset["gesture"])

    enroll_N = 10
    result_list = []
    test_id = 0
    for verify_user in tqdm(all_users):
        for gesture in tqdm(all_gestures, leave=True):
            test_pair = (dataset, verify_user, gesture, enroll_N, test_id)
            res = train_and_test(test_pair)
            result_list.append(res)
            test_id += 1

    result = pd.concat(result_list).reset_index(drop=True)
    save_path = "./result/result.csv"
    result.to_csv(save_path, index=False)
