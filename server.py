from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import sys
import argparse
from socket import *
import shutil
import torch.distributed
import tqdm
from pnasnet import pnasnet5large
import torchvision.models as models
import os
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from transforms import get_transforms

classes = {0: 403,
           1: 404,
           2: 484,
           3: 510,
           4: 147,
           5: 625,
           6: 628,
           7: 701,
           8: 833,
           9: 895
           }
valclasses = {}
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--traindata', default='./train', type=str,
                    help='path to dataset')
parser.add_argument('--newdata', default='./new', type=str,
                    help='path to dataset')
parser.add_argument('--tempdata', default='./temp', type=str,
                    help='path to dataset')
parser.add_argument('--valdata', default='./val', type=str,
                    help='path to dataset')
parser.add_argument('--model_path', default='./PNASNet.pth', type=str,
                    help='path to dataset')
parser.add_argument('--input_size', default=480, type=int,
                    help='path to dataset')
parser.add_argument('--batch_size', default=16, type=int,
                    help='path to dataset')
parser.add_argument('--workers', default=20, type=int,
                    help='path to dataset')
parser.add_argument('--num_classes', default=0, type=int,
                    help='number of classes')

args = parser.parse_args()


# 预测模块
def predict():
    torch.manual_seed(0)
    np.random.seed(0)

    print("Create data loaders", flush=True)
    print("Input size : " + str(args.input_size))
    print("Model : fixpnasnet5large")
    # 载入数据
    transformation = get_transforms(input_size=args.input_size, test_size=args.input_size,
                                    kind='full', crop=True, need=('train', 'val'), backbone='pnasnet5large')
    transform_test = transformation['val']
    test_set = datasets.ImageFolder(args.tempdata, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=(args.workers - 1),
    )
    # 载入模型
    model = pnasnet5large(pretrained='imagenet')
    pretrained_dict = torch.load(args.model_path, map_location='cpu')['model']
    model_dict = model.state_dict()
    count = 0
    count2 = 0
    for k in model_dict.keys():
        count = count + 1.0
        if (('module.' + k) in pretrained_dict.keys()):
            count2 = count2 + 1.0
            model_dict[k] = pretrained_dict.get(('module.' + k))
    model.load_state_dict(model_dict)
    print("load " + str(count2 * 100 / count) + " %")

    assert int(count2 * 100 / count) == 100, "model loading error"

    for name, child in model.named_children():
        for name2, params in child.named_parameters():
            params.requires_grad = False

    print('model_load')
    if torch.cuda.is_available():
        model.cuda(0)
    model.eval()
    predlist = []
    with torch.no_grad():
        for data in tqdm.tqdm(test_loader):
            images, labels = data
            images = images.cuda(0, non_blocking=True)
            outputs = model(images)
            pred = torch.argmax(outputs, dim=1)
            npred = pred.cpu().numpy()
            predlist.extend(list(npred))
    return predlist




def train(traindic, valdic):
    model = models.resnet18(pretrained=False)
    # print(model.state_dict().keys())
    fc_features = model.fc.in_features
    num_classes = len(traindic)
    model.fc = nn.Linear(fc_features, num_classes)
    transformation = get_transforms(input_size=224, test_size=224,
                                    kind='full', crop=True, need=('train', 'val'))
    transform_train = transformation['train']
    transform_test = transformation['val']
    train_dataset = datasets.ImageFolder(args.traindata, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_dataset = datasets.ImageFolder(args.valdata + 'val', transform=transform_test)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=2)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    print('using gpu:', 1)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        net = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
    best_acc = 0.85  # 2 初始化best test accuracy
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    for epoch in range(1, 101):  # 从先前次数开始训练
        print('\nEpoch: %d' % (epoch + 1))  # 输出当前次数
        net.train()
        sum_loss = 0.0  # 损失数量
        correct = 0.0  # 准确数量
        total = 0.0  # 总共数量
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            co = correct.numpy().tolist()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f '
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100 * co / total))
        # 每训练完一个epoch测试一下准确率
        print("Waiting Test!")
        predlist = []
        labelist = []
        with torch.no_grad():  # 没有求导
            for data in valloader:
                net.eval()  # 运用net.eval()时，由于网络已经训练完毕，参数都是固定的，因此每个min-batch的均值和方差都是不变的，因此直接运用所有batch的均值和方差。
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                outputs = net(images)
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)
                npred = predicted.cpu().numpy()
                predlist.extend(list(npred))
                nlabel = labels.cpu().numpy()
                labelist.extend(nlabel)
            pred = [traindic[j] for j in predlist]
            label = [valdic[j] for j in labelist]
            total = len(pred)
            co=0
            for i in range(total):
                if pred[i]==label[i]:
                    co += 1
            print('测试分类准确率为：%.3f%%' % (100. * co / total))
            acc = 100. * co / total
            if acc > best_acc:
                print('Saving model......')
                torch.save(net.state_dict(), './model/resnet18.pth')
                best_acc = acc
    print("Training Finished, TotalEPOCH=%d" % 100)
    with open('acc.txt','a') as f:
        f.writelines(best_acc)






#云端通信模块
#文件发送
# def sendfile(filename,dataset_dir):
#     path=os.path.join(dataset_dir,filename)
#     if os.path.exists(path):
#         tcpCliSock.send(filename.encode())
#         check = tcpCliSock.recv(BUFSIZ)
#         filesize = str(os.path.getsize(path))
#         print("文件大小为：", filesize)
#         tcpCliSock.send(filesize.encode())
#         data = tcpCliSock.recv(BUFSIZ)# 挂起服务器发送，确保客户端单独收到文件大小数据，避免粘包
#         print("开始发送")
#         f = open(path, "rb")
#         for line in f:
#             tcpCliSock.send(line)
#     else:
#         tcpCliSock.send("0001".encode())  # 如果文件不存在，那么就返回该代码
# #文件接收
# def recvfile(store_path):
#     data = tcpCliSock.recv(BUFSIZ)
#     name = data.decode()
#     # receive picture
#     tcpCliSock.send("File name received".encode())
#     data = tcpCliSock.recv(BUFSIZ)
#     tcpCliSock.send("File size received".encode())
#     name = os.path.join(store_path, name)
#     file_total_size = int(data.decode())
#     received_size = 0
#     f = open(name, "wb")
#     while received_size < file_total_size:
#         data = tcpCliSock.recv(BUFSIZ)
#         f.write(data)
#         received_size += len(data)
#         print("已接收:", received_size)
#     f.close()
#     print("receive done", file_total_size, " ", received_size)
# #文件夹发送
# def senddata(path):
#     tcpCliSock.send(str(len(os.listdir(path))).encode())
#     data = tcpCliSock.recv(BUFSIZ)
#     k = os.listdir(path)
#     for each in k:
#         # send picture
#         sendfile(each,path)
#         data = tcpCliSock.recv(BUFSIZ)
#     tcpCliSock.send("传输完成".encode())
# #文件夹接收
# def recvdata(store_path):
#     data = tcpCliSock.recv(BUFSIZ)
#     num = data.decode()
#     tcpCliSock.send("number received".encode())
#     for i in range(int(num)):
#         recvfile(store_path)
#         tcpCliSock.send("file received".encode())
#     data = tcpCliSock.recv(BUFSIZ)
#发送与接收数据流
def communicate(vadic):
    #recvdata(args.temp_dir)
    new_images = os.listdir(args.newdata)
    for i in range(len(new_images)//300):
        #抽取300张新图
        temp_images = []
        print('从客户端收到共300张未识别图片，分别为：')
        for j in range(300):
            shutil.copy(args.newdata+new_images[300*i+j], args.tempdata+'/tp/'+new_images[300*i+j])
            print(new_images[300*i+j])
            temp_images.append(new_images[300*i+j])
        temp_images.sort()
        #标注
        print('开始打上标签')
        labels = predict()
        #加入原数据库
        oldclass = os.listdir(args.traindata)
        for j in range(300):
            if str(labels[j]) not in oldclass:
                os.makedirs(args.traindata+'/'+str(labels[j]))
                oldclass.append(str(labels[j]))
            shutil.move(args.tempdata + '/tp/' + temp_images[j], args.traindata+'/'+str(labels[j]))
        #移除小于50张的类别
        for each in oldclass:
            if len(os.listdir(args.traindata+'/'+each)) < 50:
                shutil.move(args.traindata+'/'+each, args.tempdata)
        cla = os.listdir(args.traindata)
        cla.sort()
        args.num_classes = len(cla)
        print('可识别类别数目为：', args.num_classes)
        traindic={}
        for i in range(args.num_classes):
            traindic[i] = cla[i]
        print(traindic)
        print(vadic)
        train(traindic,vadic)
    #senddata(args.log_dir)
    num_classes = str(args.num_classes)
    #tcpCliSock.send(num_classes.encode())

HOST = '10.130.5.35'  #对bind（）方法的标识，表示可以使用任何可用的地址
PORT = 21567  #设置端口
BUFSIZ = 1024  #设置缓存区的大小
ADDR = (HOST, PORT)
tcpSerSock = socket(AF_INET, SOCK_STREAM)  #定义了一个套接字
tcpSerSock.bind(ADDR)  #绑定地址
tcpSerSock.listen(5)     #规定传入连接请求的最大数，异步的时候适用
if_update_model=0




while True:
    va = os.listdir(args.traindata)
    va.sort()
    vadic = {}
    for i in range(va):
        vadic[i] = va[i]
    #print('waiting for connection...')
    #tcpCliSock, addr = tcpSerSock.accept()
    #print ('...connected from:', addr)
    #while True:
    communicate(vadic)
    #tcpCliSock.close()
#tcpSerSock.close()
