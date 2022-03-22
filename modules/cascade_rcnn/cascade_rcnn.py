
from .fpn import _FPN

import torch
import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class detnet(_FPN):
    def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False):
        self.model_path = 'data/pretrained_model/detnet59.pth'
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.dout_base_model = 256

        _FPN.__init__(self, classes, class_agnostic)

    def _init_modules(self):
        detnet = detnet59()

        if self.pretrained == True:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path, map_location=lambda storage, loc: storage)
            detnet.load_state_dict({k: v for k, v in state_dict.items() if k in detnet.state_dict()})

        self.RCNN_layer0 = nn.Sequential(detnet.conv1, detnet.bn1, detnet.relu, detnet.maxpool)
        self.RCNN_layer1 = nn.Sequential(detnet.layer1)
        self.RCNN_layer2 = nn.Sequential(detnet.layer2)
        self.RCNN_layer3 = nn.Sequential(detnet.layer3)
        self.RCNN_layer4 = nn.Sequential(detnet.layer4)
        self.RCNN_layer5 = nn.Sequential(detnet.layer5)  # add one layer, for c6

        # Top layer
        self.RCNN_toplayer = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # reduce channel, for p6

        # Smooth layers
        self.RCNN_smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # for p3
        self.RCNN_smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # for p2
        # self.RCNN_smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.RCNN_latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # for c5
        self.RCNN_latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # for c4
        self.RCNN_latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # for c3
        self.RCNN_latlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)  # for c2

        self.RCNN_top = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=cfg.POOLING_SIZE, stride=cfg.POOLING_SIZE, padding=0),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True)
        )

        self.RCNN_top_2nd = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=cfg.POOLING_SIZE, stride=cfg.POOLING_SIZE, padding=0),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True)
        )

        self.RCNN_top_3rd = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=cfg.POOLING_SIZE, stride=cfg.POOLING_SIZE, padding=0),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True)
        )

        self.RCNN_cls_score = nn.Linear(1024, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(1024, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(1024, 4 * self.n_classes)

        self.RCNN_cls_score_2nd = nn.Linear(1024, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred_2nd = nn.Linear(1024, 4)
        else:
            self.RCNN_bbox_pred_2nd = nn.Linear(1024, 4 * self.n_classes)

        self.RCNN_cls_score_3rd = nn.Linear(1024, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred_3rd = nn.Linear(1024, 4)
        else:
            self.RCNN_bbox_pred_3rd = nn.Linear(1024, 4 * self.n_classes)

        # Fix blocks
        for p in self.RCNN_layer0[0].parameters(): p.requires_grad = False
        for p in self.RCNN_layer0[1].parameters(): p.requires_grad = False

        assert (0 <= cfg.DETNET.FIXED_BLOCKS < 4)
        if cfg.DETNET.FIXED_BLOCKS >= 3:
            for p in self.RCNN_layer3.parameters(): p.requires_grad = False
        if cfg.DETNET.FIXED_BLOCKS >= 2:
            for p in self.RCNN_layer2.parameters(): p.requires_grad = False
        if cfg.DETNET.FIXED_BLOCKS >= 1:
            for p in self.RCNN_layer1.parameters(): p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.RCNN_layer0.apply(set_bn_fix)
        self.RCNN_layer1.apply(set_bn_fix)
        self.RCNN_layer2.apply(set_bn_fix)
        self.RCNN_layer3.apply(set_bn_fix)
        self.RCNN_layer4.apply(set_bn_fix)
        self.RCNN_layer5.apply(set_bn_fix)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.RCNN_layer0.eval()
            self.RCNN_layer1.eval()
            self.RCNN_layer2.train()
            self.RCNN_layer3.train()
            self.RCNN_layer4.train()
            self.RCNN_layer5.train()

            self.RCNN_smooth1.train()
            self.RCNN_smooth2.train()

            self.RCNN_latlayer1.train()
            self.RCNN_latlayer2.train()
            self.RCNN_latlayer3.train()
            self.RCNN_latlayer4.train()

            self.RCNN_toplayer.train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.RCNN_layer0.apply(set_bn_eval)
            self.RCNN_layer1.apply(set_bn_eval)
            self.RCNN_layer2.apply(set_bn_eval)
            self.RCNN_layer3.apply(set_bn_eval)
            self.RCNN_layer4.apply(set_bn_eval)
            self.RCNN_layer5.apply(set_bn_eval)

    def _head_to_tail(self, pool5):
        block5 = self.RCNN_top(pool5)
        fc7 = block5.mean(3).mean(2)
        return fc7

    def _head_to_tail_2nd(self, pool5):
        block5 = self.RCNN_top_2nd(pool5)
        fc7 = block5.mean(3).mean(2)
        return fc7

    def _head_to_tail_3rd(self, pool5):
        block5 = self.RCNN_top_3rd(pool5)
        fc7 = block5.mean(3).mean(2)
        return fc7