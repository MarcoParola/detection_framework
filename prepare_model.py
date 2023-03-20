import hydra
import os
import numpy as np
import matplotlib.pyplot as plt



@hydra.main(config_path="./config/", config_name="config")
def train(cfg):

    print(cfg)


if __name__ == '__main__':
    train()