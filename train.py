import hydra
import os
import numpy as np
import matplotlib.pyplot as plt
import os




@hydra.main(config_path="./config/", config_name="config")
def train(cfg):


    print(cfg)
    if cfg.model == 'yolo':
        os.system('python ' + cfg.project_path + '/test.py')


if __name__ == '__main__':
    train()