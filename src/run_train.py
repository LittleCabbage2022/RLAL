import argparse
import yaml
from .train_jointly import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    train(cfg)
