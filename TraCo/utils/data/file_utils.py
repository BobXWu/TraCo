import os
import argparse
import yaml
import argparse


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def read_yaml(path):
    with open(path) as file:
        config = yaml.safe_load(file)
    return config


def update_args(args, path, key=None):
    config = read_yaml(path)
    if config:
        args = vars(args)
        if key:
            args[key] = config
        else:
            args.update(config)
        args = restructure_as_namespace(args)
    return args


def restructure_as_namespace(args):
    if not isinstance(args, dict):
        return args
    for key in args:
        args[key] = restructure_as_namespace(args[key])
    args = argparse.Namespace(**args)
    return args


def read_text(path):
    texts = list()
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            texts.append(line.strip())
    return texts


def save_text(texts, path, strip=True):
    with open(path, 'w', encoding='utf-8') as file:
        for text in texts:
            if strip:
                file.write(text.strip() + '\n')
            else:
                file.write(text + '\n')
