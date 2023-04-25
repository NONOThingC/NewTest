# draw grad using pkl
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
def draw_grad(grad_path, save_path):
    with open(grad_path, 'rb') as f:
        grad = pickle.load(f)
    grad