import pickle
import matplotlib.pyplot as plt
def load_pkl(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
if __name__ == '__main__':
    res=[]
    std=[]
    for i in range(1,10):
        pkl = load_pkl("/home/v-chengweihu/code/CRL-change/output/grad_top1k/grad_list_0_{}.pkl".format(i))
        res.append(list(pkl.values())[0].mean().item())
        std.append(list(pkl.values())[0].std().item())
    his_str = " ".join(list(map(str, res)))
    print(his_str)
    his_str = " ".join(list(map(str, std)))
    print(his_str)