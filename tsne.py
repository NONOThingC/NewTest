import random
import pickle
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np

def get_tsne(feature,perplexity):
    tsne = manifold.TSNE(n_components=2,n_iter=15000, init='pca', random_state=501,metric='cosine',perplexity=perplexity)#,perplexity=perplexity
    x1 = tsne.fit_transform(feature)
    # x1_min, x1_max = x1.min(0), x1.max(0)
    # x1_norm = (x1 - x1_min) / (x1_max - x1_min)
    x1_norm=x1
    return x1_norm

def tsne_plot( flag,draw_all_point=False):
    
    random.seed(501)
    
    features1=pickle.load(open(f'./{flag}_features1.pkl','rb'))
    features2=pickle.load(open(f'./{flag}_features2.pkl','rb'))
    num_p=int(1e9)
    for key, values in features1.items():
        num_p=min(len(values),num_p)
    for key, values in features2.items():
        num_p=min(len(values),num_p)
    if not draw_all_point:
        for key, values in features1.items():
            if len(values)<num_p:
                print(f'num_p is too large,cur is {len(values)}')
            else:
                values=random.sample(values,num_p)
                print(f"num_p is {num_p}")
            features1[key] = np.concatenate(values)
        
        for key, values in features2.items():
            if len(values)<num_p:
                print(f'num_p is too large,cur is {len(values)}')
            else:
                values=random.sample(values,num_p)
                print(f"num_p is {num_p}")
            features2[key] = np.concatenate(values)
        
        feat1_labels,feat1_vec=list(zip(*features1.items()))
        feat2_labels,feat2_vec=list(zip(*features2.items()))
        feat1_vec=np.concatenate(feat1_vec)
        feat2_vec=np.concatenate(feat2_vec)
        a=feat1_vec.shape[0]
        feat=np.concatenate([feat1_vec,feat2_vec])
        feat_draw = get_tsne(feat,perplexity=num_p-5)
        feat1_vec,feat2_vec=feat_draw[:a,:],feat_draw[a:,:]
        
        colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k', 'pink']
        for i, key in enumerate(feat1_labels):
            plt.scatter(feat1_vec[i*num_p:(i+1)*num_p, 0], feat1_vec[i*num_p:(i+1)*num_p, 1], c=colors[i], label='task1_' + str(key),
                        marker='.')
        for i, key in enumerate(feat2_labels):
            plt.scatter(feat2_vec[i*num_p:(i+1)*num_p, 0], feat2_vec[i*num_p:(i+1)*num_p, 1], c=colors[i + len(colors) // 2], label='task2_' + str(key),
                        marker='x')
    else:
        every_len1={}
        every_len2={}
        for key, values in features1.items():
            features1[key] = np.concatenate(values)
            every_len1[key]= len(features1[key])
        for key, values in features2.items():
            features2[key] = np.concatenate(values)
            every_len2[key]= len(features2[key])
        
        
        feat1_labels,feat1_vec=list(zip(*features1.items()))
        feat2_labels,feat2_vec=list(zip(*features2.items()))
        feat1_vec=np.concatenate(feat1_vec)
        feat2_vec=np.concatenate(feat2_vec)
        a=feat1_vec.shape[0]
        feat=np.concatenate([feat1_vec,feat2_vec])
        feat_draw = get_tsne(feat,perplexity=num_p-5)
        feat1_vec,feat2_vec=feat_draw[:a,:],feat_draw[a:,:]
        
        colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k', 'pink']
        cur1,cur2=0,0
        
        for i, key in enumerate(feat1_labels):
            plt.scatter(feat1_vec[cur1:cur1+every_len1[key], 0], feat1_vec[cur1:cur1+every_len1[key], 1], c=colors[i], label='task1_' + str(key),
                        marker='.')
            cur1+=every_len1[key]
        for i, key in enumerate(feat2_labels):
            plt.scatter(feat2_vec[cur2:cur2+every_len2[key], 0], feat2_vec[cur2:cur2+every_len2[key], 1], c=colors[i + len(colors) // 2], label='task2_' + str(key),
                        marker='x')
            cur2+=every_len2[key]
    
    if not flag:
        # plt.legend()
        plt.title('distribution after first training')
        print('first picture finished')
        plt.savefig(f'firstpicture_2-drawall_{draw_all_point}.png')
        plt.clf()
        plt.cla()
    if flag:
        # for i, (key, values) in enumerate(features1.items()):
        #     plt.scatter(values[:num_p, 0], values[:num_p, 1], c=colors[i], label='task1_' + str(key),
        #                 marker='.')
        # for i, (key, values) in enumerate(features2.items()):
        #     plt.scatter(values[:num_p, 0], values[:num_p, 1], c=colors[i + len(colors) // 2],
        #                 label='task2_' + str(key), marker='x')
        # plt.legend()
        plt.title('distribution after contrastive training')
        print('second picture finished')
        plt.savefig(f'secondpicture_2-drawall_{draw_all_point}.png')
        plt.clf()
        plt.cla()

if __name__ == '__main__':
    tsne_plot(flag=True,draw_all_point=True)