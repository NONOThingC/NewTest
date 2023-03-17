from dataloaders.sampler import data_sampler
from dataloaders.data_loader import get_data_loader
from .model import Encoder
from .utils import Moment, dot_dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm, trange
from sklearn.cluster import KMeans
from .utils import osdist
from .retrieve_module import RetrievePool
import itertools
import copy
import collections
from sklearn import manifold
import matplotlib.pyplot as plt
import pickle
import os
class Manager(object):
    def __init__(self, args):
        super().__init__()
        self.id2rel = None
        self.rel2id = None
        
    def get_tsne(self,feature,perplexity):
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501,metric='cosine',n_iter=15000,perplexity=perplexity)#,perplexity=perplexity
        x1 = tsne.fit_transform(feature)
        # x1_min, x1_max = x1.min(0), x1.max(0)
        # x1_norm = (x1 - x1_min) / (x1_max - x1_min)
        x1_norm=x1
        return x1_norm

    def tsne_plot(self,args, encoder, tokens_task1, tokens_task2, flag, num_points=50,draw_all_point=True):
        
        ####
        data_loader1 = get_data_loader(args, tokens_task1, shuffle=False, drop_last=False, batch_size=1)
        encoder.eval()
        features1 = {}
        random.seed(501)
        for step, batch_data in enumerate(data_loader1):
            labels, tokens, ind = batch_data
            tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
            with torch.no_grad():
                feature= encoder.bert_forward(tokens)[1].cpu()
                labels = int(labels)
            if labels not in features1:
                features1[labels] = [feature]
            else:
                features1[labels].append(feature)
        pickle.dump(features1,open( os.path.join(args.output_path,f'./{flag}_features1_drawall_{draw_all_point}.pkl'),'wb'))
        
        data_loader2 = get_data_loader(args, tokens_task2, shuffle=False, drop_last=False, batch_size=1)
        features2 = {}
        for step, batch_data in enumerate(data_loader2):
            labels, tokens, ind = batch_data
            tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
            with torch.no_grad():
                feature= encoder.bert_forward(tokens)[1].cpu()
                labels = int(labels)
            if labels not in features2:
                features2[labels] = [feature]
            else:
                features2[labels].append(feature)
        
        pickle.dump(features2,open(os.path.join(args.output_path,f'./{flag}_features2_drawall_{draw_all_point}.pkl'),'wb'))
        
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
            feat_draw = self.get_tsne(feat,perplexity=num_p-5)
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
            feat_draw = self.get_tsne(feat,perplexity=num_p-5)
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
            plt.savefig(os.path.join(args.output_path,f'firstpicture_2-drawall_{draw_all_point}.png'))
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
            plt.savefig(os.path.join(args.output_path,f'secondpicture_2-drawall_{draw_all_point}.png'))
            plt.clf()
            plt.cla()
        
    def cab_proto(self, args, encoder, mem_set,old_proto):
        # aggregate the prototype set for further use.
        data_loader = get_data_loader(args, mem_set, False, False, 1)

        features = []

        encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, ind = batch_data
            tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
            with torch.no_grad():
                feature, rep= encoder.bert_forward(tokens)
            features.append(feature)
            self.lbs.append(labels.item())
        features = torch.cat(features, dim=0)
        proto = torch.mean(features, dim=0, keepdim=True)
        alpha=0.2
        proto=alpha*old_proto+(1-alpha)*proto.mean(dim=0, keepdim=True)
        return proto, features
    def get_proto(self, args, encoder, mem_set):
        # aggregate the prototype set for further use.
        data_loader = get_data_loader(args, mem_set, False, False, 1)

        features = []

        encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, ind = batch_data
            tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
            with torch.no_grad():
                feature, rep= encoder.bert_forward(tokens)
            features.append(feature)
            self.lbs.append(labels.item())
        features = torch.cat(features, dim=0)

        proto = torch.mean(features, dim=0, keepdim=True)

        return proto, features
    # Use K-Means to select what samples to save, similar to at_least = 0
    def select_data(self, args, encoder, sample_set):
        data_loader = get_data_loader(args, sample_set, shuffle=False, drop_last=False, batch_size=1)
        features = []
        encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, ind = batch_data
            tokens=torch.stack([x.to(args.device) for x in tokens],dim=0)
            with torch.no_grad():
                feature, rp = encoder.bert_forward(tokens)
            features.append(feature.detach().cpu())
        
        features = np.concatenate(features)
        num_clusters = min(args.num_protos, len(sample_set))
        distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)

        mem_set = []
        current_feat = []
        for k in range(num_clusters):
            sel_index = np.argmin(distances[:, k])
            instance = sample_set[sel_index]
            mem_set.append(instance)
            current_feat.append(features[sel_index])
        
        current_feat = np.stack(current_feat, axis=0)
        current_feat = torch.from_numpy(current_feat)
        return mem_set, current_feat, current_feat.mean(0)
    
    # Use K-Means to select what samples to save, similar to at_least = 0
    def select_data_by_retrieve(self, args, encoder, sample_set):
        data_loader = get_data_loader(args, sample_set, shuffle=False, drop_last=False, batch_size=1)
        features = []
        encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, ind = batch_data
            tokens=torch.stack([x.to(args.device) for x in tokens],dim=0)
            with torch.no_grad():
                feature, rp = encoder.bert_forward(tokens)
            features.append(feature.detach().cpu())
        
        features = np.concatenate(features)
        num_clusters = min(args.num_protos, len(sample_set))
        distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)

        mem_set = []
        current_feat = []
        for k in range(num_clusters):
            sel_index = np.argmin(distances[:, k])
            instance = sample_set[sel_index]
            mem_set.append(instance)
            current_feat.append(features[sel_index])
        
        current_feat = np.stack(current_feat, axis=0)
        current_feat = torch.from_numpy(current_feat)
        return mem_set, current_feat, current_feat.mean(0)
    
    def get_embedding(self, args, encoder, sample_set):
        data_loader = get_data_loader(args, sample_set, shuffle=False, drop_last=False, batch_size=args.batch_size)
        features = []
        labelss=[]
        indss=[]
        encoder.eval()
        for step, batch_data in enumerate(data_loader):
            labels, tokens, ind = batch_data
            tokens=torch.stack([x.to(args.device) for x in tokens],dim=0)
            with torch.no_grad():
                feature, rp = encoder.bert_forward(tokens)
            features.append(feature.detach().cpu())
            labelss.append(labels)
            indss.append(ind)
        indss=np.concatenate(indss)
        labelss=np.concatenate(labelss)
        features = np.concatenate(features)
        return features,indss,labelss
    
    def get_optimizer(self, args, encoder):
        print('Use {} optim!'.format(args.optim))
        def set_param(module, lr, decay=0):
            parameters_to_optimize = list(module.named_parameters())
            no_decay = ['undecay']
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr},
                {'params': [p for n, p in parameters_to_optimize
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': lr}
            ]
            return parameters_to_optimize
        params = set_param(encoder, args.learning_rate)

        if args.optim == 'adam':
            pytorch_optim = optim.Adam
        else:
            raise NotImplementedError
        optimizer = pytorch_optim(
            params
        )
        return optimizer
    
    def train_simple_model(self, args, encoder, training_data, epochs):

        data_loader = get_data_loader(args, training_data, shuffle=True)
        encoder.train()

        optimizer = self.get_optimizer(args, encoder)
        def train_data(data_loader_, name = "", is_mem = False):
            losses = []
            td = tqdm(data_loader_, desc=name)
            for step, batch_data in enumerate(td):
                optimizer.zero_grad()
                labels, tokens, ind = batch_data
                labels = labels.to(args.device)
                tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
                hidden, reps = encoder.bert_forward(tokens)
                loss = self.moment.loss(reps, labels)
                losses.append(loss.item())
                td.set_postfix(loss = np.array(losses).mean())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                # update moemnt
                if is_mem:
                    self.moment.update_mem(ind, reps.detach())
                else:
                    self.moment.update(ind, reps.detach())
            print(f"{name} loss is {np.array(losses).mean()}")
        for epoch_i in range(epochs):
            train_data(data_loader, "init_train_{}".format(epoch_i), is_mem=False)
            
    def train_mem_model(self, args, encoder, mem_data, proto_mem, epochs, seen_relations):
        # history_nums = len(seen_relations) - args.rel_per_task
        # if len(proto_mem)>0:
            
        #     proto_mem = F.normalize(proto_mem, p =2, dim=1)
        #     dist = dot_dist(proto_mem, proto_mem)
        #     dist = dist.to(args.device)

        mem_loader = get_data_loader(args, mem_data, shuffle=True)
        encoder.train()
        # temp_rel2id = [self.rel2id[x] for x in seen_relations]
        # map_relid2tempid = {k:v for v,k in enumerate(temp_rel2id)}
        # map_tempid2relid = {k:v for k, v in map_relid2tempid.items()}
        optimizer = self.get_optimizer(args, encoder)
        def train_data(data_loader_, name = "", is_mem = False):
            # losses = []
            # kl_losses = []
            td = tqdm(data_loader_, desc=name)
            for step, batch_data in enumerate(td):

                optimizer.zero_grad()
                labels, tokens, ind = batch_data
                labels = labels.to(args.device)
                tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
                zz, reps = encoder.bert_forward(tokens)
                # hidden = reps


                # need_ratio_compute = ind < history_nums * args.num_protos
                # total_need = need_ratio_compute.sum()
                
                # if total_need >0 :
                #     # Knowledge Distillation for Relieve Forgetting
                #     need_ind = ind[need_ratio_compute]
                #     need_labels = labels[need_ratio_compute]
                #     temp_labels = [map_relid2tempid[x.item()] for x in need_labels]
                #     gold_dist = dist[temp_labels]#完全训练之前的
                #     current_proto = self.moment.get_mem_proto()[:history_nums]
                #     this_dist = dot_dist(hidden[need_ratio_compute], current_proto.to(args.device))
                #     loss1 = self.kl_div_loss(gold_dist, this_dist, t=args.kl_temp)
                #     loss1.backward(retain_graph=True)
                # else:
                #     loss1 = 0.0

                #  Contrastive Replay
                # cl_loss = self.moment.loss(reps, labels, is_mem=True, mapping=map_relid2tempid)
                cl_loss = self.moment.prototypical_loss(reps, labels, is_mem=True)
                # if isinstance(loss1, float):
                #     kl_losses.append(loss1)
                # else:
                #     kl_losses.append(loss1.item())
                loss = cl_loss
                # if isinstance(loss, float):
                #     losses.append(loss)
                #     # td.set_postfix(loss = np.array(losses).mean(),  kl_loss = np.array(kl_losses).mean())
                #     td.set_postfix(loss = np.array(losses).mean())
                #     # update moemnt
                #     if is_mem:
                #         self.moment.update_mem(ind, reps.detach(), hidden.detach())
                #     else:
                #         self.moment.update(ind, reps.detach())
                #     continue
                # losses.append(loss.item())
                # td.set_postfix(loss = np.array(losses).mean(),  kl_loss = np.array(kl_losses).mean())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                
                # update moemnt
                # if is_mem:
                #     self.moment.update_mem(ind, reps.detach())
                # else:
                #     self.moment.update(ind, reps.detach())
            # print(f"{name} loss is {np.array(losses).mean()}")
        for epoch_i in range(epochs):
            # self.moment.init_proto(args, encoder, mem_data, is_memory=True)
            train_data(mem_loader, "memory_train_{}".format(epoch_i), is_mem=True)
            
            
            
            
    def kl_div_loss(self, x1, x2, t=10):

        batch_dist = F.softmax(t * x1, dim=1)
        temp_dist = F.log_softmax(t * x2, dim=1)
        loss = F.kl_div(temp_dist, batch_dist, reduction="batchmean")
        return loss
    
    # def update_all_embedding(self,args, encoder,retrieval_pool, all_data_for_pool):
    #     all_current_embeddings,indss,labelss=self.get_embedding( args, encoder, list(itertools.chain(*all_data_for_pool.values())))
    #     retrieval_pool.reset_index()
    #     retrieval_pool.add_to_retrieve_pool(all_current_embeddings,indss)
    def calib_proto(self,old_proto,new_proto,alpha=0.7):
        
        return alpha*old_proto+(1-alpha)*new_proto.mean(dim=0)

    @torch.no_grad()
    def evaluate_strict_model(self, args, encoder, test_data, protos4eval,  seen_relations):
        data_loader = get_data_loader(args, test_data, batch_size=1)
        encoder.eval()
        n = len(test_data)
        temp_rel2id = [self.rel2id[x] for x in seen_relations]
        map_relid2tempid = {k:v for v,k in enumerate(temp_rel2id)}
        map_tempid2relid = {k:v for k, v in map_relid2tempid.items()}
        correct = 0
        for step, batch_data in enumerate(data_loader):
            labels, tokens, ind = batch_data
            labels = labels.to(args.device)
            tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
            hidden, reps = encoder.bert_forward(tokens)
            labels = [map_relid2tempid[x.item()] for x in labels]
            logits = -osdist(hidden, protos4eval)
            seen_relation_ids = [self.rel2id[relation] for relation in seen_relations]
            seen_relation_ids = [map_relid2tempid[x] for x in seen_relation_ids]
            seen_sim = logits[:,seen_relation_ids]
            seen_sim = seen_sim.cpu().data.numpy()
            max_smi = np.max(seen_sim,axis=1)
            label_smi = logits[:,labels].cpu().data.numpy()
            if label_smi >= max_smi:
                correct += 1
        return correct/n
    
    def statistic_old_new(self,pred, truth, new_rel):
        too, tnn, foo, fnn, fon, fno = 0, 0, 0, 0, 0, 0
        for p, t in zip(pred.reshape(-1).tolist(), truth.reshape(-1).tolist()):
            if p == t:
                if t in new_rel:
                    tnn += 1
                else:
                    too += 1
            else:
                if t in new_rel and p in new_rel:
                    fnn += 1
                elif t not in new_rel and p in new_rel:
                    fon += 1
                elif t in new_rel and p not in new_rel:
                    fno += 1
                else:
                    foo += 1
        # return (oo,nn,on,no),(oo/(oo+on),nn/(no+nn))#(旧关系预测对数目，旧关系预测成新关系数目，新关系预测成旧关系数目，新关系预测对数目），（两个recall，对应的是TPR和1-FRP）
        return np.array((too, tnn, foo, fnn, fon, fno))
    
    @torch.no_grad()
    def evaluate_contrastive_model(self, args, encoder, test_data, protos4eval,  seen_relations, new_rel=None):
        cur_num = np.array([0, 0, 0, 0, 0, 0])
        cum_right = 0
        cum_len = 0
        
        data_loader = get_data_loader(args, test_data, batch_size=1)
        encoder.eval()
        temp_rel2id = [self.rel2id[x] for x in seen_relations]
        map_relid2tempid = {k:v for v,k in enumerate(temp_rel2id)}
        map_tempid2relid = {k:v for k, v in map_relid2tempid.items()}
        new_rel=[map_relid2tempid[self.rel2id[x]] for x in new_rel]
        for step, batch_data in enumerate(data_loader):
            labels, tokens, ind = batch_data
            labels = labels.to(args.device)
            tokens = torch.stack([x.to(args.device) for x in tokens], dim=0)
            hidden, reps = encoder.bert_forward(tokens)
            labels = torch.stack([torch.tensor(map_relid2tempid[label.item()], device=args.device) for label in labels],
                    dim=-1)
            
            logits = -osdist(hidden, protos4eval)
            preds = torch.argmax(logits, dim=-1)  # B                                                                                     1)
            cum_right += (labels == preds).sum().item()
            cum_len += len(preds)
            num = self.statistic_old_new(preds, labels, new_rel=new_rel)
            cur_num += num
                     
        print(f"Contrastive acc is {cum_right / cum_len}")
        print(
            f"test:(too,tnn,foo,fnn,fon,fno):{cur_num},old,new pred error rate:{((cur_num[2] + cur_num[4]) / (cur_num[0] + cur_num[2] + cur_num[4]), (cur_num[3] + cur_num[5]) / (cur_num[3] + cur_num[1] + cur_num[5]))}")
        return cum_right / cum_len
        
    def train(self, args):
        
        ## DEBUG
        d_pic=True
        # set training batch
        os.makedirs(args.output_path, exist_ok=True)
        
        all_rd_his=[]
        all_rd_cur=[]
        for round in range(args.total_round):
            test_cur = []
            test_total = []
            # set random seed
            
            random.seed(args.seed+round*100)

            # sampler setup
            sampler = data_sampler(args=args, seed=args.seed+round*100)
            self.id2rel = sampler.id2rel
            self.rel2id = sampler.rel2id
            # encoder setup
            encoder = Encoder(args=args).to(args.device)

            # initialize memory and prototypes
            num_class = len(sampler.id2rel)
            memorized_samples = {}

            # load data and start computation
            
            history_relation = []
            proto4repaly = {}
            
            retrieval_pool=RetrievePool()
            all_data_for_pool={}
            for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(sampler):
                
                
                all_data_for_pool.update(copy.deepcopy(training_data)) # for retreval pool update
                print(current_relations)
                # Initial
                train_data_for_initial = []
                for relation in current_relations:
                    history_relation.append(relation)
                    train_data_for_initial += training_data[relation]
                    
                if d_pic:
                    if steps == 0:
                        tokens_task1 = train_data_for_initial
                    if steps == len(sampler)-1:#len(sampler)-1
                        tokens_task2 = train_data_for_initial
                # train model
                # no memory. first train with current task
                self.moment = Moment(args)
                self.moment.init_moment(args, encoder, train_data_for_initial, is_memory=False)
                self.train_simple_model(args, encoder, train_data_for_initial, args.step1_epochs)
                # bulid index for all class
                # draw pic before update
                if d_pic:
                    if steps == len(sampler)-1:#len(sampler)-1
                        flag = False
                        self.tsne_plot(args, encoder, tokens_task1, tokens_task2, flag,draw_all_point=True)
                        self.tsne_plot(args, encoder, tokens_task1, tokens_task2, flag,draw_all_point=False)
                # repaly
                if len(proto4repaly)>0:
                    
                    for relation in current_relations:
                        memorized_samples[relation], _, temp_proto = self.select_data(args, encoder, training_data[relation])
                        proto4repaly[relation]=temp_proto.unsqueeze(dim=0).to(args.device)
                    
                    # ## retrieve start
                    retrieval_pool.reset_index()
                    # 重建所有index
                    for i,relation in enumerate(history_relation):
                        # if relation not in current_relations:
                        all_current_embeddings,indss,cur_labelss=self.get_embedding( args, encoder, all_data_for_pool[relation])
                        retrieval_pool.add_to_retrieve_pool(all_current_embeddings,class_label=relation,ids=indss) # K*hidden
                    # 检索对应proto的index
                    retrival_res=collections.defaultdict(list)
                    for relation in history_relation:
                        # if relation not in current_relations:
                        retrieval_pool.retrieval_error_index(proto4repaly[relation],args.num_protos,relation,retrival_res)
                    #拿到对应的例子
                    train_data_for_memory = []
                    for relation in history_relation:
                        cur_rel_data=all_data_for_pool[relation]
                        memorized_samples[relation]=[cur_rel_data[k] for k in retrival_res[relation]]
                        train_data_for_memory += memorized_samples[relation]

                    # train_data_for_memory = []
                    # for relation in history_relation:
                    #     if relation not in current_relations:
                    #         cur_rel_data=all_data_for_pool[relation]
                    #         memorized_samples[relation]=[cur_rel_data[k] for k in retrival_res[relation]]
                    #         train_data_for_memory += memorized_samples[relation]
                    #     else:
                    #         train_data_for_memory += memorized_samples[relation]
                    # ## retrieve end
                    
                    ## random sampling start
                    # train_data_for_memory = []
                    # for relation in history_relation:
                    #     if relation not in current_relations:
                    #         memorized_samples[relation]=random.sample(all_data_for_pool[relation],k=args.num_protos)
                    #         train_data_for_memory += memorized_samples[relation]
                    ## random sampling end
                    
                    
                    # ini version
                    # train_data_for_memory = []
                    # for relation in history_relation:
                    #     train_data_for_memory += memorized_samples[relation]
                    
                    self.moment.init_proto(args, encoder, train_data_for_memory, is_memory=True)
                    self.train_mem_model(args, encoder, train_data_for_memory, proto4repaly1 , args.step2_epochs, seen_relations)
                    
                    # draw pic before update
                    if d_pic:
                        if steps == len(sampler)-1:#len(sampler)-1
                            flag = True
                            self.tsne_plot(args, encoder, tokens_task1, tokens_task2, flag,draw_all_point=False)
                            self.tsne_plot(args, encoder, tokens_task1, tokens_task2, flag,draw_all_point=True)
                # else:
                #     #之前类别数据添加到池子里
                #     retrieval_pool.add_to_retrieve_pool(all_current_embeddings,indss)
                
                feat_mem = []
                proto_mem = []
                for relation in current_relations:
                    # retrieval_pool=self.add_to_retrieve_pool(args, encoder, training_data[relation],retrieval_pool=retrieval_pool)
                    memorized_samples[relation], feat, temp_proto = self.select_data(args, encoder, training_data[relation])
                    feat_mem.append(feat)
                    proto_mem.append(temp_proto)
                    proto4repaly[relation]=temp_proto.unsqueeze(dim=0).to(args.device)
                
                temp_proto = torch.stack(proto_mem, dim=0)

                protos4eval = []
                
                self.lbs = []
                for relation in history_relation:
                    if relation not in current_relations:# old relation
                        # old_proto=proto4repaly[relation]
                        protos, featrues = self.get_proto(args, encoder, memorized_samples[relation])
                        # protos, _ = self.cab_proto(args, encoder, memorized_samples[relation],old_proto)## TODO:new get old proto by calibration
                        protos4eval.append(protos)
                        proto4repaly[relation]=protos
                        
                if protos4eval:
                    
                    protos4eval = torch.cat(protos4eval, dim=0).detach()
                    protos4eval = torch.cat([protos4eval, temp_proto.to(args.device)], dim=0)
                    
                else:
                    protos4eval = temp_proto.to(args.device)
                    
                # # TODO：protos4eval这里需要提供所有的prototype
                proto4repaly1 = protos4eval.clone()

                test_data_1 = []
                for relation in current_relations:
                    test_data_1 += test_data[relation]

                test_data_2 = []
                for relation in seen_relations:
                    test_data_2 += historic_test_data[relation]

                cur_acc = self.evaluate_strict_model(args, encoder, test_data_1, protos4eval, seen_relations)
                # total_acc = self.evaluate_strict_model(args, encoder, test_data_2, protos4eval, seen_relations)
                total_acc =self.evaluate_contrastive_model( args, encoder, test_data_2, protos4eval,  seen_relations, new_rel=current_relations)
                print(f'Restart Num {round+1}')
                print(f'task--{steps + 1}:')
                print(f'current test acc:{cur_acc}')
                print(f'history test acc:{total_acc}')
                # print(f'history test acc1:{total_acc1}')
                test_cur.append(cur_acc)
                test_total.append(total_acc)
                if steps==len(sampler)-1:
                    print(test_cur)
                    print(test_total)
                    all_rd_cur.append(test_cur)
                    all_rd_his.append(test_total)
                
                del self.moment
        print(f"cur acc is:")
        for i in range(len(all_rd_cur)):
            cur_str=" ".join(list(map(str,all_rd_cur[i])))
            print(cur_str)
        print(f"his acc is:")
        for i in range(len(all_rd_cur)):
            his_str=" ".join(list(map(str,all_rd_his[i])))
            print(his_str)
  
