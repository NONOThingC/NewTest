import heapq
import faiss
import numpy as np
import collections
import torch
import random


class RetrievePool:

    def __init__(self, rel2id) -> None:
        self.index = None
        self.retrieve_pool = {}
        self.rel2id = rel2id
        self.class2ind = collections.defaultdict(list)

    def add_to_retrieve_pool(self,
                             embeddings,
                             class_label,
                             ids=None,
                             labels=None):
        if self.index is None:
            self.retrieve_pool[class_label] = self.build_index(
                embeddings,
                ids=ids,
                index_type="cosine",
                cluster_num=min(embeddings.shape[0] // 2, 20),
                use_gpu=False)
            self._add_inds(relation=class_label, inds=ids.tolist())
        else:
            # if ids==None:
            #     self.index.add(embeddings)
            # else:
            #     self.index.add_with_ids(embeddings, ids)
            #     self.labels=labels
            pass

    def _add_index(self):
        pass

    def _add_inds(self, relation, inds):
        self.class2ind[relation].extend(inds)

    def update_index(self, embeddings, ids, class_label):
        try:
            ids = np.asarray(ids.astype('int64'))
            embedding = embedding.astype('float32')
            embedding = embedding.to("cpu")
        except:
            pass
        self.retrieve_pool[class_label].remove_ids(ids)
        self.retrieve_pool[class_label].add_with_ids(embeddings, ids)

    def reset_index(self):
        self.index = None
        self.labels = None
        self.retrieve_pool = {}

    def build_index(self,
                    user_embedding,
                    ids=None,
                    labels=None,
                    index_type="cosine",
                    quantizer_type=None,
                    dist_metric=0,
                    cluster_num=50,
                    use_gpu=False):
        dimension = user_embedding.shape[-1]
        try:
            embedding = user_embedding.astype('float32')
            embedding = embedding.to("cpu")
        except:
            pass

        index = None
        if index_type == "IndexFlatL2":
            index = faiss.IndexFlatL2(dimension)
        elif index_type == "IndexFlatIP":
            index = faiss.IndexFlatIP(dimension)
        elif index_type == "IndexLSH":
            index = faiss.IndexLSH(dimension, nbits)
        elif index_type == "IndexPQ":
            index = faiss.IndexPQ(dimension, m, nbits)
        elif index_type == "cosine":
            # cosine = normalize & dot
            faiss.normalize_L2(embedding)
            index = faiss.IndexFlatIP(dimension)  # 构建索引index

        quantizer = None
        if quantizer_type == "IndexFlatL2":
            quantizer = faiss.IndexFlatL2(dimension)
        elif quantizer_type == "IndexFlatIP":
            quantizer = faiss.IndexFlatIP(dimension)
        elif quantizer_type == "IndexLSH":
            quantizer = faiss.IndexLSH(dimension, nbits)
        elif quantizer_type == "IndexPQ":
            quantizer = faiss.IndexPQ(dimension, m, nbits)

        metric = faiss.METRIC_INNER_PRODUCT
        if dist_metric == 1:
            metric = faiss.METRIC_L2
        if quantizer_type == "IndexIVFFlat":
            index = faiss.IndexIVFFlat(
                quantizer, dimension, cluster_num, faiss.METRIC_INNER_PRODUCT
            )  #faiss.METRIC_L2（欧式距离）、②faiss.METRIC_INNER_PRODUCT（向量内积）
            index.set_direct_map_type(faiss.DirectMap.Hashtable)
        # index = faiss.IndexIVFPQ(quantizer, dimension, cluster_num, m, 8)
        elif quantizer_type == "IndexIVFPQ":
            index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
            index.set_direct_map_type(faiss.DirectMap.Hashtable)

        # index=faiss.downcast_index(index.index).make_direct_map()
        if use_gpu == True:
            index = faiss.index_cpu_to_all_gpus(index)
        if not index.is_trained:
            index.train(embedding)
        if ids is None:
            index.add(embedding)
        else:
            index = faiss.IndexIDMap2(index)
            ids = np.asarray(ids.astype('int64'))
            index.add_with_ids(embedding, ids)

        return index

    def read_vectors(input_dir, vec_col=0, id_col=-1):
        if not os.path.exists(input_dir):
            print("SystemLog: File {} NOT FOUND\n".format(input_dir))
            raise FileNotFoundError
        input_files = os.listdir(input_dir)
        print("SystemLog: Will read file {} from {}\n".format(
            input_files, input_dir))
        vec_res = []
        id_res = []
        for file in input_files:
            file_path = os.path.join(input_dir, file)
            with open(file_path, 'r', encoding='utf8') as f:
                for line in f:
                    line_sep = line.strip().split(SEP)
                    vec_str = line_sep[vec_col]
                    vec = [float(s) for s in vec_str.split(VEC_SEP)]
                    vec_res.append(vec)
                    if id_col >= 0:
                        id_res.append(int(line_sep[id_col]))
        if len(id_res) < 1:
            id_res = None
        else:
            id_res = np.array(id_res)
        return np.array(vec_res).astype('float32'), id_res

    # def build_index(data, index_type="IndexIVFPQ", quantizer_type="IndexFlatIP", nlist=1, m=1, nbits=8, dist_metric=0, *args, **kwargs):
    #     d = len(data[0])
    #     index = None
    #     if index_type == "IndexFlatL2":
    #         index = faiss.IndexFlatL2(d)
    #     elif index_type == "IndexFlatIP":
    #         index = faiss.IndexFlatIP(d)
    #     elif index_type == "IndexLSH":
    #         index = faiss.IndexLSH(d, nbits)
    #     elif index_type == "IndexPQ":
    #         index = faiss.IndexPQ(d, m, nbits)
    #     if index is not None:
    #         index.add(data)
    #         return index

    #     quantizer = None
    #     if quantizer_type == "IndexFlatL2":
    #         quantizer = faiss.IndexFlatL2(d)
    #     elif quantizer_type == "IndexFlatIP":
    #         quantizer = faiss.IndexFlatIP(d)
    #     elif  quantizer_type == "IndexLSH":
    #         quantizer = faiss.IndexLSH(d, nbits)
    #     elif quantizer_type == "IndexPQ":
    #         quantizer = faiss.IndexPQ(d, m, nbits)

    #     metric = faiss.METRIC_INNER_PRODUCT
    #     if dist_metric == 1:
    #         metric = faiss.METRIC_L2

    #     if index_type == "IndexIVFFlat":
    #         index = faiss.IndexIVFFlat(quantizer, d, nlist, metric)
    #     elif index_type == "IndexIVFPQ":
    #         index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    #     index.train(data)
    #     index.add(data)
    #     return index
    def _get_cur_rel(self):
        return list(self.retrieve_pool.keys())

    def retrieval_old_data(self, q, K):
        # Rejection Sampling
        # q [L,h]
        D, I = self.batch_query(q, self.index, k=K)
        # D N,K->K
        I = I.reshape(-1)
        D = D.view(-1)
        _, Ik = torch.sort(D)
        retrieved = set()
        i = 0
        Ik = Ik.tolist()
        while i < len(Ik) and len(retrieved) < K:
            cur = I[Ik[i]].item()
            if cur not in retrieved:
                retrieved.add(cur)
            i += 1
        indexes = list(retrieved)
        ret_vec = np.vstack([self.index.reconstruct(i) for i in indexes])
        return ret_vec, indexes  #K,H

    def retrieval_in_batch(self,
                           q,
                           K,
                           labels,
                           random_ratio=0,
                           ret_embedding=1,
                           must_every_class=False):
        # q [L,h] class_item [L]
        ## Method 1: retreve top 1
        # K=1
        # q=q.view(1,-1).cpu().numpy()
        # cur_all_rels=self._get_cur_rel()
        # # retrieved_res=collections.defaultdict(list)
        # rel2id={rel:i for i,rel in enumerate(cur_all_rels)}
        # id2rel={v:k for k,v in rel2id.items()}
        # for rel in cur_all_rels:
        #     if rel!=class_item:
        #         D,I=self.batch_query(q,self.retrieve_pool[rel],k=K)

        #         retrieved_res[rel].append(I.item())# (C-1)*K
        # Rejection Sampling
        # q [L,h]
        # ## Method 2: retreve top K
        # q=q.view(q.shape[0],1,-1).detach().cpu().numpy()
        # labels=labels.cpu().tolist()
        # cur_all_rels=self._get_cur_rel()
        # # retrieved_res=collections.defaultdict(list)
        # rel2id={rel:i for i,rel in enumerate(cur_all_rels)}
        # id2rel={v:k for k,v in rel2id.items()}
        # positive_num=K//4
        # ret_res=set()
        # for ii in range(q.shape[0]):
        #     cls_mat=[]
        #     res=[]

        #     for rel in cur_all_rels:
        #         if self.rel2id[rel]!=labels[ii]:
        #             D,I=self.batch_query(q[ii,:,:],self.retrieve_pool[rel],k=K-positive_num)# q (1,h) D(1,K-1)
        #             # class matrice
        #             cls_mat.append(rel2id[rel])
        #             res.append((D,I))
        #         else:
        #             _,I=self.batch_query(-q[ii,:,:],self.retrieve_pool[rel],k=positive_num)
        #             for i in range(positive_num):
        #                 ids=I[0][i].item()
        #                 if ids!=-1:
        #                     ret_res.add(ids)
        #             # ret_res.add(I[0][0].item())
        #             # retrieved_res[rel].append(I.item())# (C-1)*K
        #     # return retrieved_res
        #     # D (C-1)*(NR-1)->  NR-1
        #     D,I=list(zip(*res)) # D (C-1)*(NR-1)->  NR-1
        #     D=np.concatenate(D,axis=0)
        #     I=np.concatenate(I,axis=0)
        #     m,n = len(D),len(D[0])
        #     pq = [(D[i][0], i, 0) for i in range(m)]

        #     heapq.heapify(pq)
        #     cnt=0
        #     while len(pq)>0 and cnt < K-1:
        #         num, x, y = heapq.heappop(pq)
        #         id=I[x][y].item()
        #         if id==-1:
        #             continue
        #         if id not in ret_res:
        #             ret_res.add(id)
        #             cnt+=1
        #         if y != n - 1:
        #             heapq.heappush(pq, (D[x][y + 1], x, y + 1))
        #     # assert len(ret_res)==K
        # print(f"retrieved num is: {len(ret_res)},k is {K},C is {len(cur_all_rels)},positive num is {positive_num}")

        # # assert len(ret_res)==K*len(q)
        # return list(ret_res)
        # ## method 3: vector retrieved by class
        # q=q.view(q.shape[0],1,-1).detach().cpu().numpy()
        # labels=labels.cpu().tolist()
        # cur_all_rels=self._get_cur_rel()

        # # retrieved_res=collections.defaultdict(list)
        # rel2id={rel:i for i,rel in enumerate(cur_all_rels)}
        # id2rel={v:k for k,v in rel2id.items()}
        # total_embeddings=[]
        # total_labels=[]
        # positive_num=10
        # for ii in range(q.shape[0]):
        #     cls_mat=[]
        #     res=[]
        #     retrieved=collections.defaultdict(set)
        #     embeddings=[]

        #     for rel in cur_all_rels:
        #         if self.rel2id[rel]!=labels[ii]:
        #             D,I=self.batch_query(q[ii,:,:],self.retrieve_pool[rel],k=K-positive_num)# q (1,h) D(1,K-1)
        #             # class matrice
        #             cls_mat.append(rel2id[rel])
        #             res.append((D,I))
        #         else:
        #             _,I=self.batch_query(-q[ii,:,:],self.retrieve_pool[rel],k=positive_num)
        #             # ret.add(I[0][0].item())
        #             for i in range(positive_num):
        #                 ids=I[0][i].item()
        #                 if ids!=-1:
        #                     retrieved[rel].add(ids)
        #             # retrieved_res[rel].append(I.item())# (C-1)*K
        #     # return retrieved_res
        #     # D (C-1)*(NR-1)->  NR-1
        #     D,I=list(zip(*res)) # D (C-1)*(NR-1)->  NR-1
        #     D=np.concatenate(D,axis=0)
        #     I=np.concatenate(I,axis=0)
        #     m,n = len(D),len(D[0])
        #     pq = [(D[i][0], i, 0) for i in range(m)]

        #     heapq.heapify(pq)

        #     cnt=0
        #     while cnt < K-1:
        #         _, x, y = heapq.heappop(pq)
        #         id=I[x][y].item()
        #         if id==-1:
        #             continue
        #         rel_id=cls_mat[x]
        #         if id not in retrieved[id2rel[rel_id]]:
        #             retrieved[id2rel[rel_id]].add(id)
        #             cnt+=1
        #         if y != n - 1:
        #             heapq.heappush(pq, (D[x][y + 1], x, y + 1))

        #     # print(f"retrieved num is: {len(ret)},k is {K}")
        #     cur_labels=[]
        #     for rel,ids in retrieved.items():
        #         cur_labels.extend([self.rel2id[rel]]*len(ids))
        #         embeddings.extend([self.retrieve_pool[rel].reconstruct(id) for id in ids])
        #     embeddings=torch.tensor(np.array(embeddings))
        #     total_embeddings.append(embeddings)
        #     total_labels.append(cur_labels)
        # total_labels=torch.tensor(total_labels)
        # total_embeddings=torch.stack(total_embeddings,dim=0)
        # return total_embeddings,total_labels
        # ## method 4: vector retrieved by every class
        # q=q.view(q.shape[0],1,-1).detach().cpu().numpy()
        # # q=q.view(q.shape[0],1,-1).detach().cpu().numpy()
        # labels=labels.cpu().tolist()
        # cur_all_rels=self._get_cur_rel()
        # # retrieved_res=collections.defaultdict(list)
        # rel2id={rel:i for i,rel in enumerate(cur_all_rels)}
        # id2rel={v:k for k,v in rel2id.items()}
        # total_embeddings=[]
        # total_labels=[]
        # for ii in range(q.shape[0]):
        #     embeddings=[]
        #     cur_labels=[]
        #     for rel in cur_all_rels:
        #         # if self.rel2id[rel]!=labels[ii]:
        #         _,I=self.batch_query(q[ii,:,:],self.retrieve_pool[rel],k=1)
        #         id=I[0][0].item()
        #         embeddings.append(self.retrieve_pool[rel].reconstruct(id))
        #         cur_labels.extend([self.rel2id[rel]])
        #     embeddings=torch.tensor(np.array(embeddings))
        #     total_embeddings.append(embeddings)
        #     total_labels.append(cur_labels)
        # total_labels=torch.tensor(total_labels)
        # total_embeddings=torch.stack(total_embeddings,dim=0)
        # return total_embeddings,total_labels
        ## method 5: vector retrieved by all class
        # q=np.random.rand(q.shape[0],q.shape[1]).reshape(q.shape[0],1,-1).astype("float32")
        q = q.view(q.shape[0], 1, -1).detach().cpu().numpy()
        labels = labels.cpu().tolist()
        cur_all_rels = self._get_cur_rel()
        # retrieved_res=collections.defaultdict(list)
        rel2id = {rel: i for i, rel in enumerate(cur_all_rels)}
        id2rel = {v: k for k, v in rel2id.items()}
        total_embeddings = []
        total_labels = []

        retrieved = collections.defaultdict(set)
        random_retrieved = set()

        close_retrieve = int((1 - random_ratio) * K)
        positive_num = max(1, close_retrieve // 4)
        random_total_num = max(K - close_retrieve - positive_num, 0)
        startK = 0  # [0,C-1)
        print(
            f"random num is {random_total_num},retrieve num is {close_retrieve},positive num is {positive_num}"
        )
        close_cnt = 0
        for ii in range(q.shape[0]):
            cls_mat = []
            res = []
            pos_num = 0

            for rel in cur_all_rels:
                ## random part start
                ids = random.sample(self.class2ind[rel],
                                    k=min(random_total_num,
                                          len(self.class2ind[rel])))
                for id in ids:
                    random_retrieved.add((rel, id))
                ## random part end

                ## retrieve method part start
                if self.rel2id[rel] != labels[ii]:
                    if close_retrieve - positive_num + 1 < 1:
                        continue
                    D, I = self.batch_query(q[ii, :, :],
                                            self.retrieve_pool[rel],
                                            k=(startK + 1) *
                                            (close_retrieve - positive_num +
                                             10))  # q (1,h) D(1,K-1)
                    # class matrice
                    cls_mat.append(rel2id[rel])
                    res.append((D, I))
                else:
                    _, I = self.batch_query(-q[ii, :, :],
                                            self.retrieve_pool[rel],
                                            k=(startK + 1) * positive_num)
                    # ret.add(I[0][0].item())
                    for i in range(positive_num):
                        ids = I[0][i].item()
                        if ids != -1:
                            if ids not in retrieved[rel]:
                                retrieved[rel].add(ids)
                                pos_num += 1
                    # retrieved_res[rel].append(I.item())# (C-1)*K
                ## retrieve method part end

            # return retrieved_res
            # D (C-1)*(NR-1)->  NR-1
            if len(res) != 0:
                D, I = list(zip(*res))  # D (C-1)*(NR-1)->  NR-1
                D = np.concatenate(D, axis=0)
                I = np.concatenate(I, axis=0)
                m, n = len(D), len(D[0])
                pq = [(D[i][startK], i, startK) for i in range(m)]

                heapq.heapify(pq)
                close_cnt += pos_num
                cnt = 0
                # jump_flag=0
                if must_every_class:
                    x, y = 0, 0
                    while cnt < close_retrieve - pos_num:
                        id = I[x][y].item()
                        
                        if id != -1:
                            rel_id = cls_mat[x]
                            if id not in retrieved[id2rel[rel_id]]:
                                retrieved[id2rel[rel_id]].add(id)
                                cnt += 1
                                close_cnt += 1
                        x += 1
                        
                        if x > m - 1:
                            x = 0
                            y = y + 1
                            if y >= n:
                                break
                else:
                    while len(pq) > 0 and cnt < close_retrieve - pos_num:
                        _, x, y = heapq.heappop(pq)
                        id = I[x][y].item()
                        if id == -1:
                            continue
                        rel_id = cls_mat[x]
                        if id not in retrieved[id2rel[rel_id]]:
                            # if (jump_flag%3)==0:
                            retrieved[id2rel[rel_id]].add(id)
                            cnt += 1
                            close_cnt += 1
                        # jump_flag+=1
                        if y != n - 1:
                            heapq.heappush(pq, (D[x][y + 1], x, y + 1))
                    if cnt < close_retrieve - pos_num:
                        pq = [(D[i][0], i, 0) for i in range(m)]
                        heapq.heapify(pq)
                        while len(pq) > 0 and cnt < close_retrieve - pos_num:
                            _, x, y = heapq.heappop(pq)
                            id = I[x][y].item()
                            if id == -1:
                                continue
                            rel_id = cls_mat[x]
                            if id not in retrieved[id2rel[rel_id]]:
                                retrieved[id2rel[rel_id]].add(id)
                                cnt += 1
                                close_cnt += 1
                            if y != startK:
                                heapq.heappush(pq, (D[x][y + 1], x, y + 1))
        random_retrieved = list(random_retrieved)
        if len(random_retrieved) > 0:
            random_indexes = random.sample(random_retrieved,
                                           k=min(K * q.shape[0] - close_cnt,len(random_retrieved)))
        else:
            random_indexes = []

        print(f"retrieved res is:")
        for rel in sorted(retrieved.keys()):
            ids = sorted(retrieved[rel])
            print(f"{rel}:{ids}")
        if ret_embedding:
            for rel, id in random_indexes:
                total_labels.append(self.rel2id[rel])
                total_embeddings.append(
                    self.retrieve_pool[rel].reconstruct(id))
            for rel, ids in retrieved.items():
                total_labels.extend([self.rel2id[rel]] * len(ids))
                total_embeddings.extend(
                    [self.retrieve_pool[rel].reconstruct(id) for id in ids])

            total_labels = torch.tensor(total_labels)
            total_embeddings = torch.tensor(np.array(total_embeddings))
            print(
                f"retrieved num is: {len(total_labels)},k is {K},C is {len(cur_all_rels)},positive num is {positive_num}"
            )
            return total_embeddings, total_labels
        else:

            for rel, ids in retrieved.items():
                random_indexes.extend(ids)
            print(
                f"retrieved num is: {len(random_indexes)},k is {K},C is {len(cur_all_rels)},positive num is {positive_num}"
            )
            return random_indexes

    def retrieval_in_batch_random(self, q, K, labels, func=0):

        # ## method: random retrieved
        # q=np.random.rand(q.shape[0],q.shape[1]).reshape(q.shape[0],1,-1).astype("float32")
        # # q=q.view(q.shape[0],1,-1).detach().cpu().numpy()
        # labels=labels.cpu().tolist()
        # cur_all_rels=self._get_cur_rel()
        # # retrieved_res=collections.defaultdict(list)
        # rel2id={rel:i for i,rel in enumerate(cur_all_rels)}
        # id2rel={v:k for k,v in rel2id.items()}
        # total_embeddings=[]
        # total_labels=[]
        # if func==1:
        #     for ii in range(q.shape[0]):

        #         embeddings=[]
        #         cur_labels=[]
        #         for rel in cur_all_rels:
        #             if self.rel2id[rel]!=labels[ii]:
        #                 _,I=self.batch_query(q[ii,:,:],self.retrieve_pool[rel],k=1)
        #             else:
        #                 _,I=self.batch_query(-q[ii,:,:],self.retrieve_pool[rel],k=1)
        #             id=I[0][0].item()
        #             embeddings.append(self.retrieve_pool[rel].reconstruct(id))
        #             cur_labels.extend([self.rel2id[rel]])
        #         embeddings=torch.tensor(np.array(embeddings))
        #         total_embeddings.append(embeddings)
        #         total_labels.append(cur_labels)
        #     total_labels=torch.tensor(total_labels)
        #     total_embeddings=torch.stack(total_embeddings,dim=0)
        #     return total_embeddings,total_labels
        # else:
        #     retrieved=collections.defaultdict(set)
        #     positive_retrieved=collections.defaultdict(set)
        #     pos_cnt=0
        #     for ii in range(q.shape[0]):
        #         embeddings=[]
        #         cur_labels=[]
        #         for rel in cur_all_rels:
        #             if self.rel2id[rel]==labels[ii]:
        #                 _,I=self.batch_query(-q[ii,:,:],self.retrieve_pool[rel],k=1)
        #                 ids=I[0][0].item()
        #                 if ids!=-1:
        #                     positive_retrieved[rel].add(ids)
        #                     pos_cnt+=1
        #             _,I=self.batch_query(q[ii,:,:],self.retrieve_pool[rel],k=K)

        #             rd_num=random.choice(list(range(K)))
        #             for jj in range(rd_num):
        #                 id=I[0][jj].item()
        #                 if id!=-1:
        #                     retrieved[rel].add(id)

        #     cnt=K*len(cur_all_rels)-pos_cnt
        #     for rel,ids in retrieved.items():
        #         rd_num=random.choice(list(range(cnt)))
        #         if rd_num==0:
        #             continue
        #         re_num=min(rd_num,len(ids))
        #         ids=random.sample(ids,re_num)
        #         total_labels.extend([self.rel2id[rel]]*len(ids))
        #         total_embeddings.extend([self.retrieve_pool[rel].reconstruct(id) for id in ids])
        #         cnt-=re_num
        #         if cnt<=0:
        #             break
        #     for rel,ids in positive_retrieved.items():
        #         total_labels.extend([self.rel2id[rel]]*len(ids))
        #         total_embeddings.extend([self.retrieve_pool[rel].reconstruct(id) for id in ids])
        #     print(f"retrieved num is: {len(total_labels)},k is {K},C is {len(cur_all_rels)}")
        #     total_labels=torch.tensor(total_labels)
        #     total_embeddings=torch.tensor(np.array(total_embeddings))
        #     return total_embeddings,total_labels
        # ## method: random retrieved

        # q=q.view(q.shape[0],1,-1).detach().cpu().numpy()

        cur_all_rels = self._get_cur_rel()
        # retrieved_res=collections.defaultdict(list)
        rel2id = {rel: i for i, rel in enumerate(cur_all_rels)}
        id2rel = {v: k for k, v in rel2id.items()}
        total_embeddings = []
        total_labels = []

        retrieved = []
        positive_retrieved = collections.defaultdict(set)
        pos_cnt = 0

        for ii in range(q.shape[0]):

            for rel in cur_all_rels:
                if self.rel2id[rel] == labels[ii]:
                    ids = random.sample(self.class2ind[rel], k=1)[0]
                    positive_retrieved[rel].add(ids)
                    pos_cnt += 1
                else:
                    ids = random.sample(self.class2ind[rel],
                                        k=min(K - 1, len(self.class2ind[rel])))
                    for id in ids:
                        retrieved.append((rel, id))

        cnt = K * len(cur_all_rels) - pos_cnt
        ret_ind = random.sample(list(range(len(retrieved))), cnt)

        for ind in ret_ind:
            rel, id = retrieved[ind]
            total_labels.append(self.rel2id[rel])
            total_embeddings.append(self.retrieve_pool[rel].reconstruct(id))

        for rel, ids in positive_retrieved.items():
            total_labels.extend([self.rel2id[rel]] * len(ids))
            total_embeddings.extend(
                [self.retrieve_pool[rel].reconstruct(id) for id in ids])
        print(
            f"retrieved num is: {len(total_labels)},k is {K},C is {len(cur_all_rels)}"
        )
        total_labels = torch.tensor(total_labels)
        total_embeddings = torch.tensor(np.array(total_embeddings))
        return total_embeddings, total_labels

    def retrieve_query(self, q, rel):
        q = q.view(q.shape[0], 1, -1).detach().cpu().numpy()
        close_retrieve = q.shape[0]
        res = []
        retrieved = set()
        ret_cnt = 1
        while len(retrieved) < close_retrieve:
            for ii in range(close_retrieve):
                D, I = self.batch_query(-q[ii, :, :],
                                        self.retrieve_pool[rel],
                                        k=ret_cnt)
                # ret.add(I[0][0].item())
                # ids=I[0][0].item()
                res.append((D, I))
                for i in range(max(0, ret_cnt - 4), ret_cnt):
                    ids = I[0][i].item()
                    if ids != -1:
                        retrieved.add(ids)
            ret_cnt += 4
        retrieved = list(retrieved)[:q.shape[0]]

        print(f"explore cnt is: {ret_cnt}")
        print(f"query retrieved result is: {retrieved}")
        print(f"retrieved query num is: {len(retrieved)},q is {q.shape[0]}")
        return retrieved

    def retrieval_error_index(self, q, K, class_item, retrieved_res):
        ## Method 1: retreve top 1
        # K=1
        # q=q.view(1,-1).cpu().numpy()
        # cur_all_rels=self._get_cur_rel()
        # # retrieved_res=collections.defaultdict(list)
        # rel2id={rel:i for i,rel in enumerate(cur_all_rels)}
        # id2rel={v:k for k,v in rel2id.items()}
        # for rel in cur_all_rels:
        #     if rel!=class_item:
        #         D,I=self.batch_query(q,self.retrieve_pool[rel],k=K)

        #         retrieved_res[rel].append(I.item())# (C-1)*K
        # Rejection Sampling
        # q [L,h]
        ## Method 2: retreve top K
        q = q.view(1, -1).cpu().numpy()
        cur_all_rels = self._get_cur_rel()
        # retrieved_res=collections.defaultdict(list)
        rel2id = {rel: i for i, rel in enumerate(cur_all_rels)}
        id2rel = {v: k for k, v in rel2id.items()}
        res = []
        for rel in cur_all_rels:
            if rel != class_item:
                D, I = self.batch_query(q, self.retrieve_pool[rel], k=K)
                # class matrice
                cls_mat = rel2id[rel] * np.ones_like(I)
                res.append((D, I, cls_mat))
                # retrieved_res[rel].append(I.item())# (C-1)*K
        # return retrieved_res
        # D (C-1)*K->K
        D, I, cls_mat = list(zip(*res))
        D = np.concatenate(D, axis=0).reshape(-1)
        I = np.concatenate(I, axis=0).reshape(-1)
        cls_mat = np.concatenate(cls_mat, axis=0).reshape(-1)
        Ik = np.argsort(D)
        retrieved = collections.defaultdict(set)
        cl_cnt = len(cur_all_rels)
        i = 0
        cnt = 0
        Ik = Ik.tolist()
        while i < len(Ik) and cl_cnt > 0:
            j = Ik[i]
            rel_id = cls_mat[j].item()
            id = I[j].item()
            if len(retrieved[id2rel[rel_id]]) == 0:
                cl_cnt -= 1
                retrieved[id2rel[rel_id]].add(id)
                cnt += 1
            i += 1
        i = 0
        while i < len(Ik) and cnt < K:
            j = Ik[i]
            rel_id = cls_mat[j].item()
            id = I[j].item()
            if id not in retrieved[id2rel[rel_id]]:
                retrieved[id2rel[rel_id]].add(id)
                cnt += 1
            i += 1
        retrieved = {k: list(v) for k, v in retrieved.items()}
        for k, v in retrieved.items():
            retrieved_res[k].extend(list(v))

        print(f"retrieved num is: {cnt},k is {K}")
        # indexes=list(retrieved)
        # ret_vec= np.vstack([self.index.reconstruct(i) for i in indexes ])
        # return list(retrieved) #K,H

    def batch_query(self, query_arr, index, k=1, nprobe=300, ids=None):
        index.nprobe = nprobe
        D, I = index.search(query_arr, k)
        # if ids is not None:
        #     I = ids[I]
        return D, I  #n,k
