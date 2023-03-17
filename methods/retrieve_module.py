import faiss
import numpy as np
import collections
import torch
class RetrievePool:
    
    def __init__(self) -> None:
        self.index=None
        self.retrieve_pool={}
        
    def add_to_retrieve_pool(self,embeddings,class_label,ids=None,labels=None):
        if self.index is None:
            self.retrieve_pool[class_label]=self.build_index(embeddings,ids=ids,index_type="cosine",cluster_num=min(embeddings.shape[0]//2,20),use_gpu=False)
        else:
            # if ids==None:
            #     self.index.add(embeddings)
            # else:
            #     self.index.add_with_ids(embeddings, ids)
            #     self.labels=labels
            pass
            
    def _add_index(self ):
        pass
    
    def reset_index(self):
        self.index=None
        self.labels=None
        self.retrieve_pool={}
    
    def build_index(self,user_embedding,ids=None,labels=None,index_type="cosine",cluster_num=50,use_gpu=False):
        dimension=user_embedding.shape[-1]
        try:
            embedding=user_embedding.astype('float32')
            embedding=embedding.to("cpu")
        except:
            pass
        if index_type == "L2":
            quantizer  = faiss.IndexFlatL2(dimension)  # 构建索引index
        if index_type == "dot":
            quantizer  = faiss.IndexFlatIP(dimension)  # 构建索引index
        if index_type == "cosine":
            # cosine = normalize & dot
            faiss.normalize_L2(embedding)
            quantizer  = faiss.IndexFlatIP(dimension)  # 构建索引index
            
        index = faiss.IndexIVFFlat(quantizer, dimension, cluster_num, faiss.METRIC_INNER_PRODUCT)#faiss.METRIC_L2（欧式距离）、②faiss.METRIC_INNER_PRODUCT（向量内积）
        # index = faiss.IndexIVFPQ(quantizer, dimension, cluster_num, m, 8)
        
        # index=faiss.downcast_index(index.index).make_direct_map()
        if use_gpu==True:
            index = faiss.index_cpu_to_all_gpus(index)
        if not index.is_trained:
            index.train(embedding)
        if ids is None:
            index.add(embedding)
        else:
            index.add_with_ids(embedding, ids)
        return index
    
    def read_vectors(input_dir, vec_col=0, id_col=-1):
        if not os.path.exists(input_dir):
            print("SystemLog: File {} NOT FOUND\n".format(input_dir))
            raise FileNotFoundError
        input_files = os.listdir(input_dir)
        print("SystemLog: Will read file {} from {}\n".format(input_files, input_dir))
        vec_res = []; id_res = []
        for file in input_files:
            file_path = os.path.join(input_dir, file)
            with open(file_path, 'r', encoding='utf8') as f:
                for line in f:
                    line_sep = line.strip().split(SEP)
                    vec_str = line_sep[vec_col]
                    vec = [float(s) for s in vec_str.split(VEC_SEP)]
                    vec_res.append(vec)
                    if id_col>=0:
                        id_res.append(int(line_sep[id_col]))
        if len(id_res)<1:
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
    
    def retrieval_old_data(self,q,K):
        # Rejection Sampling
        # q [L,h]
        D,I=self.batch_query(q,self.index,k=K)
        # D N,K->K
        I=I.reshape(-1)
        D=D.view(-1)
        _, Ik=torch.sort(D)
        retrieved=set()
        i=0
        Ik=Ik.tolist()
        while i<len(Ik) and len(retrieved)<K:
            cur=I[Ik[i]].item()
            if cur not in retrieved:
                retrieved.add(cur)
            i+=1
        indexes=list(retrieved)
        ret_vec= np.vstack([self.index.reconstruct(i) for i in indexes ])
        return ret_vec,indexes #K,H
    

    
    def retrieval_error_index(self,q,K,class_item,retrieved_res):
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
        q=q.view(1,-1).cpu().numpy()
        cur_all_rels=self._get_cur_rel()
        # retrieved_res=collections.defaultdict(list)
        rel2id={rel:i for i,rel in enumerate(cur_all_rels)}
        id2rel={v:k for k,v in rel2id.items()}
        res=[]
        for rel in cur_all_rels:
            if rel!=class_item:
                D,I=self.batch_query(q,self.retrieve_pool[rel],k=K)
                # class matrice
                cls_mat=rel2id[rel]*np.ones_like(I)
                res.append((D,I,cls_mat))
                # retrieved_res[rel].append(I.item())# (C-1)*K
        # return retrieved_res
        # D (C-1)*K->K
        D,I,cls_mat=list(zip(*res))
        D=np.concatenate(D,axis=0).reshape(-1)
        I=np.concatenate(I,axis=0).reshape(-1)
        cls_mat=np.concatenate(cls_mat,axis=0).reshape(-1)
        Ik=np.argsort(D)
        retrieved=collections.defaultdict(set)
        cl_cnt=len(cur_all_rels)
        i=0
        cnt=0
        Ik=Ik.tolist()
        while i<len(Ik) and cl_cnt>0 :
            j=Ik[i]
            rel_id=cls_mat[j].item()
            id=I[j].item()
            if len(retrieved[id2rel[rel_id]])==0:
                cl_cnt-=1
                retrieved[id2rel[rel_id]].add(id)
                cnt+=1
            i+=1
        i=0
        while i<len(Ik) and cnt<K :
            j=Ik[i]
            rel_id=cls_mat[j].item()
            id=I[j].item()
            if id not in retrieved[id2rel[rel_id]]:
                retrieved[id2rel[rel_id]].add(id)
                cnt+=1
            i+=1
        retrieved={k:list(v) for k,v in retrieved.items()}
        for k,v in retrieved.items():
            retrieved_res[k].extend(list(v))
            
        print(f"retrieved num is: {cnt},k is {K}")
        # indexes=list(retrieved)
        # ret_vec= np.vstack([self.index.reconstruct(i) for i in indexes ])
        # return list(retrieved) #K,H
    
    def batch_query(self,query_arr, index, k=1, nprobe = 300, ids=None):
        index.nprobe = nprobe
        D, I = index.search(query_arr, k)
        # if ids is not None:
        #     I = ids[I]
        return D,I #n,k