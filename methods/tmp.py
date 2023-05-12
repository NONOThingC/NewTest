import collections
words=["xbc","pcxbcf","xb","cxbc","pcxbc"]
len2words=collections.defaultdict(list)
for word in words:
    len2words[len(word)].append(word)
word2cnt=collections.defaultdict(int)
ans=0
len_lists=sorted(len2words.keys())
for i in range(len(len_lists)-1):
    if len_lists[i]+1!=len_lists[i+1]:
        continue
    else:
        pre,cur=len2words[len_lists[i]],len2words[len_lists[i+1]]
        for x in pre:
            for y in cur:
                p_y,p_x=0,0
                while p_x<len(x) and p_y<len(y):
                    if y[p_y]==x[p_x]:
                        p_x+=1
                    p_y+=1
                if p_x==len(x):
                    word2cnt[y]=max(word2cnt[x]+1,word2cnt[y])
                    ans=max(ans,word2cnt[y])
print(ans+1)