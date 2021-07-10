import argparse
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import torch.nn as nn

def load(args):
    with open(args.input, 'r') as f:
        file_data = f.readlines()

    if args.label_num != 0:
        sentence_list = []
        label_list = []
        for sentence in file_data:
            tmp = sentence.split()
            sentence_list.append(tmp[0])
            label_list.append(int(tmp[1]))
        return sentence_list, label_list
    else:
        sentence_list = [sentence.strip() for sentence in file_data]
        return sentence_list, None

    return -1

def search_with_sbert(args):
    print('loading files...')
    sentence_list, label_list = load(args)

    #print(f'{sentence_list=}')
    #print(f'{label_list=}')

    print('loading sbert models...')
    sbert_model = SentenceTransformer(args.model)

    print('calculate embs...')
    embs = sbert_model.encode(sentence_list)
    embs = torch.from_numpy(embs.astype(np.float32)).clone()

    print(embs.shape)

    print('make adj matrix...')
    adj = make_adj(embs)
    print(adj)
    save_adj(adj,"{}.csv".format(args.input.split('.')[0]))

    avg_sim = cal_avg_sim(adj)
    print(avg_sim)

def make_adj(embs):
    size = len(embs)
    adj = torch.zeros(size,size)
    print(adj)

    # Cosine similarity function with torh
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    for i in range(size):
        for j in range(i,size):
            adj[i,j] = cos(embs[i],embs[j])

    return adj

def cal_avg_sim(adj):
    dim = adj.shape[0]
    return (torch.sum(adj) - dim) / ((dim * (dim - 1)) / 2)


def save_adj(adj,path):
    data = adj.to('cpu').detach().numpy().copy()
    np.savetxt(path, data, delimiter=',')
    

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', '-i', help='input text')
    parser.add_argument('--model', '-m',  default='paraphrase-multilingual-mpnet-base-v2' ,help='model of sentence-bert')
    parser.add_argument('--label_num', '-n', default=0 ,help='number of label. When you use positive and negative examples, you can specify "2"')

    args = parser.parse_args()

    search_with_sbert(args)



main()