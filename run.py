import argparse
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import torch.nn as nn
import pprint


def search_with_sbert(args):

    def load(args):
        """
        load text file
        """
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

        return -1 , -1

    def make_adj(embs):
        """
        make adjacency upper triangular matrix
        i,j element indicates Cosine similarity between Sentence_i's sbert vector and Sentence_j's sbert vector
        """
        size = len(embs)
        adj = torch.zeros(size,size)

        # Cosine similarity function with torh
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)

        for i in range(size):
            for j in range(i,size):
                adj[i,j] = cos(embs[i],embs[j])

        return adj

    def make_not_triangular_adj(adj):
        for i in range(1,adj.shape[0]):
            for j in range(i):
                adj[i][j] = adj[j][i]
        return adj


    def cal_avg_sim(adj):
        dim = adj.shape[0]
        return (torch.sum(adj) - dim) / ((dim * (dim - 1)) / 2)

    def save_adj(adj,path):
        """
        save adj matrix into csv file.
        """
        data = adj.to('cpu').detach().numpy().copy()
        np.savetxt(path, data, delimiter=',')

    print('...loading files...')
    sentence_list, label_list = load(args)
    if sentence_list == -1:
        print("Error:input file error")
        exit()

    print('...loading sbert models...')
    sbert_model = SentenceTransformer(args.model)

    print('...calculate embs...')
    embs = sbert_model.encode(sentence_list)
    embs = torch.from_numpy(embs.astype(np.float32)).clone()

    print('...make adj matrix...')
    adj = make_adj(embs)
    adj = make_not_triangular_adj(adj)

    print("|sentence list=")
    pprint.pprint(sentence_list)
    print("|label list={}".format(label_list))
    print("|embed dimension={}".format(embs.shape))
    print("|adj matrix=\n{}".format(adj))

    argsorted_adj = torch.argsort(adj,descending=True)
    print("|ranking=\n{}".format(argsorted_adj))

    for i in range(adj.shape[0]):
        tmp = []
        for j in range(1,adj.shape[0]):
            tmp.append(label_list[argsorted_adj[i][j]])
        print("| Sentence No.{:0=2},Label={}:{}".format(i,label_list[i], ''.join(str(tmp))))

    if args.save_csv:
        print('...saving adj matrix into csv...')
        save_adj(adj,"{}.csv".format(args.input.split('.')[0]))



def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', '-i', help='input text')
    parser.add_argument('--model', '-m',  default='paraphrase-multilingual-mpnet-base-v2' ,help='model of sentence-bert')
    parser.add_argument('--label_num', '-n', default=0 ,help='number of label. When you use positive and negative examples, you can specify "2"')
    parser.add_argument('--save_csv', '-csv', default=False ,help='If you want to save adj matrix into csv data, please assign this option to be True')

    args = parser.parse_args()

    search_with_sbert(args)



main()