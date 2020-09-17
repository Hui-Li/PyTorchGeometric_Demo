from torch.utils.data import DataLoader
from Seq2Graph import Seq2Graph
from ToyDataset import GraphData, construct_weighted_graph

if __name__ == "__main__":

    raw_data = [
                [1,3,0,2],
                [2,3,1,3],
                [2,3,2,1],
                [0,3,2,2]
               ]

    batch_size = 2
    emb_dim = 8
    hidden_size = 8
    num_heads = 2
    dropout = 0
    node_num = 4

    graph_data = GraphData(raw_data=raw_data)

    dataset_loader = DataLoader(dataset=graph_data, batch_size=batch_size,
                                collate_fn=construct_weighted_graph, shuffle=False)

    model1 = Seq2Graph(emb_dim=emb_dim, hidden_size=hidden_size, num_heads=num_heads, dropout=dropout, node_num=node_num, weighted=False)

    print("--------- model 1 using GAT ----------")
    for graph_data in dataset_loader:
        scores = model1(graphs=graph_data)
        print(scores)


    model2 = Seq2Graph(emb_dim=emb_dim, hidden_size=hidden_size, num_heads=num_heads, dropout=dropout, node_num=node_num, weighted=True)

    print("--------- model 2 using Weighted GAT ----------")
    for graph_data in dataset_loader:
        scores = model2(graphs=graph_data)
        print(scores)
