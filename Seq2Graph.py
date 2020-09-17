import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_scatter import scatter

from WGAT import WGATConv


class Seq2Graph(nn.Module):

    def __init__(self, emb_dim, hidden_size, num_heads, dropout, node_num, weighted):
        super(Seq2Graph, self).__init__()

        self.weighted = weighted

        if weighted:
            self.graph_layer1 = WGATConv(in_channels=emb_dim, out_channels=hidden_size, heads=num_heads, dropout=dropout)
            self.graph_layer2 = WGATConv(in_channels=hidden_size, out_channels=hidden_size, heads=num_heads, dropout=dropout)
        else:
            self.graph_layer1 = GATConv(in_channels=emb_dim, out_channels=hidden_size, heads=num_heads, dropout=dropout,
                                        concat=True, add_self_loops=False)
            self.graph_layer2 = GATConv(in_channels=hidden_size * num_heads, out_channels=hidden_size, heads=num_heads,
                                        dropout=dropout,
                                        concat=False, add_self_loops=False)

        self.node_embs = nn.Embedding(num_embeddings=node_num, embedding_dim=emb_dim)

        self.pred_layer = nn.Sequential(nn.Linear(in_features=hidden_size, out_features=node_num), nn.Sigmoid())


    def predict(self, graph_batch, hs):
        """
            use the average hidden state of all nodes in a sequence as the sequence representation
        :param graph_batch:
        :param hs:  node_num x hidden_size
        """

        # batch_size x hidden_size
        hs = scatter(src=hs, index=graph_batch, dim=0, reduce="mean")

        # batch_size x node_num
        scores = self.pred_layer(hs)

        return scores


    def forward(self, graphs):

        hs = self.node_embs(graphs.x)
        # add position embedding
        hs = hs + graphs.pos_emb.reshape(hs.shape[0], 1)

        if self.weighted:
            # node_num x hidden_size
            hs = self.graph_layer1(hs, graphs.edge_index, graphs.edge_attr)
            # node_num x hidden_size
            hs = self.graph_layer2(hs, graphs.edge_index, graphs.edge_attr)
        else:
            # node_num x (hidden_size * num_heads)
            hs = self.graph_layer1(hs, graphs.edge_index)
            # node_num x hidden_size
            hs = self.graph_layer2(hs, graphs.edge_index)

        scores = self.predict(graphs.batch, hs)

        return scores
