import torch
import torch.nn as nn
from util.attention import MultiHeadAttention

class Agent_Embedding(nn.Module):
    def __init__(self, args):
        super(Agent_Embedding, self).__init__()
        self.input_dim = args.obs_dim
        self.agent_embedding_dim = args.agent_embedding_dim

        self.fc1 = nn.Linear(self.input_dim, self.input_dim)
        self.rnn_hidden = None
        self.rnn_fc = nn.GRUCell(self.input_dim, self.agent_embedding_dim)
        
    def forward(self, obs, detach=False):
        inputs = obs.view(-1, self.input_dim)
        fc1_out = torch.relu(self.fc1(inputs))
        self.rnn_hidden = self.rnn_fc(fc1_out, self.rnn_hidden)
        fc2_out = self.rnn_hidden
        if detach:
            fc2_out.detach()
        return fc2_out
    
class Agent_Embedding_Decoder(nn.Module):
    def __init__(self, args):
        super(Agent_Embedding_Decoder, self).__init__()
        self.agent_embedding_dim = args.agent_embedding_dim
        self.decoder_out_dim = args.obs_dim + args.N    # out_put: o(t+1)+agent_idx
        
        self.fc1 = nn.Linear(self.agent_embedding_dim, self.agent_embedding_dim)
        self.fc2 = nn.Linear(self.agent_embedding_dim, self.decoder_out_dim)

    def forward(self, agent_embedding):
        fc1_out = torch.relu(self.fc1(agent_embedding))
        decoder_out = self.fc2(fc1_out)
        return decoder_out
    

class Role_Embedding(nn.Module):
    def __init__(self, args):
        super(Role_Embedding, self).__init__()
        self.agent_embedding_dim = args.agent_embedding_dim
        self.role_embedding_dim = args.role_embedding_dim
        self.encoder = nn.ModuleList([nn.Linear(self.agent_embedding_dim, self.agent_embedding_dim),
                                      nn.Linear(self.agent_embedding_dim, self.role_embedding_dim)])
        self.target_encoder = nn.ModuleList([nn.Linear(self.agent_embedding_dim, self.agent_embedding_dim),
                                      nn.Linear(self.agent_embedding_dim, self.role_embedding_dim)])
        
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        
    def forward(self, agent_embedding, detach=False, ema=False):
        if ema: # target encoder
            output = torch.relu(self.target_encoder[0](agent_embedding))
            output = self.target_encoder[1](output)
        else:   # encoder
            output = torch.relu(self.encoder[0](agent_embedding))
            output = self.encoder[1](output)
        
        if detach:
            output.detach()
        return output
    
class Embedding_Net(nn.Module):
    def __init__(self, args):
        super(Embedding_Net, self).__init__()
        self.agent_embedding_net = Agent_Embedding(args)
        self.agent_embedding_decoder = Agent_Embedding_Decoder(args)
        self.role_embedding_net = Role_Embedding(args)
        self.W = nn.Parameter(torch.rand(args.role_embedding_dim, args.role_embedding_dim))

    def role_embed_foward(self, agent_embedding, detach=False, ema=False):
        return self.role_embedding_net(agent_embedding, detach, ema)
    
    def agent_embed_forward(self, obs, detach=False):
        return self.agent_embedding_net(obs, detach)
    
    def encoder_decoder_forward(self, obs):
        agent_embedding = self.agent_embed_forward(obs, detach=False)
        decoder_out = self.agent_embedding_decoder(agent_embedding)
        return decoder_out

class ACORM_Actor(nn.Module):
    def __init__(self, args):
        super(ACORM_Actor, self).__init__()
        self.args = args
        self.embedding_net = Embedding_Net(args)
        self.actor_input_dim = args.agent_embedding_dim + args.role_embedding_dim
        self.actor_net = nn.ModuleList([nn.Linear(self.actor_input_dim, self.actor_input_dim),
                                    nn.Linear(self.actor_input_dim, args.action_dim)])
        
    def actor_forward(self, agent_embedding, role_embedding, avail_a_n):
        actor_input = torch.cat([agent_embedding, role_embedding], dim=-1)
        output = torch.relu(self.actor_net[0](actor_input))
        output = self.actor_net[1](output)
        output[avail_a_n==0] = -1e10    # mask the unavailable action
        prob = torch.softmax(output, dim=-1)
        return prob
    
    def forward(self, obs, avail_a_n):
        agent_embed = self.embedding_net.agent_embed_forward(obs, detach=True)
        role_embed = self.embedding_net.role_embed_foward(agent_embed, detach=True, ema=False)
        prob = self.actor_forward(agent_embed, role_embed, avail_a_n)
        return prob


class ACORM_Critic(nn.Module):
    def __init__(self, args):
        super(ACORM_Critic, self).__init__()
        self.N = args.N       
        self.att_out_dim = args.att_out_dim
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.state_dim = args.state_dim
        
        self.state_gru_net = nn.ModuleList([nn.Linear(args.state_dim, args.state_dim),
                                            nn.GRUCell(args.state_dim, args.N*args.rnn_hidden_dim)])
        self.state_gru_hidden = None
        self.attention_net = MultiHeadAttention(args.n_heads, args.att_dim, args.att_out_dim, args.soft_temp, 
                                                args.rnn_hidden_dim, args.role_embedding_dim, args.role_embedding_dim)
        self.obs_gru_net = nn.ModuleList([nn.Linear(args.obs_dim, args.rnn_hidden_dim),
                                          nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)])
        self.obs_gru_hidden = None
        self.fc_final = nn.ModuleList([nn.Linear(args.rnn_hidden_dim+self.N*args.rnn_hidden_dim+self.N*args.att_out_dim, 2*args.rnn_hidden_dim),
                                       nn.Linear(2*args.rnn_hidden_dim, 1)])

    def state_forward(self, state):
        fc_out = torch.relu(self.state_gru_net[0](state))
        self.state_gru_hidden = self.state_gru_net[1](fc_out, self.state_gru_hidden)    
        return self.state_gru_hidden    # (batch, rnn_dim)
    
    def att_forward(self, tau_s, role_embeddings):
        # tau_s.shape=(batch, N, rnn_hidden_dim), role_embeddings.shape=(batch, N, role_dim)
        return self.attention_net(tau_s, role_embeddings, role_embeddings)  # output.shape=(batch, N, att_out_dim)
    
    def obs_forward(self, obs):
        # x = torch.cat([obs],dim=-1)
        x = torch.relu(self.obs_gru_net[0](obs))    
        self.obs_gru_hidden = self.obs_gru_net[1](x, self.obs_gru_hidden)
        return self.obs_gru_hidden  # (batch*N, rnn_dim)
    
    def critic_forward(self, tau_obs, tau_s, att_out):
        x = torch.cat([tau_obs, tau_s, att_out], dim=-1)
        x = torch.relu(self.fc_final[0](x))
        value = self.fc_final[1](x)
        return value
    
    def forward(self, obs, state, role_embeding):
        tau_s = self.state_forward(state)   # (batch, state_dim)-> (batch, N*rnn_dim)
        tau_s = tau_s.reshape(-1, self.N, self.rnn_hidden_dim)    # (batch, N, rnn_dim)
        att = self.att_forward(tau_s, role_embeding).unsqueeze(1).repeat(1,self.N,1,1).reshape(-1, self.N*self.att_out_dim)    # (batch, N, att_out_dim)->(batch,N,N,att_out_dim)->(batch*N, att_out_dim)
        # tau_obs = self.obs_forward(obs, state.unsqueeze(1).repeat(1,self.N,1).reshape(-1,self.state_dim))     # (batch*N, obs_dim)
        tau_obs = self.obs_forward(obs)     # (batch*N, obs_dim)
        value = self.critic_forward(tau_obs, tau_s.unsqueeze(1).repeat(1,self.N,1,1).reshape(-1,self.N*self.rnn_hidden_dim), att)   # (batch*N, 1)
        return value