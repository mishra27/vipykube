#!/usr/bin/env python
# coding: utf-8

# ## Import dependencies
# 

# In[31]:


import os, json, torch, datetime, pickle, copy, shutil, time
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import os
import random
import math
import re
#random.seed(1996)


import cv2
from extract_features import features






# In[32]:


MAX_TOKEN = 14
IMG_FEAT_PAD_SIZE = 100
CKPTS_PATH = './ckpts/'

CKPT_EPOCH = 13
CKPT_VERSION= 'default_model'
MODEL = 'small'
RUN_MODE = 'val'
N_GPU = 0
DEVICES = 0

# Set Devices
# If use multi-gpu training, set e.g.'0, 1, 2' instead
GPU = '1'


# Resume training
RESUME = False


# Print loss every step
VERBOSE = True


# A external method to set train split
TRAIN_SPLIT = 'train+val+vg'

# Set True to use pretrained word embedding
# (GloVe: spaCy https://spacy.io/)
USE_GLOVE = True

# Word embedding matrix size
# (token size x WORD_EMBED_SIZE)
WORD_EMBED_SIZE = 300

# Max length of question sentences
MAX_TOKEN = 14

# Filter the answer by occurrence
# self.ANS_FREQ = 8

# Max length of extracted faster-rcnn 2048D features
# (bottom-up and Top-down: https://github.com/peteanderson80/bottom-up-attention)
IMG_FEAT_PAD_SIZE = 100

# Faster-rcnn 2048D features
IMG_FEAT_SIZE = 2048

# Default training batch size: 64
BATCH_SIZE = 64

# Multi-thread I/O
NUM_WORKERS = 8

# Use pin memory
# (Warning: pin memory can accelerate GPU loading but may
# increase the CPU memory usage when NUM_WORKS is large)
PIN_MEM = True

# Large model can not training with batch size 64
# Gradient accumulate can split batch to reduce gpu memory usage
# (Warning: BATCH_SIZE should be divided by GRAD_ACCU_STEPS)
GRAD_ACCU_STEPS = 1

# Use a small eval batch will reduce gpu memory usage
EVAL_BATCH_SIZE = 32

# Set 'external': use external shuffle method to implement training shuffle
# Set 'internal': use pytorch dataloader default shuffle method
SHUFFLE_MODE = 'external'


# ------------------------
# ---- Network Params ----
# ------------------------

# Model deeps
# (Encoder and Decoder will be same deeps)
LAYER = 6

# Model hidden size
# (512 as default, bigger will be a sharp increase of gpu memory usage)
HIDDEN_SIZE = 512

# Multi-head number in MCA layers
# (Warning: HIDDEN_SIZE should be divided by MULTI_HEAD)
MULTI_HEAD = 8

# Dropout rate for all dropout layers
# (dropout can prevent overfitting： [Dropout: a simple way to prevent neural networks from overfitting])
DROPOUT_R = 0.1

# MLP size in flatten layers
FLAT_MLP_SIZE = 512

# Flatten the last hidden to vector with {n} attention glimpses
FLAT_GLIMPSES = 1
FLAT_OUT_SIZE = 1024

FF_SIZE = 2048
HIDDEN_SIZE_HEAD = 64


# In[33]:


# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------



class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# In[34]:


# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------


class MHAtt(nn.Module):
    def __init__(self):
        super(MHAtt, self).__init__()

        self.linear_v = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_k = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_q = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_merge = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

        self.dropout = nn.Dropout(DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            MULTI_HEAD,
            HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            MULTI_HEAD,
            HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            MULTI_HEAD,
            HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=HIDDEN_SIZE,
            mid_size=FF_SIZE,
            out_size=HIDDEN_SIZE,
            dropout_r=DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self):
        super(SA, self).__init__()

        self.mhatt = MHAtt()
        self.ffn = FFN()

        self.dropout1 = nn.Dropout(DROPOUT_R)
        self.norm1 = LayerNorm(HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(DROPOUT_R)
        self.norm2 = LayerNorm(HIDDEN_SIZE)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt()
        self.mhatt2 = MHAtt()
        self.ffn = FFN()

        self.dropout1 = nn.Dropout(DROPOUT_R)
        self.norm1 = LayerNorm(HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(DROPOUT_R)
        self.norm2 = LayerNorm(HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(DROPOUT_R)
        self.norm3 = LayerNorm(HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA() for _ in range(LAYER)])
        self.dec_list = nn.ModuleList([SGA() for _ in range(LAYER)])

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)

        return x, y


# In[35]:


# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------




# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self):
        super(AttFlat, self).__init__()
        

        self.mlp = MLP(
            in_size=HIDDEN_SIZE,
            mid_size=FLAT_MLP_SIZE,
            out_size=FLAT_GLIMPSES,
            dropout_r=DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            HIDDEN_SIZE *FLAT_GLIMPSES,
            FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------


class Net(nn.Module):
    def __init__(self, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=WORD_EMBED_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.img_feat_linear = nn.Linear(
            IMG_FEAT_SIZE,
            HIDDEN_SIZE
        )

        self.backbone = MCA_ED()

        self.attflat_img = AttFlat()
        self.attflat_lang = AttFlat()

        self.proj_norm = LayerNorm(FLAT_OUT_SIZE)
        self.proj = nn.Linear(FLAT_OUT_SIZE, answer_size)


    def forward(self, img_feat, ques_ix):

        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = torch.sigmoid(self.proj(proj_feat))

        return proj_feat


    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)


# ## Choose ques and answer

# In[36]:


images = []


"""
st.title("VQA tool using Streamlit")
html_temp = """
#<div style="background-color:tomato;padding:10px">
#<h2 style="color:white;text-align:center;">ViPyKube ML App </h2>
#</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)
st.sidebar.title("Image selection and Question")
button = st.sidebar.radio('Randomly generate images',('With predefined questons','With custom question'))
#while True:
"""

img_path_list = []



for x in range(0,1):

    name = random.choice(os.listdir('./datasets/coco_extract/images'))
    print(str(name))

    name = './datasets/coco_extract/images/'+name


    #image = Image.open(name)
    img_path_list.append(name)
    #images.append(image)


"""
image_iterator = paginator("Select a sunset page", images)
indices_on_page, images_on_page = map(list, zip(*image_iterator))
st.image(images_on_page, width=200, caption=indices_on_page)
pick_img = st.sidebar.selectbox("Which image?", 
    [x for x in range(1, len(images))])
"""

img_path = img_path_list[0]

im = cv2.imread(img_path)
       
f = features(im)        
roi = f.doit()


"""
print("chosen image " ,img_path)
image = Image.open(img_path)
st.header("Selected Image")
st.image(image)
"""

img_id = img_path[44:-4]
for ix in range(len(img_id)):
    if img_id[ix] !="0":
        img_id = img_id[ix:]
        print(img_id)
        break

q_list = []

"""
if button  == "With predefined questons":
    pass
elif button == "With custom question":
    question = st.sidebar.text_input("What is your question?")
    q_list = [{"image_id": int(img_id), "question": question, "question_id": 1}]
start_eval = st.sidebar.button('Get the answer!')
"""

print('Loading testing set ........')


# ## Create dataset

# In[37]:


given_list=q_list
path_image = img_path[31:]   

ques_list = []
ans_list = []


if len(given_list)>0:
    ques_list = given_list
else:      
    temp = json.load(open( './datasets/vqa/v2_OpenEnded_mscoco_val2014_questions.json', 'r'))['questions']
    print("\n")
    print("here")
    #if temp["image_id"] == int(self.img_id):
    ques_list += temp

    temp = []
    for x in range(len(ques_list)):
        if ques_list[x]["image_id"] == int(img_id):
            temp.append(ques_list[x])
    ques_list=temp


print("QUESTION LIST ",ques_list[0])

data_size = ques_list.__len__()

print('== Dataset size:', data_size)



qid_to_ques = {}

for ques in ques_list:
    qid = str(ques['question_id'])
    qid_to_ques[qid] = ques


# {question id} -> {question}
qid_to_ques = qid_to_ques

# Tokenize
# self.token_to_ix, self.pretrained_emb = tokenize(self.stat_ques_list, __C.USE_GLOVE)
with open('token2ix.json') as json_file:
    token_to_ix=json.load(json_file)
with open('pretrained_emb.npy', 'rb') as npf:
    pretrained_emb = np.load(npf)
token_size = token_to_ix.__len__()
print('== Question token vocab size:', token_size)


ans_to_ix, ix_to_ans = json.load(open('answer_dict.json', 'r'))

ans_size = ans_to_ix.__len__()
print('== Answer vocab size (occurr more than {} times):'.format(8), ans_size)
print('Finished!')
print('')


# ## Load pretrained model 

# In[38]:


valid=True
state_dict=None


# Load parameters
path = CKPTS_PATH +         'ckpt_' + CKPT_VERSION +         '/epoch' + str(CKPT_EPOCH) + '.pkl'

val_ckpt_flag = False
if state_dict is None:
    val_ckpt_flag = True
    print('Loading ckpt {}'.format(path))
    state_dict = torch.load(path,map_location= torch.device('cpu'))['state_dict']
    print('Finish!')

# Store the prediction list
qid_list = [ques['question_id'] for ques in ques_list]
q_list = [ques['question'] for ques in ques_list]
im_id_list =  [ques['image_id'] for ques in ques_list]


ans_ix_list = []
pred_list = []


net = Net(
    pretrained_emb,
    token_size,
    ans_size
)

net.eval()


if N_GPU > 1:
    net = nn.DataParallel(net, device_ids=DEVICES)

from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
    
# load params
net.load_state_dict(new_state_dict)


# ## Process image features and question features

# In[39]:



# Load the run data from list
test = torch.empty(14)
i = 0

img_tensor = []
ans_tensor =[]
ques_tensor = []

for ques in ques_list:
    i = i +1
    img_feat_iter = np.zeros(1)
    ques_ix_iter = np.zeros(1)
    
    img_feat_x = roi.cpu()
    
    
    #Process image feature 
    if img_feat_x.shape[0] >  IMG_FEAT_PAD_SIZE:
        img_feat_x = img_feat_x[: IMG_FEAT_PAD_SIZE]
    img_feat_x = np.pad(
        img_feat_x,
        ((0, IMG_FEAT_PAD_SIZE - img_feat_x.shape[0]), (0, 0)),
        mode='constant',
        constant_values=0
    )
    img_feat_iter=img_feat_x
 

    # Process question   
    ques_ix = np.zeros(MAX_TOKEN, np.int64)
    words = re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        ques['question'].lower()
    ).replace('-', ' ').replace('/', ' ').split()
    for ix, word in enumerate(words):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix + 1 == MAX_TOKEN:
            break   
    ques_ix_iter=ques_ix
    
    
    img_tensor.append(torch.from_numpy(img_feat_iter))
    ques_tensor.append(torch.from_numpy(ques_ix_iter))
     

img_feat_iter = (torch.stack(img_tensor))
ques_ix_iter = (torch.stack(ques_tensor))

print("Image and question tensor created")


# ## Prediction

# In[40]:



pred = net(
    img_feat_iter,
    ques_ix_iter
)

pred_np = pred.cpu().data.numpy()
pred_argmax = np.argmax(pred_np, axis=1)

# Save the answer index
if pred_argmax.shape[0] != EVAL_BATCH_SIZE:
    pred_argmax = np.pad(
        pred_argmax,
        (0, EVAL_BATCH_SIZE - pred_argmax.shape[0]),
        mode='constant',
        constant_values=-1
    )

ans_ix_list.append((pred_argmax))

ans_ix_list = np.array(ans_ix_list).reshape(-1)
old = 0
#st.header("Question and Answer")
for qix in range(qid_list.__len__()):
    question_id = int(qid_list[qix])
    answer =  ix_to_ans[str(ans_ix_list[qix])]

    if old != int(im_id_list[qix]):

        num_digit = len(str(im_id_list[qix]))
        name = 'COCO_val2014_'
        for x in range(0, 12-num_digit):
            name = name+'0'
        #image = Image.open('./datasets/coco_extract/images/'+name+str(im_id_list[qix])+'.jpg')
        cap =  q_list[qix]+ "            "+ answer
       # st.image(image)
        old = int(im_id_list[qix])

    q_a = ("("+str(qix+1)+")   "+q_list[qix]+ "          "+ answer)
   # st.subheader(q_a)
    print(q_a)



    result = [{
    'answer': answer,  # ix_to_ans(load with json) keys are type of string   
    'question_id': question_id}]