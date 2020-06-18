from core.data.load_data import DataSet
from core.model.net import Net
from core.model.optim import get_optim, adjust_lr
from core.data.data_utils import shuffle_list
from utils.vqa import VQA
from utils.vqaEval import VQAEval

import os, json, torch, datetime, pickle, copy, shutil, time
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import streamlit as st
from PIL import Image

from cfgs.base_cfgs import Cfgs
from core.exec import Execution
import argparse, yaml
import os
import random
random.seed(23)


def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='MCAN Args')

    parser.add_argument('--RUN', dest='RUN_MODE',
                      choices=['train', 'val', 'test'],
                      help='{train, val, test}',
                      type=str, required=True)

    parser.add_argument('--MODEL', dest='MODEL',
                      choices=['small', 'large'],
                      help='{small, large}',
                      default='small', type=str)

    parser.add_argument('--SPLIT', dest='TRAIN_SPLIT',
                      choices=['train', 'train+val', 'train+val+vg'],
                      help="set training split, "
                           "eg.'train', 'train+val+vg'"
                           "set 'train' can trigger the "
                           "eval after every epoch",
                      type=str)

    parser.add_argument('--EVAL_EE', dest='EVAL_EVERY_EPOCH',
                      help='set True to evaluate the '
                           'val split when an epoch finished'
                           "(only work when train with "
                           "'train' split)",
                      type=bool)

    parser.add_argument('--SAVE_PRED', dest='TEST_SAVE_PRED',
                      help='set True to save the '
                           'prediction vectors'
                           '(only work in testing)',
                      type=bool)

    parser.add_argument('--BS', dest='BATCH_SIZE',
                      help='batch size during training',
                      type=int)

    parser.add_argument('--MAX_EPOCH', dest='MAX_EPOCH',
                      help='max training epoch',
                      type=int)

    parser.add_argument('--PRELOAD', dest='PRELOAD',
                      help='pre-load the features into memory'
                           'to increase the I/O speed',
                      type=bool)

    parser.add_argument('--GPU', dest='GPU',
                      help="gpu select, eg.'0, 1, 2'",
                      type=str)

    parser.add_argument('--SEED', dest='SEED',
                      help='fix random seed',
                      type=int)

    parser.add_argument('--VERSION', dest='VERSION',
                      help='version control',
                      type=str)

    parser.add_argument('--RESUME', dest='RESUME',
                      help='resume training',
                      type=bool)

    parser.add_argument('--CKPT_V', dest='CKPT_VERSION',
                      help='checkpoint version',
                      type=str)

    parser.add_argument('--CKPT_E', dest='CKPT_EPOCH',
                      help='checkpoint epoch',
                      type=int)

    parser.add_argument('--CKPT_PATH', dest='CKPT_PATH',
                      help='load checkpoint path, we '
                           'recommend that you use '
                           'CKPT_VERSION and CKPT_EPOCH '
                           'instead',
                      type=str)

    parser.add_argument('--ACCU', dest='GRAD_ACCU_STEPS',
                      help='reduce gpu memory usage',
                      type=int)

    parser.add_argument('--NW', dest='NUM_WORKERS',
                      help='multithreaded loading',
                      type=int)

    parser.add_argument('--PINM', dest='PIN_MEM',
                      help='use pin memory',
                      type=bool)

    parser.add_argument('--VERB', dest='VERBOSE',
                      help='verbose print',
                      type=bool)

    parser.add_argument('--DATA_PATH', dest='DATASET_PATH',
                      help='vqav2 dataset root path',
                      type=str)

    parser.add_argument('--FEAT_PATH', dest='FEATURE_PATH',
                      help='bottom up features root path',
                      type=str)

    args = parser.parse_args()
    return args

def main():
    """Visual Qustion Answering
    Using Machine Learning and Streamlit
    """

    __C = Cfgs()

    args = parse_args()
    args_dict = __C.parse_to_dict(args)

    cfg_file = "cfgs/{}_model.yml".format(args.MODEL)
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.load(f)

    args_dict = {**yaml_dict, **args_dict}
    __C.add_args(args_dict)
    __C.proc()

    print('Hyper Parameters:')
    print(__C)


    

    st.title("VQA tool using Streamlit")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">ViPyKube ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

        
    images = []
    st.sidebar.title("Image selection and Question")

    button = st.sidebar.radio('Randomly generate images',('With predefined questons','With custom question'))


    #while True:

    img_path_list = []

    for x in range(0,10):
           
        name = random.choice(os.listdir('./datasets/coco_extract/images'))
        print(str(name))
        
        name = './datasets/coco_extract/images/'+name


        image = Image.open(name)
        img_path_list.append(name)
        images.append(image)


    image_iterator = paginator("Select a sunset page", images)
    indices_on_page, images_on_page = map(list, zip(*image_iterator))
    st.image(images_on_page, width=200, caption=indices_on_page)


        
    pick_img = st.sidebar.selectbox("Which image?", 
        [x for x in range(1, len(images))])
    

    imp_path = img_path_list[int(pick_img)]
    print("chosen image " ,imp_path)
    image = Image.open(imp_path)
    st.header("Selected Image")
    st.image(image)


    img_id = imp_path[44:-4]
    for ix in range(len(img_id)):
        if img_id[ix] !="0":
            img_id = img_id[ix:]
            print(img_id)
            break

    q_list = []
    if button  == "With predefined questons":
        pass
    elif button == "With custom question":
        question = st.sidebar.text_input("What is your question?")

        q_list = [{"image_id": int(img_id), "question": question, "question_id": 1}]
    
    start_eval = st.sidebar.button('Get the answer!')
    if start_eval:
        print('Loading testing set ........')

        dataset = DataSet(__C,q_list, imp_path[31:], img_id)
        eval(__C,dataset, valid=True)

def eval(hyper, dataset, state_dict=None, valid=False):


    # Load parameters
    if hyper.CKPT_PATH is not None:
        print('Warning: you are now using CKPT_PATH args, '
                'CKPT_VERSION and CKPT_EPOCH will not work')

        path = hyper.CKPT_PATH
    else:
        path = hyper.CKPTS_PATH + \
                'ckpt_' + hyper.CKPT_VERSION + \
                '/epoch' + str(hyper.CKPT_EPOCH) + '.pkl'

    val_ckpt_flag = False
    if state_dict is None:
        val_ckpt_flag = True
        print('Loading ckpt {}'.format(path))
        state_dict = torch.load(path)['state_dict']
        print('Finish!')

    # Store the prediction list
    qid_list = [ques['question_id'] for ques in dataset.ques_list]
    q_list = [ques['question'] for ques in dataset.ques_list]
    im_id_list =  [ques['image_id'] for ques in dataset.ques_list]


    ans_ix_list = []
    pred_list = []

    data_size = dataset.data_size
    token_size = dataset.token_size
    ans_size = dataset.ans_size
    pretrained_emb = dataset.pretrained_emb

    net = Net(
        hyper,
        pretrained_emb,
        token_size,
        ans_size
    )
    net.cuda()
    net.eval()

    if hyper.N_GPU > 1:
        net = nn.DataParallel(net, device_ids=hyper.DEVICES)

    net.load_state_dict(state_dict)

    dataloader = Data.DataLoader(
        dataset,
        batch_size=hyper.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=hyper.NUM_WORKERS,
        pin_memory=True
    )

    for step, (
            img_feat_iter,
            ques_ix_iter,
            ans_iter
    ) in enumerate(dataloader):
        print("\rEvaluation: [step %4d/%4d]" % (
            step,
            int(data_size / hyper.EVAL_BATCH_SIZE),
        ), end='          ')


        img_feat_iter = img_feat_iter.cuda()
        ques_ix_iter = ques_ix_iter.cuda()

        pred = net(
            img_feat_iter,
            ques_ix_iter
        )
        pred_np = pred.cpu().data.numpy()
        pred_argmax = np.argmax(pred_np, axis=1)



        # Save the answer index
        if pred_argmax.shape[0] != hyper.EVAL_BATCH_SIZE:
            pred_argmax = np.pad(
                pred_argmax,
                (0, hyper.EVAL_BATCH_SIZE - pred_argmax.shape[0]),
                mode='constant',
                constant_values=-1
            )

        ans_ix_list.append((pred_argmax))
        break
        #st.write(dataset.ix_to_ans[str([pred_argmax])])

        # Save the whole prediction vector
        if hyper.TEST_SAVE_PRED:
            if pred_np.shape[0] != hyper.EVAL_BATCH_SIZE:
                pred_np = np.pad(
                    pred_np,
                    ((0, hyper.EVAL_BATCH_SIZE - pred_np.shape[0]), (0, 0)),
                    mode='constant',
                    constant_values=-1
                )

            pred_list.append(pred_np)


    print('')
    ans_ix_list = np.array(ans_ix_list).reshape(-1)
    old = 0
    st.header("Question and Answer")
    for qix in range(qid_list.__len__()):
        bbb = int(qid_list[qix])
        aaa =  dataset.ix_to_ans[str(ans_ix_list[qix])]

        if old != int(im_id_list[qix]):
            
            num_digit = len(str(im_id_list[qix]))
            name = 'COCO_val2014_'
            for x in range(0, 12-num_digit):
                name = name+'0'
            image = Image.open('./datasets/coco_extract/images/'+name+str(im_id_list[qix])+'.jpg')
            cap =  q_list[qix]+ "            "+ aaa
           # st.image(image)
            old = int(im_id_list[qix])
        
        q_a = ("("+str(qix+1)+")   "+q_list[qix]+ "          "+ aaa)
        st.subheader(q_a)



        result = [{
        'answer': aaa,  # ix_to_ans(load with json) keys are type of string   
        'question_id': bbb}]

    # Write the results to result file
    if valid:
        if val_ckpt_flag:
            result_eval_file = \
                hyper.CACHE_PATH + \
                'result_run_' + hyper.CKPT_VERSION + \
                '.json'
        else:
            result_eval_file = \
                hyper.CACHE_PATH + \
                'result_run_' + hyper.VERSION + \
                '.json'

    else:
        if hyper.CKPT_PATH is not None:
            result_eval_file = \
                hyper.RESULT_PATH + \
                'result_run_' + hyper.CKPT_VERSION + \
                '.json'
        else:
            result_eval_file = \
                hyper.RESULT_PATH + \
                'result_run_' + hyper.CKPT_VERSION + \
                '_epoch' + str(hyper.CKPT_EPOCH) + \
                '.json'

        print('Save the result to file: {}'.format(result_eval_file))

    json.dump(result, open(result_eval_file, 'w'))

    # Save the whole prediction vector
    if hyper.TEST_SAVE_PRED:

        if hyper.CKPT_PATH is not None:
            ensemble_file = \
                hyper.PRED_PATH + \
                'result_run_' + hyper.CKPT_VERSION + \
                '.json'
        else:
            ensemble_file = \
                hyper.PRED_PATH + \
                'result_run_' + hyper.CKPT_VERSION + \
                '_epoch' + str(hyper.CKPT_EPOCH) + \
                '.json'

        print('Save the prediction vector to file: {}'.format(ensemble_file))

        pred_list = np.array(pred_list).reshape(-1, ans_size)
        result_pred = [{
            'pred': pred_list[qix],
            'question_id': int(qid_list[qix])
        }for qix in range(qid_list.__len__())]

        pickle.dump(result_pred, open(ensemble_file, 'wb+'), protocol=-1)

    
def empty_log(self, version):

    print('Initializing log file ........')
    if (os.path.exists(hyper.LOG_PATH + 'log_run_' + version + '.txt')):
        os.remove(hyper.LOG_PATH + 'log_run_' + version + '.txt')
    print('Finished!')
    print('')

def paginator(label, items, items_per_page=10, on_sidebar=True):



        # Display a pagination selectbox in the specified location.
        items = list(items)
        n_pages = len(items)
        n_pages = (len(items) - 1) // items_per_page + 1
        page_format_func = lambda i: "Page %s" % i

        # Iterate over the items in the page to let the user display them.
        min_index = 1
        max_index = min_index + items_per_page
        import itertools
        return itertools.islice(enumerate(items), min_index, max_index)


main()




    



