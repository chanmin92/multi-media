import os
import pickle

from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def train():
    # load train dataset
    data = load_coco_data(data_path='./data', split='train', toy=False)
    print "load_train_Done"
    word_to_idx = data['word_to_idx']
    # load val dataset to print out bleu scores every epoch
    val_data = load_coco_data(data_path='./data', split='val', toy=False)
    print "load_val_Done"
    model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
                                       dim_hidden=512, n_time_step=16, prev2out=True,
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    solver = CaptioningSolver(model, data, val_data, n_epochs=20, batch_size=128, update_rule='adam',
                                          learning_rate=0.001, print_every=1000, save_every=1, image_path='./image/',
                                    pretrained_model=None, model_path='model/HighwayLSTM01_lstm/', test_model='model/HighwayLSTM01_lstm/model-10',
                                     print_bleu=True, log_path='log/')

    solver.train()

def test(toy=None):
    if toy == True:
        toy = "toy_"
    else:
        toy =""
    data_path = os.path.join('./data', 'train')
    with open(os.path.join(data_path, '%sword_to_idx.pkl' % toy), 'rb') as f:
        word_to_idx = pickle.load(f)

    val_data = load_coco_data(data_path='./data', split='val', toy=toy)
    model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
                             dim_hidden=512, n_time_step=16, prev2out=True,
                             ctx2out=True, alpha_c=1.0, selector=True, dropout=True)
    solver = CaptioningSolver(model, None, val_data, n_epochs=20, batch_size=128, update_rule='adam',
                              learning_rate=0.001, print_every=1000, save_every=1, image_path='./image/',
                              pretrained_model=None, model_path='model/HighwayLSTM01_lstm/', test_model='model/HighwayLSTM01_lstm/model-20',
                              print_bleu=True, log_path='log/')

    solver.test(val_data)



if __name__ == "__main__":
    train()
    test(toy=False)