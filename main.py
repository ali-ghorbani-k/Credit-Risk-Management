import argparse
from model import classifier

parser = argparse.ArgumentParser(description='')
parser.add_argument('--use_ftools', dest='use_ftools', default=False, action='store_true', help='use automated feature extraction with ft')
parser.add_argument('--ft_maxdep', dest='ft_maxdep', default=1, type= int, help='Max depth of deep feature synthesis in feature tools')
parser.add_argument('--use_cnnft', dest='use_cnnft', default=False, action='store_true', help='use cnn feature extraction method')
parser.add_argument('--cnn_bsize', dest='cnn_bsize', default=256, type= int, help='batch_size when training cnn for feature extraction')
parser.add_argument('--cnn_epoch', dest='cnn_epoch', default=100, type= int, help='number of epochs when training cnn for feature extraction')
parser.add_argument('--use_rnnft', dest='use_rnnft', default=False, action='store_true', help='use rnn feature extraction method')
parser.add_argument('--rnn_bsize', dest='rnn_bsize', default=256, type= int, help='batch_size when training cnn for feature extraction')
parser.add_argument('--rnn_epoch', dest='rnn_epoch', default=100, type= int, help='number of epochs when training cnn for feature extraction')
parser.add_argument('--resample', dest='resample', default=False, action='store_true', help='resample training dataset to get balanced positive/negative label ratio')
parser.add_argument('--use_hclstr', dest='use_hclstr', default=False, action='store_true', help='use hierarchical clustering (undersampling) of majority class')
parser.add_argument('--use_hclstrsmote', dest='use_hclstrsmote', default=False, action='store_true', help='hierarchical clustering (undersampling) of majority class & Oversampling of minority class')
parser.add_argument('--nfolds', dest='nfolds', type=int, default=5, help='# of folds for cross-validation')
parser.add_argument('--test_size', dest='test_size', type=float, default=0.05, help='test to train data ratio')
parser.add_argument('--pca_n', dest='pca_n', type=int, default=150, help='number of pca components considered for training xgb')
parser.add_argument('--lgbm', dest='lgbm', default=True, action='store_true', help='use lightGBM algorithm')
parser.add_argument('--xgb',  dest='xgb' , default=False, action='store_true', help='use XGBoost algorithm')
parser.add_argument('--catb', dest='catb', default=False, action='store_true', help='use Catboost algorithm')
parser.add_argument('--fcnn', dest='fcnn', default=False, action='store_true', help='use fully connected neural network')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=256, help='batch size for FCNN algorithm')
parser.add_argument('--epoch', dest='epoch', type=int, default=10, help='number of epochs for FCNN algorithm')

args = parser.parse_args()

if __name__ == '__main__':
    model = classifier(args)
    model.train(args)
