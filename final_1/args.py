dataset_path = '/home/pkhorram/PascalVOC2012/' #'/home/pkhorram/VOCdevkit/VOC2007/',
dataset_type = 'voc'
validation_dataset = '/home/pkhorram/VOCdevkit/VOC2007/'
balance_data = 'store_true'
arg_net = 'vgg16-ssd'
freeze_base_net = 'store_true'
freeze_net = 'store_true'
mb2_width_mult = 1.0 
lr = 1e-3
momentum = 0.9
weight_decay = 5e-4
gamma = 0.1
base_net_lr = None
extra_layers_lr = None  
base_net = '/home/pkhorram/MLIP-Project-master/final_1/models/vgg16_reducedfc.pth'
#pretrained_ssd = 
resume = None
scheduler = 'multi-step'
milestones = '80,100'  
t_max = 120
batch_size = 32
num_epochs = 120
num_workers = 4
validation_epochs = 5 
debug_steps = 10
use_cuda = True
checkpoint_folder = 'models/'