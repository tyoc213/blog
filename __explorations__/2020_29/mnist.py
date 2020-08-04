# %%
from fastai2.vision.all import *

# %%
path = untar_data(URLs.MNIST)
datablock = DataBlock(
    blocks=(ImageBlock,CategoryBlock),
    get_items=get_image_files,
    get_y=parent_label,
    splitter=GrandparentSplitter(train_name='training',valid_name='testing'),
#    item_tfms=Resize(28),
    batch_tfms=aug_transforms(size=224)
)
datablock.summary(path)

# %%
dls_tpu = datablock.dataloaders(path)

# %%
NUM_PIXELS=3 # 1 single channedl 3 rgb
NUM_OUTPUTS=10 #2 for MNIST_TINY 10 MNIST
class MyLenet(nn.Module):
    """Lenet with convs and F.max_pool2d"""
    def __init__(self):
        super(MyLenet, self).__init__()
        self.conv1 = nn.Conv2d(NUM_PIXELS, 6, 3) # set 3 for first item if RGB
        print(self.conv1.shape)
        self.conv2 = nn.Conv2d(6,16,3)
        print(self.conv2.shape)
        self.hiden4 = nn.Linear(400, NUM_OUTPUTS) # 2 outputs (3 and 7) instead of 10
        print(self.hidden4.shape)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.hiden4(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Lenet2(nn.Module):
    """Lenet with layers"""
    def __init__(self):
        super(Lenet2, self).__init__()
        self.conv1 = nn.Conv2d(NUM_PIXELS, 6, 3) # set 3 for first item if RGB
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(46656, 120) #(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_OUTPUTS) # Only 2 outputs (3 and 7) instead of 10
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# %%
def print_local(msg):
  if True: return
  print(msg)

class CallbackXLA(Callback):
  def after_step(self):
    xm.optimizer_step(self.opt, barrier=True)


class XLAOptimProxy:
    def __init__(self,opt:Optimizer):
        self.opt = opt

    def xla_step(self):
        xm.optimizer_step(self.opt,barrier=True) # sync on gradient update

    def __getattr__(self,name):
        if name == 'stepfsdfsdfsd': # override proxying for step method
                print_local("calling xla_step")
                return getattr(self,'xla_step')
        # proxy everything else
        print_local(f"calling {name}")
        return getattr(self.opt,name)


@patch_to(ParamScheduler)
def _update_val(self, pct):
#        for n,f in self.scheds.items(): self.opt.set_hyper(n, f(pct))
        for n,f in self.scheds.items():
            v = f(pct)
            print_local(f"---------------------- A f(pct) = {v}")
            self.opt.set_hyper(n, v)

@patch_to(ParamScheduler)
def after_batch(self):
#        for p in self.scheds.keys(): self.hps[p].append(self.opt.hypers[-1][p])
        for p in self.scheds.keys():
            v = self.opt.hypers[-1][p]
            print_local(f"---------------------- B after_batch ParamScheduler {v}")
            self.hps[p].append(v)
@patch_to(Learner)
def create_opt(self):
        print_local("create_opt!!!")
        ooo = self.opt_func(self.splitter(self.model), lr=self.lr)
        prox = XLAOptimProxy(ooo)
        self.opt = prox
        if not self.wd_bn_bias:
            for p in self._bn_bias_state(True ): p['do_wd'] = False
        if self.train_bn:
            for p in self._bn_bias_state(False): p['force_train'] = True

proxyLearn = Learner(dls_tpu, Lenet2(), metrics=accuracy, opt_func=Adam)#, cbs=CallbackXLA)

# %%
%%time
proxyLearn.fit(1, 10e-3) # 0.05) NOTE: Im not sure if this works...!!! it is now 96!

# %%
%%time
proxyLearn.lr_find()

# %%
print("--------------------\n"*100)

# %%
