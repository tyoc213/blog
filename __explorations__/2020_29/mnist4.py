#%%
### %%heat
#import pdb
print("---------------------------------------------------------- START")
import torch
#import torch_xla.core.xla_model as xm

#tpu_device = xm.xla_device()
tpu_device = torch.device('cuda:0') #torch.cuda.device("cuda:0")#.current_device()
print(f"device is {tpu_device}")
print("---------------------------------------------------------- START:1")
from fastai2.vision.all import *

# %%
def print_local(msg):
  if False: return
  print(msg)

print("---------------------------------------------------------- START:5")

class XLAOptimProxy:
    def __init__(self,opt:Optimizer):
        #print("XLAOptimProxy#inicializando __init__")
        self.opt = opt

    def xla_step(self):
        #print("------------- xla optimizer!!!!!!!! BARRIER TRYE")
        #xm.optimizer_step(self.opt,barrier=True) # sync on gradient update
        self.opt.step()

    def __getattr__(self,name):
        if name == 'step': # override proxying for step method
            #print_local("calling xla_step")
            return getattr(self,'xla_step')
        # proxy everything else
        #print_local(f"calling XLAOptimProxy#{name}")
        return getattr(self.opt,name)


def sort_by_run(fs):
    end = L(fs).attrgot('toward_end')
    inp,res = L(fs)[~end] + L(fs)[end], L()
    while len(inp):
        for i,o in enumerate(inp):
            if _is_first(o, inp):
                res.append(inp.pop(i))
                break
        else: raise Exception("Impossible to sort")
    # TRACE: print(f"will invoke toward_end {res}")
    return res


print("---------------------- PASSED LEARNER")
################################################################33





from datetime import datetime

@patch_to(Learner)
def one_batch(self, i, b):
# TRACE:         print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}                   BATCH {i}")
        self.iter = i
        try:
            # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}                   SPLIT {i}")
            self._split(b);                                  self('begin_batch')
            # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}                 {i} called begin_batch")
            self.pred = self.model(*self.xb);                self('after_pred')
            # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}                 {i} called after_pred")
            if len(self.yb) == 0: return
            # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}                 {i} did not return becasue yb.length == 0")
            self.loss = self.loss_func(self.pred, *self.yb); self('after_loss')
            # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}                 {i} called after_loss")
            if not self.training: return
            # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}                 {i} did not return because we are training")
            self.loss.backward();                            self('after_backward')
            # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}                 {i} called after_backward")
            self.opt.step();                                 self('after_step')
            #xm.optimizer_step(self.opt,barrier=True);        self('after_step')
            # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}                 {i} called after_step")
            self.opt.zero_grad()
            # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}                 {i} called zero_grad!!!!!!!")
        except CancelBatchException:
            self('after_cancel_batch')
            # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}                 {i} called after_cancel_batch")
        finally:
            if False:
                # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}                 {i} AFTER BATCH NOT CALLED")
                # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}                 {i} AFTER BATCH NOT CALLED")
                # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}                 {i} AFTER BATCH NOT CALLED")
                # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}                 {i} AFTER BATCH NOT CALLED")
                print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}                 {i} AFTER BATCH NOT CALLED")
            else:
                # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}                 {i} WILL CALL after_batch")
                self('after_batch')
                # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}                 {i} called after_batch")





from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter,_SingleProcessDataLoaderIter,_DatasetKind
_loaders = (_MultiProcessingDataLoaderIter,_SingleProcessDataLoaderIter)


@patch_to(DataLoader)
def __iter__(self):
        # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}  DataLoader#DataLoader#DataLoader#__iter__                         0")
        self.randomize()
        self.before_iter()
        # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}  DataLoader#DataLoader#DataLoader#__iter__ START FOR               1")
        for b in _loaders[self.fake_l.num_workers==0](self.fake_l):
            if self.device is not None:
                # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}  DataLoader#DataLoader#DataLoader#iterator to device from {b[0].device} y {b[1].device} to {self.device}")
                b = to_device(b, self.device)
                # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}  DataLoader#DataLoader#DataLoader#iterator to done!!!!")
            # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}  DataLoader#DataLoader#DataLoader#yielding                    3!!!! yield self.after_batch({b[0].device}) len of b is {len(b)}")
            yield self.after_batch(b)
            # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}  DataLoader#DataLoader#DataLoader#yielding                    4!!!!")
        # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}  DataLoader#DataLoader#DataLoader#__iter__ END FOR                 2")
        self.after_iter()
        # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}   DataLoader#DataLoader#DataLoader#__iter__ after ITER")
        if hasattr(self, 'it'): delattr(self, 'it')
        # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}     DataLoader#DataLoader#DataLoader#END __iter__")



@patch_to(Learner)
def all_batches(self):
        # TRACE: print("                                          Learner#ALL_BATCHES")
        self.n_iter = len(self.dl)
        for o in enumerate(self.dl):
            # TRACE: print("                                          Learner#ALL_BATCHES CALL ENTER")
            self.one_batch(*o)
            # TRACE: print("=======================================================================")
            # TRACE: print("=======================================================================")
            # TRACE: print("=======================================================================")
            # TRACE: print("                                          Learner#ALL_BATCHES CALL EXIT")
            # TRACE: print("=======================================================================")
            # TRACE: print("=======================================================================")
            # TRACE: print("=======================================================================")
        





@patch_to(Learner)
def create_opt(self):
        ooo = self.opt_func(self.splitter(self.model), lr=self.lr)
        prox = XLAOptimProxy(ooo)
        self.opt = prox
        if not self.wd_bn_bias:
            for p in self._bn_bias_state(True ): p['do_wd'] = False
        if self.train_bn:
            for p in self._bn_bias_state(False): p['force_train'] = True

# %%
path = untar_data(URLs.PETS)/'images'
pat = r'(.+)_\d+.jpg$'
datablock = DataBlock(
    blocks=(ImageBlock,CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(seed=42),
    get_y=using_attr(RegexLabeller(pat),'name'),
    item_tfms=Resize(224),
    # batch_tfms=[]
    batch_tfms=aug_transforms(size=224,min_scale=0.75)
)
datablock.summary(path)
dls = datablock.dataloaders(path,bs=256, device=tpu_device)
print("CNN Learner")
learner = cnn_learner(dls, resnet34, metrics=accuracy)
print("FINE TUNE")
learner.fine_tune(1,base_lr=4e-3,freeze_epochs=2)
print("end FINE TUNE")

# %%


# %%
