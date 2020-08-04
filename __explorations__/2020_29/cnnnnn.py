# %%
from fastai2.vision.all import *

# %%

tpu_device = torch.device('cuda:0')


torch.cuda.empty_cache()


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



@patch_to(Learner)
def create_opt(self):
        ooo = self.opt_func(self.splitter(self.model), lr=self.lr)
        prox = XLAOptimProxy(ooo)
        self.opt = prox
        if not self.wd_bn_bias:
            for p in self._bn_bias_state(True ): p['do_wd'] = False
        if self.train_bn:
            for p in self._bn_bias_state(False): p['force_train'] = True



from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter,_SingleProcessDataLoaderIter,_DatasetKind
_loaders = (_MultiProcessingDataLoaderIter,_SingleProcessDataLoaderIter)

import inspect

@patch_to(DataLoader)
def __iter__(self):
        print("__iter__")
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
            #print(inspect.getsource(self.after_batch))
            print(type(self.after_batch))
            print(self.after_batch)
            yield self.after_batch(b)
            # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}  DataLoader#DataLoader#DataLoader#yielding                    4!!!!")
        # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}  DataLoader#DataLoader#DataLoader#__iter__ END FOR                 2")
        self.after_iter()
        # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}   DataLoader#DataLoader#DataLoader#__iter__ after ITER")
        if hasattr(self, 'it'): delattr(self, 'it')
        # TRACE: print(f"{datetime.now().strftime(' (%H:%M:%S.%f)')}     DataLoader#DataLoader#DataLoader#END __iter__")


#%%

path = untar_data(URLs.PETS)/'images'
pat = r'(.+)_\d+.jpg$'
datablock = DataBlock(
    blocks=(ImageBlock,CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(seed=42),
    get_y=using_attr(RegexLabeller(pat),'name'),
    item_tfms=aug_transforms(size=224,min_scale=0.75),
)
datablock.summary(path)
dls = datablock.dataloaders(path,bs=64, device=tpu_device)
print("CNN Learner")
learner = cnn_learner(dls, resnet34, metrics=accuracy)
print("FINE TUNE")
learner.fine_tune(1,base_lr=4e-3,freeze_epochs=2)
print("end FINE TUNE")


# %%
