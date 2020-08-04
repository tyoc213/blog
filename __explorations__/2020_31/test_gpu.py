# %%
from fastai2.vision.all import *

# %%
print("wawawa")
# %%
path = untar_data(URLs.MNIST_TINY)


batch_tfms = aug_transforms(max_zoom=1.1, max_warp=0., max_rotate=0., max_lighting=0.0,do_flip=False)
datablock = DataBlock(
    blocks=(ImageBlock(cls=PILImageBW),CategoryBlock),
    get_items=get_image_files,
    get_y=parent_label,
    splitter=GrandparentSplitter(),
    item_tfms=Resize(28),
    batch_tfms=batch_tfms
)
datablock.summary(path)



# %%

from torch import stack, zeros_like as t0, ones_like as t1
from torch.distributions.bernoulli import Bernoulli


def zoom_mat(x, min_zoom=1., max_zoom=1.1, p=0.5, draw=None, draw_x=None, draw_y=None, batch=False):
    "Return a random zoom matrix with `max_zoom` and `p`"
    def _def_draw(x):       return x.new(x.size(0)).uniform_(min_zoom, max_zoom)
    def _def_draw_b(x):     return x.new_zeros(x.size(0)) + random.uniform(min_zoom, max_zoom)
    def _def_draw_ctr(x):   return x.new(x.size(0)).uniform_(0,1)
    def _def_draw_ctr_b(x): return x.new_zeros(x.size(0)) + random.uniform(0,1)
    assert(min_zoom<=max_zoom)
    s = 1/_draw_mask(x, _def_draw_b if batch else _def_draw, draw=draw, p=p, neutral=1., batch=batch)
    def_draw_c = _def_draw_ctr_b if batch else _def_draw_ctr
    col_pct = _draw_mask(x, def_draw_c, draw=draw_x, p=1., batch=batch)
    row_pct = _draw_mask(x, def_draw_c, draw=draw_y, p=1., batch=batch)
    col_c = (1-s) * (2*col_pct - 1)
    row_c = (1-s) * (2*row_pct - 1)
    return affine_mat(s,     t0(s), col_c,
                      t0(s), s,     row_c)

# Cell
@delegates(zoom_mat)
@patch
def zoom(x: (TensorImage,TensorMask,TensorPoint,TensorBBox), size=None, mode='bilinear', pad_mode=PadMode.Reflection,
         align_corners=True, **kwargs):
    x0,mode,pad_mode = _get_default(x, mode, pad_mode)
    return x.affine_coord(mat=zoom_mat(x0, **kwargs)[:,:2], sz=size, mode=mode, pad_mode=pad_mode, align_corners=align_corners)

# Cell
def Zoom(min_zoom=1., max_zoom=1.1, p=0.5, draw=None, draw_x=None, draw_y=None, size=None, mode='bilinear',
         pad_mode=PadMode.Reflection, batch=False, align_corners=True):
    "Apply a random zoom of at most `max_zoom` with probability `p` to a batch of images"
    return AffineCoordTfm(partial(zoom_mat, min_zoom=min_zoom, max_zoom=max_zoom, p=p, draw=draw, draw_x=draw_x, draw_y=draw_y, batch=batch),
                          size=size, mode=mode, pad_mode=pad_mode, align_corners=align_corners)


# Cell
def aug_transforms(mult=1.0, do_flip=True, flip_vert=False, max_rotate=10., min_zoom=1., max_zoom=1.1,
                   max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75, xtra_tfms=None, size=None,
                   mode='bilinear', pad_mode=PadMode.Reflection, align_corners=True, batch=False, min_scale=1.):
    "Utility func to easily create a list of flip, rotate, zoom, warp, lighting transforms."
    res,tkw = [],dict(size=size if min_scale==1. else None, mode=mode, pad_mode=pad_mode, batch=batch, align_corners=align_corners)
    max_rotate,max_lighting,max_warp = array([max_rotate,max_lighting,max_warp])*mult
    if do_flip: res.append(Dihedral(p=0.5, **tkw) if flip_vert else Flip(p=0.5, **tkw))
    if max_warp:   res.append(Warp(magnitude=max_warp, p=p_affine, **tkw))
    if max_rotate: res.append(Rotate(max_deg=max_rotate, p=p_affine, **tkw))
    if min_zoom<1 or max_zoom>1: res.append(Zoom(min_zoom=min_zoom, max_zoom=max_zoom, p=p_affine, **tkw))
    if max_lighting:
        res.append(Brightness(max_lighting=max_lighting, p=p_lighting, batch=batch))
        res.append(Contrast(max_lighting=max_lighting, p=p_lighting, batch=batch))
    if min_scale!=1.: xtra_tfms = RandomResizedCropGPU(size, min_scale=min_scale, ratio=(1,1)) + L(xtra_tfms)
    return setup_aug_tfms(res + L(xtra_tfms))
























































# %%
dls = datablock.dataloaders(path,bs=1)

dls.device
train_dl = dls.train

# def fetch_all_batches():
#     global train_dl
#     for b in train_dl:
#         print(f"...{b}")

# fetch_all_batches()

for b in train_dl:
    print(f"{b}")
    break

# %%
print(train_dl)

# %%
myz = Zoom(min_zoom=0.5, max_zoom=1.1, p=1)
myz
# %%

i2 = TensorImage(i)
ift = IntToFloatTensor()
i2 = ift(i2)
i2 = to_device(i2, "cuda:0")
# %%
print("\n"*60)

myz(i2)

# %%
dir(ift)

# %%
