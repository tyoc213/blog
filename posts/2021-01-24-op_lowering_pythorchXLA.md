---
aliases:
- /xla/2021/01/24/op_lowering_pythorchXLA
categories:
- xla
date: '2021-01-24'
description: Lowering SiLU operation in Pytorch/XLA
layout: post
title: Lowering SiLU in pytorch/XLA
toc: true

---

# Introduction

As a follow up of [compiling pytorch locally](https://tyoc213.github.io/blog/xla/fastai/2020/11/28/compiling-xla-locally.html), the next objective was to lower an operation, but the documentation on pytorch and XLA is almost the same (think of XLA as an extension of Pytorch), you can check the [OP_LOWERING_GUIDE](https://github.com/pytorch/xla/blob/master/OP_LOWERING_GUIDE.md) which basically is [OP LOWERING GUIDE](https://pytorch.org/xla/release/1.7/index.html#op-lowering-guide) from pytorch, probably you will get it at first hand, but I was not exactly sure what to do next for example I didn't know when to modify [gen.py](https://github.com/pytorch/xla/blob/master/scripts/gen.py) (which you don't need if it is already in [see](https://github.com/pytorch/xla/issues/2717#issuecomment-755007143) and other things.

## What says the op lowering guide?

Now that I have lowered an op, I think what the lowering guide says is something like this:

1. That you need to have an environment to compile/test/run
2. That you need to understand the operation [SiLU](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html)
3. That you need to implement new operations copying verbatim the signature from header and cpp files to `aten_xla_type.h/cpp` from `aten_xla_type_default.h/cpp`
4. You will use `XLATensor` tensors as input and output as pytorch tensor (ATen means "a tensor library")
5. From this level in `tensor_methods` will be `ir` ops that are constructed from pytorch to use the [tensorflow XLA compiler ops](https://www.tensorflow.org/xla/operation_semantics) (which I think it is not exhaustive? not all is output to the doc?). This operations include for example the `*` I used and [xla::Sigmoid](https://github.com/pytorch/xla/blob/3eaee46ef679cc6a0f1f694bd0a007dbfd09c51b/torch_xla/csrc/elementwise.cpp#L176-L181)

They also list 2 [Add inverse op and test](https://github.com/pytorch/xla/pull/1592) and [Implement the lowering for HardSigmoid and HardSigmoidBackward](https://github.com/pytorch/xla/pull/1940) which could help and are a good start too.

# The opportunity

I asked in which op I can contribute and [JackCaoG](https://github.com/JackCaoG) suggested the ones that were in the backlog, unfortunately they were probably not entry level, by fortune little time after the opportunity showed up  [here](https://github.com/pytorch/xla/issues/2717) and this SiLU op was good for entry level because as description says it uses as base Sigmoid which is already lowered also JackCaoG said it should be mostly like [sigmoid](https://github.com/pytorch/xla/blob/1a56d70a9a48446536912d80c6f929519453258e/torch_xla/csrc/tensor_methods.cpp#L2352-L2364) or [log_sigmoid](https://github.com/pytorch/xla/blob/1a56d70a9a48446536912d80c6f929519453258e/torch_xla/csrc/tensor_methods.cpp#L1590-L1602)  and from the description on the documentation it looked like that.

I really don't know much about pytorch and sure I didn't know of SiLU before, but the signature of SiLU was not like those provided as base, I finally checked the other ops that ended with `_out(` as example [arange_out](https://github.com/pytorch/xla/blob/1a56d70a9a48446536912d80c6f929519453258e/torch_xla/csrc/aten_xla_type.cpp#L566-L572)

# Implementation

As always, create a new branch and don't forget to update to the latest master in pytorch and XLA (which in my case caused some behaviour about synching the repos).

## 1. Create the base

First commit https://github.com/pytorch/xla/pull/2721/commits/c16fedbbee3662d3470629dc7fff51c63dd60855

It provides the base and starting point:

1. Copy the signature from the header and implementation of to ``
2. Copy the body from the header to .
3. It also reused at a higher level the Sigmoid as expected, the problem with this is that the generated graph for the compiler will list this as a `x * sigmoid(x)` (which was basically this `input.GetIrValue() * ir::ops::Sigmoid(input.GetIrValue());`) instead of a `SiLU` in the node graph.

But this implementation was good enough to compile without errors and actually run my rudimentary base test that output the same values for cpu implementation and XLA implementation

```python
import torch
from torch.nn import SiLU
import torch_xla.core.xla_model as xm


dede=xm.xla_device()
m = SiLU()
m = m.to(dede)

input = torch.randn(2)
input2 = input.clone() # this is on CPU
input = input.to(dede)

output = m(input)
print(output)
print(output.device) # should print xla

m2 = SiLU()
print("normal")
print('input2', input2)
o2 = m2(input2)
print(o2) # this should match print above
```

The review of the PR suggested the next step, which is:


## 2. Go deeper with the lowering

Now that you have a base go deeper and make your node appear.

Because when people is debugging the generated graph of tensor ops in XLA with the previous implementation it would be better if calling `SiLU` would generate a `SiLU` node and not `x * sigmoid(x)` as the previous step.

Second commit https://github.com/pytorch/xla/pull/2721/commits/c16fedbbee3662d3470629dc7fff51c63dd60855


It shows how to add an op that will be used as a node in the generated graph for the operation.

1. It adds the operation to `ops.h/cpp`
2. It converts from tensors to XLA tensors and back.
3. It reuses at this level the implementation of SiLU (which is valid because you have already named the node at this level) which is `node.ReturnOp(xla_input * BuildSigmoid(xla_input), loctx)` and provides a "name" for the node with `GenericOp(OpKind(at::aten::silu), {input}, input.shape(), std::move(lower_fn))`.

My first approach was to repeat all so I duplicated the [sigmoid implemented](https://github.com/pytorch/xla/blob/3eaee46ef679cc6a0f1f694bd0a007dbfd09c51b/torch_xla/csrc/elementwise.cpp#L176-L181) in `elementwise.h/cpp`  and used that but the review of the PR suggested that I can call sigmoid because the node was already a `SiLU` so it doesn't matter if I reused what was already there at that moment. I corrected with an amended and just reused Sigmoid instead of my SiLU in elementwise, making the commit writes 2 less files than amended commit.

## The backward pass

This operation didn't include a backward pass, you should implement it if the header contains the forward and the backward pass, this was more a `_out` operation that is also used for in place methods.

Note: I haven't seen how this *dispatch* works, but I guess works like *simple inheritance*, when `aten_xla_type` don't provide the method then the ones from `aten_xla_type_default` are used (which is the CPU implementations and *fallbacks*). But see that the type `AtenXlaType` is not a `subclass` [aten_xla_type](https://github.com/pytorch/xla/blob/master/torch_xla/csrc/aten_xla_type.cpp#L26-L30)"

"Also note that `aten_xla_type_default` which is auto generated after build in some stage because it is not in repo and is ignored in `.gitignore`. So it should be other type of `dynamic dispatching` somewhere *deep in the code*.


# Conclusion

Lowering an op is difficult, but practice does help and easy tasks does too. You also need to provide a test case, which probably is just to take a template from a previous one (because test from CPU pytorch are used as base).

There are different things you need to know at less a little: pytorch, XLA, C++ (to see how the default operation is implemented), even some cuda if you can read that and take it as reference apart from the default CPU implementations and "cpu_fallback" (which I still don't know how they differ from CPU implementations or when they are used).

Hopefully this little explanation will help another person who wants to contribute lowering ops and understand a little better what is explained on the op lowering guide.