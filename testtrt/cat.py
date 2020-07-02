from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test

@tensorrt_converter('torch.cat')
def convert_cat(ctx):
    #print("DEBUG: inside convert_cat()")
    inputs = get_arg(ctx, 'input', pos=0, default=None) 
    #print("DEBUG: inputs[0].shape = ", inputs[0].shape)
    #print("DEBUG: inputs[1].shape = ", inputs[1].shape)
    dim = get_arg(ctx, 'dim', pos=1, default=0) 
    #print("Concat dimension = ", dim)

    output = ctx.method_return
    trt_inputs = [trt_(ctx.network, i) for i in inputs]

    layer = ctx.network.add_concatenation(inputs=trt_inputs)
    # layer.axis = dim -1
    layer.axis = dim
    output._trt = layer.get_output(0)
    print(output._trt)

class Cat(torch.nn.Module):
    def __init__(self, dim):
        #print("DEBUG: inside Cat()")
        super(Cat, self).__init__()
        self.dim = dim

    def forward(self, *x):
        #print("DEBUG: inside Cat() forward")
        return torch.cat(x, dim=self.dim)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 4), (1, 3, 4), (1, 17, 4)])
def test_Cat_basic():
    return Cat(1)
