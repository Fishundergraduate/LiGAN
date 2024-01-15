import sys
from turtle import forward

import torch_geometric
import numpy as np
from scipy import stats
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .interpolation import Interpolation
from torch_geometric.nn import GATConv, pool, GCNConv
from torch_geometric.data import Data, Batch
# mapping of unpool_types to Upsample modes
unpool_type_map = dict(
    n='nearest',
    t='trilinear',
)

#for debug
from torch.autograd import set_detect_anomaly
set_detect_anomaly(True)

def as_list(obj):
    return obj if isinstance(obj, list) else [obj]


def reduce_list(obj):
    return obj[0] if isinstance(obj, list) and len(obj) == 1 else obj


def is_positive_int(x):
    return isinstance(x, int) and x > 0


def get_n_params(model):
    total = 0
    for p in list(model.parameters()):
        n = 1
        for dim in p.shape:
            n *= dim
        total += n
    return total


def caffe_init_weights(module):
    '''
    Xavier initialization with fan-in variance
    norm mode, as implemented in caffe.
    '''
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        fan_in = nn.init._calculate_correct_fan(module.weight, 'fan_in')
        scale = np.sqrt(3 / fan_in)
        nn.init.uniform_(module.weight, -scale, scale)
        nn.init.constant_(module.bias, 0)


def compute_grad_norm(model):
    '''
    Compute the L2 norm of the gradient
    on model parameters.
    '''
    grad_norm2 = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        grad_norm2 += (p.grad.data**2).sum().item()
    return grad_norm2**(1/2)


def clip_grad_norm(model, max_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def sample_latent(
    batch_size,
    n_latent,
    means=None,
    log_stds=None,
    var_factor=1.0,
    post_factor=1.0,
    truncate=None,
    z_score=None,
    device='cuda',
):
    '''
    Draw batch_size latent vectors of size n_latent
    from a standard normal distribution (the prior)
    and reparameterize them using the posterior pa-
    rameters (means and log_stds), if provided.

    The standard deviation of the latent distribution
    is scaled by var_factor.

    If posterior parameters are provided, they are
    linearly interpolated with the prior parameters
    according to post_factor, where post_factor=1.0
    is purely posterior and 0.0 is purely prior.

    If truncate is provided, samples are instead drawn
    from a normal distribution truncated at that value.

    If z_score is provided, the magnitude of each
    vector is normalized and then scaled by z_score.
    '''
    assert batch_size is not None, batch_size

    # draw samples from standard normal distribution
    if not truncate:
        #print('Drawing latent samples from normal distribution')
        latents = torch.normal(means, log_stds.exp()).to(device)
    else:
        #print('Drawing latent samples from truncated normal distribution')
        assert NotImplementedError
        latents = torch.as_tensor(stats.truncnorm.rvs(
            a=-truncate,
            b=truncate,
            size=(batch_size, n_latent)
        ))

    if z_score not in {None, False}:
        # normalize and scale by z_score
        #  CAUTION: don't know how applicable this is in high-dims
        #print('Normalizing and scaling latent samples')
        latents = latents / latents.norm(dim=1, keepdim=True) * z_score

    #print(f'var_factor = {var_factor}, post_factor = {post_factor}')

    if log_stds is not None: # posterior stds
        stds = torch.exp(log_stds)

        # interpolate b/tw posterior and prior
        #   post_factor*stds + (1-post_factor)*1
        stds = post_factor*stds + (1-post_factor)

        # scale by standard deviation
        latents *= stds

    latents *= var_factor

    if means is not None:

        # interpolate b/tw posterior and prior
        #   post_factor*means + (1-post_factor)*0
        means = post_factor*means

        # shift by mean
        latents += means

    return latents


class Conv2dReLU(nn.Sequential):
    '''
    A 2D convolutional layer followed by leaky ReLU.

    Batch normalization can be applied either before
    (batch_norm=1) or after (batch_norm=2) the ReLU.

    Spectral normalization is applied by indicating
    the number of power iterations (spectral_norm).
    '''
    conv_type = GATConv

    def __init__(
        self,
        n_channels_in,
        n_channels_out,
        kernel_size=3,
        relu_leak=0.1,
        batch_norm=False,
        spectral_norm=False,
    ):
        self.modules = [
            self.conv_type(
                in_channels=n_channels_in,
                out_channels=n_channels_out,
            ),
            nn.LeakyReLU(
                negative_slope=relu_leak,
                inplace=True,
            )
        ]

        if batch_norm > 0: # value indicates order wrt conv and relu
            self.modules.insert(batch_norm, nn.BatchNorm2d(n_channels_out))

        if spectral_norm > 0: # value indicates num power iterations
            self.modules[0].lin_src = nn.utils.spectral_norm(
                self.modules[0].lin_src, n_power_iterations=spectral_norm
            )

        super().__init__(*(self.modules))
    
    def forward(self, data):
        x = self.modules[0](data.x,data.edge_index,data.edge_attr)
        x = self.modules[1](x)
        return Data(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr, y = data.y)

class TConv2dReLU(nn.Sequential):
    '''
    A 2d transposed convolution layer and leaky ReLU.

    Batch normalization can be applied either before
    (batch_norm=1) or after (batch_norm=2) the ReLU.

    Spectral normalization is applied by indicating
    the number of power iterations (spectral_norm).
    '''
    #TODO: transpose convolution
    conv_type = nn.Linear
    def __init__(
        self,
        n_channels_in,
        n_channels_out,
        kernel_size=3,
        relu_leak=0.1,
        batch_norm=False,
        spectral_norm=False,
    ):
        self.modules = [
            self.conv_type(
                in_features=n_channels_in,
                out_features=n_channels_out,
            ),
            nn.ReLU(
                inplace=True,
            )
        ]

        if batch_norm > 0: # value indicates order wrt conv and relu
            self.modules.insert(batch_norm, nn.BatchNorm2d(n_channels_out))

        if False:
            self.modules[0] = nn.utils.spectral_norm(
                self.modules[0], n_power_iterations=spectral_norm
            )

        super().__init__(*(self.modules))
    
    def forward(self, data):
        #import ipdb; ipdb.set_trace()
        x = self.modules[0](data.x).relu()
        data.x = x
        return data

class Conv2dBlock(nn.Module):
    '''
    A sequence of n_convs ConvReLUs with the same settings.
    '''
    conv_type = Conv2dReLU

    def __init__(
        self,
        n_convs,
        n_channels_in,
        n_channels_out,
        block_type='c',
        growth_rate=8,
        bottleneck_factor=0,
        debug=False,
        **kwargs
    ):
        super().__init__()

        assert block_type in {'c', 'r', 'd'}, block_type
        self.residual = (block_type == 'r')
        self.dense = (block_type == 'd')

        if self.residual:
            self.init_skip_conv(
                n_channels_in=n_channels_in,
                n_channels_out=n_channels_out,
                **kwargs
            )

        if self.dense:
            self.init_final_conv(
                n_channels_in=n_channels_in,
                n_convs=n_convs,
                growth_rate=growth_rate,
                n_channels_out=n_channels_out,
                **kwargs
            )
            n_channels_out = growth_rate

        self.init_conv_sequence(
            n_convs=n_convs,
            n_channels_in=n_channels_in,
            n_channels_out=n_channels_out,
            bottleneck_factor=bottleneck_factor, 
            **kwargs
        )

    def init_skip_conv(
        self, n_channels_in, n_channels_out, kernel_size, **kwargs
    ):
        if n_channels_out != n_channels_in:

            # 1x1x1 conv to map input to output channels
            self.skip_conv = self.conv_type(
                n_channels_in=n_channels_in,
                n_channels_out=n_channels_out,
            )
        else:
            self.skip_conv = nn.Identity()

    def init_final_conv(
        self,
        n_channels_in,
        n_convs,
        growth_rate,
        n_channels_out,
        kernel_size,
        **kwargs
    ):
        # 1x1x1 final "compression" convolution
        self.final_conv = self.conv_type(
            n_channels_in=n_channels_in + n_convs*growth_rate,
            n_channels_out=n_channels_out,
            kernel_size=1,
            **kwargs
        )

    def bottleneck_conv(
        self,
        n_channels_in,
        n_channels_bn,
        n_channels_out,
        kernel_size,
        **kwargs
    ):
        assert n_channels_bn > 0, \
            (n_channels_in, n_channels_bn, n_channels_out)

        return nn.Sequential(
            self.conv_type(
                n_channels_in=n_channels_in,
                n_channels_out=n_channels_bn,
                kernel_size=1,
                **kwargs,
            ),
            self.conv_type(
                n_channels_in=n_channels_bn,
                n_channels_out=n_channels_bn,
                kernel_size=kernel_size,
                **kwargs
            ),
            self.conv_type(
                n_channels_in=n_channels_bn,
                n_channels_out=n_channels_out,
                kernel_size=1,
                **kwargs,
            )
        )

    def init_conv_sequence(
        self,
        n_convs,
        n_channels_in,
        n_channels_out,
        bottleneck_factor,
        **kwargs
    ):
        self.conv_modules = []
        for i in range(n_convs):

            if bottleneck_factor: # bottleneck convolution
                conv = self.bottleneck_conv(
                    n_channels_in=n_channels_in,
                    n_channels_bn=n_channels_in//bottleneck_factor,
                    n_channels_out=n_channels_out,
                    **kwargs
                )
            else: # single convolution
                conv = self.conv_type(
                    n_channels_in=n_channels_in,
                    n_channels_out=n_channels_out,
                    **kwargs
                )
            self.conv_modules.append(conv)
            self.add_module(str(i), conv)

            if self.dense:
                n_channels_in += n_channels_out
            else:
                n_channels_in = n_channels_out

    def __len__(self):
        return len(self.conv_modules)

    def forward(self, inputs):

        if not self.conv_modules:
            return inputs

        if self.dense:
            all_inputs = [inputs]

        # convolution sequence
        for i, f in enumerate(self.conv_modules):
            
            if self.residual:
                identity = self.skip_conv(inputs) if i == 0 else inputs
                outputs = f(inputs)
                outputs.x = outputs.x + identity.x
            else:
                outputs = f(inputs)

            if self.dense:
                all_inputs.append(outputs)
                inputs = torch.cat(all_inputs, dim=1)
            else:
                inputs = outputs

        if self.dense:
            outputs = self.final_conv(inputs)

        return outputs


class TConv2dBlock(nn.Module):
    '''
    A sequence of n_convs TConvReLUs with the same settings.
    '''
    conv_type = TConv2dReLU

    def __init__(
        self,
        n_convs,
        n_channels_in,
        n_channels_out,
        block_type='c',
        growth_rate=8,
        bottleneck_factor=0,
        debug=False,
        **kwargs
    ):
        super().__init__()

        assert block_type in {'c', 'r', 'd'}, block_type
        self.residual = (block_type == 'r')
        self.dense = (block_type == 'd')

        if self.residual:
            self.init_skip_conv(
                n_channels_in=n_channels_in,
                n_channels_out=n_channels_out,
                **kwargs
            )

        if self.dense:
            self.init_final_conv(
                n_channels_in=n_channels_in,
                n_convs=n_convs,
                growth_rate=growth_rate,
                n_channels_out=n_channels_out,
                **kwargs
            )
            n_channels_out = growth_rate

        self.init_conv_sequence(
            n_convs=n_convs,
            n_channels_in=n_channels_in,
            n_channels_out=n_channels_out,
            bottleneck_factor=bottleneck_factor, 
            **kwargs
        )

    def init_skip_conv(
        self, n_channels_in, n_channels_out, kernel_size, **kwargs
    ):
        self.skip_conv = nn.Identity()

    def init_final_conv(
        self,
        n_channels_in,
        n_convs,
        growth_rate,
        n_channels_out,
        kernel_size,
        **kwargs
    ):
        # 1x1x1 final "compression" convolution
        self.final_conv = self.conv_type(
            n_channels_in=n_channels_in + n_convs*growth_rate,
            n_channels_out=n_channels_out,
            kernel_size=1,
            **kwargs
        )

    def bottleneck_conv(
        self,
        n_channels_in,
        n_channels_bn,
        n_channels_out,
        kernel_size,
        **kwargs
    ):
        assert n_channels_bn > 0, \
            (n_channels_in, n_channels_bn, n_channels_out)

        return nn.Sequential(
            self.conv_type(
                n_channels_in=n_channels_in,
                n_channels_out=n_channels_bn,
                kernel_size=1,
                **kwargs,
            ),
            self.conv_type(
                n_channels_in=n_channels_bn,
                n_channels_out=n_channels_bn,
                kernel_size=kernel_size,
                **kwargs
            ),
            self.conv_type(
                n_channels_in=n_channels_bn,
                n_channels_out=n_channels_out,
                kernel_size=1,
                **kwargs,
            )
        )

    def init_conv_sequence(
        self,
        n_convs,
        n_channels_in,
        n_channels_out,
        bottleneck_factor,
        **kwargs
    ):
        self.conv_modules = []
        for i in range(n_convs):

            if bottleneck_factor: # bottleneck convolution
                conv = self.bottleneck_conv(
                    n_channels_in=n_channels_in,
                    n_channels_bn=n_channels_in//bottleneck_factor,
                    n_channels_out=n_channels_out,
                    **kwargs
                )
            else: # single convolution
                conv = self.conv_type(
                    n_channels_in=n_channels_in,
                    n_channels_out=n_channels_out,
                    **kwargs
                )
            self.conv_modules.append(conv)
            self.add_module(str(i), conv)

            if self.dense:
                n_channels_in += n_channels_out
            else:
                n_channels_in = n_channels_out

    def __len__(self):
        return len(self.conv_modules)

    def forward(self, inputs):
        if torch.isnan(inputs.x).any():
            import ipdb; ipdb.set_trace()
            pass

        if not self.conv_modules:
            return inputs

        if self.dense:
            all_inputs = [inputs]

        # convolution sequence
        for i, f in enumerate(self.conv_modules):
            
            if self.residual:
                identity = self.skip_conv(inputs) if i == 0 else inputs
                outputs = f(inputs)
                outputs.x = outputs.x + identity.x
            else:
                outputs = f(inputs)
            if torch.isnan(outputs.x).any():
                import ipdb; ipdb.set_trace()
                pass

            if self.dense:
                all_inputs.append(outputs)
                inputs = torch.cat(all_inputs, dim=1)
            else:
                inputs = outputs

        if self.dense:
            outputs = self.final_conv(inputs)

        return outputs

class Pool2d(nn.Sequential):
    '''
    A layer that decreases 2d spatial dimensions,
    either by max pooling (pool_type=m), average
    pooling (pool_type=a), or strided convolution
    (pool_type=c).
    '''
    def __init__(self, n_channels, pool_type, pool_factor):

        if pool_type == 'm':
            pool = nn.MaxPool2d(
                kernel_size=pool_factor,
                stride=pool_factor,
            )

        elif pool_type == 'a':
            pool = nn.AvgPool2d(
                kernel_size=pool_factor,
                stride=pool_factor,
            )

        elif pool_type == 'c':
            pool = nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                groups=n_channels,
                kernel_size=pool_factor,
                stride=pool_factor,
            )

        else:
            raise ValueError('unknown pool_type ' + repr(pool_type))

        super().__init__(pool)


class Unpool2d(nn.Sequential):
    '''
    A layer that increases the 2d spatial dimensions,
    either by nearest neighbor (unpool_type=n), tri-
    linear interpolation (unpool_type=t), or strided
    transposed convolution (unpool_type=c).
    '''
    def __init__(self, n_channels, unpool_type, unpool_factor):
        """ import ipdb; ipdb.set_trace()
        if unpool_type in unpool_type_map:
            
            unpool = nn.Upsample(
                scale_factor=unpool_factor,
                mode=unpool_type_map[unpool_type],
            )

        elif unpool_type == 'c':
            
            unpool = nn.ConvTranspose2d(
                in_channels=n_channels,
                out_channels=n_channels,
                groups=n_channels,
                kernel_size=unpool_factor,
                stride=unpool_factor,
            )

        else:
            raise ValueError('unknown unpool_type ' + repr(unpool_type)) """
        super().__init__()
        self.unpool = nn.Identity()


    def forward(self, input: Data):
        #import ipdb; ipdb.set_trace()
        return input if isinstance(input, Data) else Data(x=input)

class Reshape(nn.Module):
    '''
    A layer that reshapes the input.
    '''
    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(shape)

    def __repr__(self):
        return 'Reshape(shape={})'.format(self.shape)

    def forward(self, x):
        return x.reshape(self.shape)


class Grid2Vec(nn.Sequential):
    '''
    A fully connected layer applied to a
    flattened version of the input, for
    transforming from grids to vectors.
    '''
    def __init__(
        self, in_shape, n_output, activ_fn=None, spectral_norm=0
    ):
        n_input = np.prod(in_shape)
        modules = [
            Reshape(shape=(-1, n_input)),
            nn.Linear(n_input, n_output)
        ]

        if activ_fn:
            modules.append(activ_fn)

        if spectral_norm > 0:
            modules[1] = nn.utils.spectral_norm(
                modules[1], n_power_iterations=spectral_norm
            )

        super().__init__(*modules)

class Graph2Vec(nn.Module):
    def __init__(self, in_shape, n_output, activ_fn=None, spectral_norm=0):
        super().__init__()
        n_input = int(np.prod(in_shape))
        self.gcn = GATConv(n_input, n_output)
        self.activ_fn = activ_fn 

    def forward(self, data:Data):
        x = self.gcn(data.x,data.edge_index,data.edge_attr)
        if self.activ_fn:
            x = self.activ_fn(x.to("cuda"), data.edge_index)
        return Data(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr, y = data.y)
class Vec2Graph(nn.Sequential):
    def __init__(self, n_input, out_shape, relu_leak, batch_norm, spectral_norm):
        super().__init__()
        n_output = int(np.prod(out_shape))
        self.fc1 = nn.Linear(n_input, n_output)
        self.relu = nn.LeakyReLU(negative_slope=relu_leak, inplace=True)
        
        if batch_norm > 0:
            self.batch_norm = nn.BatchNorm2d(out_shape[0])
        else:
            self.batch_norm = None
    
    def forward(self, data:Data):
        x = self.fc1(data.x)
        x = self.relu(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        data.x = x
        return data

class Vec2Grid(nn.Sequential):
    '''
    A fully connected layer followed by
    reshaping the output, for transforming
    from vectors to grids.
    '''
    def __init__(
        self, n_input, out_shape, relu_leak, batch_norm, spectral_norm
    ):
        n_output = np.prod(out_shape)
        modules = [
            nn.Linear(n_input, n_output),
            Reshape(shape=(-1, *out_shape)),
            nn.LeakyReLU(negative_slope=relu_leak, inplace=True),
        ]

        if batch_norm > 0:
            modules.insert(batch_norm+1, nn.BatchNorm2d(out_shape[0]))

        if spectral_norm > 0:
            modules[0] = nn.utils.spectral_norm(
                modules[0], n_power_iterations=spectral_norm
            )

        super().__init__(*modules)


class GridEncoder(nn.Module):
    '''
    A sequence of 2d convolution blocks and
    pooling layers, followed by one or more
    fully connected output tasks.
    '''
    # TODO reimplement the following:
    # - self-attention
    # - batch discrimination
    
    def __init__(
        self,
        n_channels,
        n_filters=32,
        width_factor=2,
        n_levels=4,
        conv_per_level=3,
        kernel_size=3,
        relu_leak=0.1,
        batch_norm=0,
        spectral_norm=0,
        pool_type='a',
        pool_factor=2,
        n_output=1,
        output_activ_fn=None,
        init_conv_pool=False,
        block_type='c',
        growth_rate=8,
        bottleneck_factor=0,
        debug=False,
    ):
        super().__init__()
        self.debug = debug

        # sequence of convs and/or pools
        self.grid_modules = []

        # track changing grid dimensions
        self.n_channels = n_channels
        """ if init_conv_pool:

            self.add_conv2d(
                name='init_conv',
                n_filters=n_filters,
                kernel_size=kernel_size+2,
                relu_leak=relu_leak,
                batch_norm=batch_norm,
                spectral_norm=spectral_norm,
            )
            self.add_pool2d(
                name='init_pool',
                pool_type=pool_type,
                pool_factor=pool_factor
            )
            n_filters *= width_factor """

        for i in range(n_levels):

            if i > 0: # downsample between conv blocks
                self.add_pool2d(
                    name='level'+str(i)+'_pool',
                    pool_type=pool_type,
                    pool_factor=pool_factor
                )
                n_filters *= width_factor
 
            self.add_conv2d_block(
                name='level'+str(i),
                n_convs=conv_per_level,
                n_filters=n_filters,
                kernel_size=kernel_size,
                relu_leak=relu_leak,
                batch_norm=batch_norm,
                spectral_norm=spectral_norm,
                block_type=block_type,
                growth_rate=growth_rate,
                bottleneck_factor=bottleneck_factor,
                debug=debug,
            )

        # fully-connected outputs
        n_output = as_list(n_output)
        assert n_output and all(n_o > 0 for n_o in n_output)

        output_activ_fn = as_list(output_activ_fn)
        if len(output_activ_fn) == 1:
            output_activ_fn *= len(n_output)
        assert len(output_activ_fn) == len(n_output)

        self.n_tasks = len(n_output)
        self.task_modules = []
        for i, (n_output_i, activ_fn_i) in enumerate(
            zip(n_output, output_activ_fn)
        ):
            self.add_grid2vec(
                name='fc'+str(i),
                n_output=n_output_i,
                activ_fn=activ_fn_i,
                spectral_norm=spectral_norm
            )

    def print(self, *args, **kwargs):
        if self.debug:
            print('DEBUG', *args, **kwargs, file=sys.stderr)

    def add_conv2d(self, name, n_filters, **kwargs):
        conv = Conv2dReLU(
            n_channels_in=self.n_channels,
            n_channels_out=n_filters,
            **kwargs
        )
        self.add_module(name, conv)
        self.grid_modules.append(conv)
        self.n_channels = n_filters
        self.print(name, self.n_channels)

    def add_pool2d(self, name, pool_factor, **kwargs):
        pool = Pool2d(
            n_channels=self.n_channels,
            pool_factor=pool_factor,
            **kwargs
        )
        self.add_module(name, pool)
        self.grid_modules.append(pool)
        self.print(name, self.n_channels)

    def add_conv2d_block(self, name, n_filters, **kwargs):
        conv_block = Conv2dBlock(
            n_channels_in=self.n_channels,
            n_channels_out=n_filters,
            **kwargs
        )
        self.add_module(name, conv_block)
        self.grid_modules.append(conv_block)
        self.n_channels = n_filters
        self.print(name, self.n_channels)

    def add_grid2vec(self, name, **kwargs):
        #fc = Grid2Vec(
        fc = Graph2Vec(
            in_shape=(self.n_channels,),
            **kwargs
        )
        self.add_module(name, fc)
        self.task_modules.append(fc)
        self.print(name, self.n_channels)

    def forward(self, inputs):
        # conv-pool sequence
        conv_features = []
        for f in self.grid_modules:
            
            if not isinstance(f, Pool2d):
                outputs = f(inputs)
            else:
                outputs = pool.avg_pool_neighbor_x(inputs) #TODO: add to max_pool_neighbor_x
            self.print(inputs.x.shape, '->', f, '->', outputs.x.shape)

            if not isinstance(f, Pool2d):
                conv_features.append(outputs)
            inputs = outputs
        # fully-connected outputs
        outputs = [f(inputs) for f in self.task_modules]
        #outputs_shape = [o.x.shape for o in outputs]
        #self.print(inputs.x.shape, '->', self.task_modules, '->', outputs_shape)

        return reduce_list(outputs), conv_features


class GridDecoder(nn.Module):
    '''
    A fully connected layer followed by a
    sequence of 2d transposed convolution
    blocks and unpooling layers.
    '''
    # TODO re-implement the following:
    # - self-attention
    # - gaussian output

    def __init__(
        self,
        n_input,
        n_channels,
        width_factor,
        n_levels,
        tconv_per_level,
        kernel_size,
        relu_leak,
        batch_norm,
        spectral_norm,
        unpool_type,
        unpool_factor,
        n_channels_out,
        final_unpool=False,
        skip_connect=False,
        block_type='c',
        growth_rate=8,
        bottleneck_factor=0,
        debug=False,
    ):
        super().__init__()
        self.skip_connect = bool(skip_connect)
        self.debug = debug

        # first fc layer maps to initial grid shape
        self.fc_modules = []
        self.n_input = n_input
        self.add_vec2grid(
            name='fc',
            n_input=n_input,
            n_channels=n_channels,
            relu_leak=relu_leak,
            batch_norm=batch_norm,
            spectral_norm=spectral_norm,
        )
        n_filters = n_channels

        self.grid_modules = []
        for i in reversed(range(n_levels)):

            if i + 1 < n_levels: # unpool between deconv blocks
                unpool_name = 'level' + str(i) + '_unpool'
                self.add_unpool2d(
                    name=unpool_name,
                    unpool_type=unpool_type,
                    unpool_factor=unpool_factor
                )
                n_filters //= width_factor

            if skip_connect:
                n_skip_channels = self.n_channels
                if i < n_levels - 1:
                    n_skip_channels //= width_factor
            else:
                n_skip_channels = 0

            tconv_block_name = 'level' + str(i)
            self.add_tconv2d_block(
                name=tconv_block_name,
                n_convs=tconv_per_level,
                n_filters=n_filters,
                kernel_size=kernel_size,
                relu_leak=relu_leak,
                batch_norm=batch_norm,
                spectral_norm=spectral_norm,
                block_type=block_type,
                growth_rate=growth_rate,
                bottleneck_factor=bottleneck_factor,
                n_skip_channels=n_skip_channels,
                debug=debug,
            )

        if final_unpool:

            self.add_unpool2d(
                name='final_unpool',
                unpool_type=unpool_type,
                unpool_factor=unpool_factor,
            )
            n_skip_channels //= width_factor

            self.add_tconv2d_block(
                name='final_conv',
                n_convs=tconv_per_level,
                n_filters=n_channels_out,
                kernel_size=kernel_size,
                relu_leak=relu_leak,
                batch_norm=batch_norm,
                spectral_norm=spectral_norm,
                block_type=block_type,
                growth_rate=growth_rate,
                bottleneck_factor=bottleneck_factor,
                n_skip_channels=n_skip_channels,
                debug=debug,
            )

        else: # final tconv maps to correct n_output channels

            self.add_tconv2d(
                name='final_conv',
                n_filters=n_channels_out,
                kernel_size=kernel_size,
                relu_leak=relu_leak,
                batch_norm=batch_norm,
                spectral_norm=spectral_norm,
            )

    def print(self, *args, **kwargs):
        if self.debug:
            print('DEBUG', *args, **kwargs, file=sys.stderr)

    def add_vec2grid(self, name, n_channels, **kwargs):
        vec2grid = Vec2Graph(
            out_shape=(n_channels,),
            **kwargs
        )
        self.add_module(name, vec2grid)
        self.fc_modules.append(vec2grid)
        self.n_channels = n_channels
        self.print(name, self.n_channels)

    def add_unpool2d(self, name, unpool_factor, **kwargs):
        unpool = Unpool2d(
            n_channels=self.n_channels,
            unpool_factor=unpool_factor,
            **kwargs
        )
        self.add_module(name, unpool)
        self.grid_modules.append(unpool)
        self.print(name, self.n_channels)

    def add_tconv2d(self, name, n_filters, **kwargs):
        tconv = TConv2dReLU(
            n_channels_in=self.n_channels,
            n_channels_out=n_filters,
            **kwargs
        )
        self.add_module(name, tconv)
        self.grid_modules.append(tconv)
        self.n_channels = n_filters
        self.print(name, self.n_channels)

    def add_tconv2d_block(
        self, name, n_filters, n_skip_channels, **kwargs
    ):
        tconv_block = TConv2dBlock(
            n_channels_in=self.n_channels + n_skip_channels,
            n_channels_out=n_filters,
            **kwargs
        )
        self.add_module(name, tconv_block)
        self.grid_modules.append(tconv_block)
        self.n_channels = n_filters
        self.print(name, self.n_channels)

    def forward(self, inputs:Tensor, skip_features=None):
        for f in self.fc_modules:
            outputs = f(inputs)
            self.print(inputs.x.shape, '->', f, '->', outputs.x.shape)
            if torch.isnan(outputs.x).any():
                import ipdb; ipdb.set_trace()
                pass
            inputs = outputs

        for f in self.grid_modules:

            if self.skip_connect and isinstance(f, TConv2dBlock):
                skip_inputs = skip_features.pop()
                batched_list = []
                for i in range(inputs.batch.max().int()+1):
                    skip_tens = torch.cat([skip_inputs[i].x, torch.zeros([inputs[i].x.shape[0] - skip_inputs[i].x.shape[0] % inputs[i].x.shape[0], skip_inputs[i].x.shape[1]], device=inputs[i].x.device)])
                    skip_tens = skip_tens.view(inputs[i].x.shape[0], -1, skip_tens.shape[1]).mean(dim=1)
                    input_tens = torch.cat([inputs[i].x,skip_tens], dim=1)
                    inputs_shape = [inputs[i].x.shape, skip_tens.shape]
                    batched_list.append(Data(x = input_tens, batch = inputs[i].batch, device=inputs[i].x.device, num_nodes = inputs[i].num_nodes))
                inputs = Batch.from_data_list(batched_list)
            else:
                    inputs_shape = inputs.x.shape

            outputs = f(inputs)
            self.print(inputs_shape, '->', f, '->', outputs.x.shape)
            if torch.isnan(outputs.x).any():
                import ipdb; ipdb.set_trace()
                pass
            inputs = outputs
        return outputs

class AdjDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1,2)
        self.fc2 = nn.Linear(2,4)
        self.fc3 = nn.Linear(4,2)
        self.fc4 = nn.Linear(2,1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, data:Batch):
        x = self.fc1(data.x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = torch.tanh(x)
        data.x = x
        return data

class GridGenerator(nn.Sequential):
    '''
    A generative model of 2d grids that can take the form
    of an encoder-decoder architecture (e.g. AE, VAE) or
    a decoder-only architecture (e.g. GAN). The model can
    also have a conditional encoder (e.g. CE, CVAE, CGAN).
    '''
    is_variational = False
    has_input_encoder = False
    has_conditional_encoder = False
    has_stage2 = False

    def __init__(
        self,
        n_channels_in=None,
        n_channels_cond=None,
        n_channels_out=19,
        n_filters=32,
        width_factor=2,
        n_levels=4,
        conv_per_level=3,
        kernel_size=3,
        relu_leak=0.1,
        batch_norm=0,
        spectral_norm=0,
        pool_type='a',
        unpool_type='n',
        pool_factor=2,
        n_latent=1024,
        init_conv_pool=False,
        skip_connect=False,
        block_type='c',
        growth_rate=8,
        bottleneck_factor=0,
        n_samples=0,
        device='cuda',
        debug=False,
    ):
        assert type(self) != GridGenerator, 'GridGenerator is abstract'
        self.debug = debug

        super().__init__()
        self.check_encoder_channels(n_channels_in, n_channels_cond)
        assert is_positive_int(n_channels_out)
        assert is_positive_int(n_latent)

        self.n_channels_in = n_channels_in
        self.n_channels_cond = n_channels_cond
        self.n_channels_out = n_channels_out
        self.n_latent = n_latent

        if self.has_input_encoder:

            if self.is_variational: # means and log_stds
                encoder_output = [n_latent, n_latent]
            else:
                encoder_output = n_latent

            self.input_encoder = GridEncoder(
                n_channels=n_channels_in,
                n_filters=n_filters,
                width_factor=width_factor,
                n_levels=n_levels,
                conv_per_level=conv_per_level,
                kernel_size=kernel_size,
                relu_leak=relu_leak,
                batch_norm=batch_norm,
                spectral_norm=spectral_norm,
                pool_type=pool_type,
                pool_factor=pool_factor,
                n_output=encoder_output,
                init_conv_pool=init_conv_pool,
                block_type=block_type,
                growth_rate=growth_rate,
                bottleneck_factor=bottleneck_factor,
                debug=debug,
            )

        if self.has_conditional_encoder:

            self.conditional_encoder = GridEncoder(
                n_channels=n_channels_cond,
                n_filters=n_filters,
                width_factor=width_factor,
                n_levels=n_levels,
                conv_per_level=conv_per_level,
                kernel_size=kernel_size,
                relu_leak=relu_leak,
                batch_norm=batch_norm,
                spectral_norm=spectral_norm,
                pool_type=pool_type,
                pool_factor=pool_factor,
                n_output=n_latent,
                init_conv_pool=init_conv_pool,
                block_type=block_type,
                growth_rate=growth_rate,
                bottleneck_factor=bottleneck_factor,
                debug=debug,
            )

        self.attr_decoder = AdjDecoder()

        n_pools = n_levels - 1 + init_conv_pool

        self.decoder = GridDecoder(
            n_input=self.n_decoder_input,
            n_channels=n_filters * width_factor**n_pools,
            width_factor=width_factor,
            n_levels=n_levels,
            tconv_per_level=conv_per_level,
            kernel_size=kernel_size,
            relu_leak=relu_leak,
            batch_norm=batch_norm,
            spectral_norm=spectral_norm,
            unpool_type=unpool_type,
            unpool_factor=pool_factor,
            n_channels_out=n_channels_out,
            final_unpool=init_conv_pool,
            skip_connect=skip_connect,
            block_type=block_type,
            growth_rate=growth_rate,
            bottleneck_factor=bottleneck_factor,
            debug=debug,
        )

        # latent interpolation state
        self.latent_interp = Interpolation(n_samples=n_samples)

        super().to(device)
        self.device = device

    def check_encoder_channels(self, n_channels_in, n_channels_cond):
        if self.has_input_encoder:
            assert is_positive_int(n_channels_in), n_channels_in
        else:
            assert n_channels_in is None, n_channels_in

        if self.has_conditional_encoder:
            assert is_positive_int(n_channels_cond), n_channels_cond
        else:
            assert n_channels_cond is None, n_channels_cond

    @property
    def n_decoder_input(self):
        n = 0
        if self.has_input_encoder or self.is_variational:
            n += self.n_latent
        if self.has_conditional_encoder:
            n += self.n_latent
        return n

    def sample_latent(
        self, batch_size, means=None, log_stds=None, interpolate=False, spherical=False, **kwargs
    ):
        latent_vecs = sample_latent(
            batch_size=batch_size,
            n_latent=self.n_latent,
            means=means,
            log_stds=log_stds,
            device=self.device,
            **kwargs
        )

        if interpolate:
            if not self.latent_interp.is_initialized:
                self.latent_interp.initialize(sample_latent(
                    batch_size=1,
                    n_latent=self.n_latent,
                    device=self.device,
                    **kwargs
                )[0])
            latent_vecs = self.latent_interp(latent_vecs, spherical=spherical)

        return latent_vecs

class CVAE(GridGenerator):
    is_variational = True
    has_input_encoder = True
    has_conditional_encoder = True

    def forward(
        self, inputs=None, conditions=None, batch_size=1, **kwargs
    ):
        if inputs.batch is not None:
            batch_size = inputs.batch.max().item() + 1
        if inputs is None: # prior
            assert NotImplementedError
            means, log_stds = None, None
            in_latents = self.sample_latent(batch_size, means.x, log_stds.x, **kwargs)
        else: # posterior
            (means, log_stds), _ = self.input_encoder(inputs)
            in_latents = torch.normal(means.x, log_stds.x.exp())
        cond_latents, cond_features = self.conditional_encoder(conditions)
        
        # concatenate input and conditional latents
        batch_cond_lat = cond_latents.x.mean(dim=0,keepdim=True).repeat(in_latents.shape[0], 1)        
        cat_latents = torch.cat([in_latents, batch_cond_lat], dim=1)
        batch_inds = torch.arange(batch_size, device=self.device)
        num_nodes_list = [inputs[i].num_nodes for i in range(batch_size)]
        max_num_nodes = max(num_nodes_list)
        starts = np.cumsum([0] + num_nodes_list[:-1])
        ends = np.cumsum(num_nodes_list)
        indices = list(zip(starts, ends))
        latent_vecs = [cat_latents[start: end] for start,end in indices]
        vecs_data_list = [Data(x=latent_vecs[i], batch=batch_inds[i], num_nodes = num_nodes_list[i], device = latent_vecs[i].device) for i in range(batch_size)]
        batched_vecs = Batch.from_data_list(vecs_data_list)
        
        in_latents_tens = in_latents @ in_latents.T
        #in_latents_stacked = [F.pad(in_latents_tens[start:end, start:end].unsqueeze(-1).repeat(1, 1, 3), (0, 0, 0, max_num_nodes - (end - start), 0, max_num_nodes - (end - start)), "constant", 0) for start, end in indices] # 3 for 3 channels in edge_attr
        in_latents_stacked = [F.pad(in_latents_tens[start:end, start:end].unsqueeze(-1), ( 0, 0, 0, max_num_nodes - (end - start), 0,max_num_nodes - (end - start)), "constant", 0) for start, end in indices] # 3 for 3 channels in edge_attr
        in_vecs_data_list = [Data(x=in_latents_stacked[i], batch=batch_inds[i], num_nodes = num_nodes_list[i], device = in_latents_stacked[i].device) for i in range(batch_size)]
        batched_in_vecs = Batch.from_data_list(in_vecs_data_list)
        #import ipdb; ipdb.set_trace()
        adj_matr = self.attr_decoder(batched_in_vecs)
        
        ## TODO: conditions should be batched
        num_nodes_list = [conditions[i].num_nodes for i in range(batch_size)]
        max_num_nodes = max(num_nodes_list)
        batch_cond = []
        for feat in cond_features:
            single_cond_feat = []
            for i in range(batch_size):
                __myx = F.pad(feat.x.mean(dim=0,keepdim=True).repeat(num_nodes_list[i], 1), (0, 0, 0, max_num_nodes - num_nodes_list[i]), "constant", 0)
                single_cond_feat.append(Data(x=__myx, batch=batch_inds[i], num_nodes = num_nodes_list[i], device = feat.x.device))
            batch_cond.append(Batch.from_data_list(single_cond_feat))
        outputs = self.decoder(
            inputs=batched_vecs, skip_features=batch_cond
        )

        return outputs, adj_matr, means, log_stds

""" class CVAE2(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv1 = GATConv(in_channel, 2*out_channel) """