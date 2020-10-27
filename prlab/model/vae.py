import torch
import torch.nn as nn
from torch.nn import functional as F

from prlab.common.utils import lazy_object_fn_call


class DNNEncoder(nn.Module):
    """
    Implement simple DNN Encoder for VAE
    Input: [bs, input_dim]
    Output: z_mu, z_var which [bs, z_dim], [bs, latent_dim]
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, is_relu=True, **kwargs):
        """
        :param input_dim: number, the size of input
        :param hidden_dim: number or list of number, the size of hidden(s)
        :param latent_dim: dim of representation space
        :param kwargs:
        """
        super(DNNEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim]
        self.latent_dim = latent_dim
        self.is_relu = is_relu

        self.seqs = [nn.Linear(self.input_dim, self.hidden_dim[0])]
        for pos in range(1, len(self.hidden_dim)):
            current_hidden_size = self.hidden_dim[pos]
            previous_hidden_size = self.hidden_dim[pos - 1]
            current_hidden = nn.Linear(previous_hidden_size, current_hidden_size)
            self.seqs.append(current_hidden)
            if self.is_relu:
                self.seqs.append(nn.ReLU(inplace=False))

        self.hiddens = nn.Sequential(*self.seqs)

        # make the latent mu and var
        self.mu = nn.Linear(self.hidden_dim[-1], self.latent_dim)
        self.var = nn.Linear(self.hidden_dim[-1], self.latent_dim)

    def forward(self, x, **kwargs):
        hidden_val = self.hiddens(x)
        z_mu = self.mu(hidden_val)
        z_var = self.var(hidden_val)
        return z_mu, z_var, x


class ExtendedEncoder(nn.Module):
    """
    Extend the basic encoder, e.g. fastai.tabular.models.TabularModel
    to work as Encoder for VAE.
    Assume that base network return the output is one-dimension, if not then the Flatten
    step is applied.
    """

    def __init__(self, base_net, out_dim, latent_dim, **kwargs):
        """
        :param base_net: the base network instance, made outside of this class
        :param out_dim: dimension of the output of base_net (not the final output of VAE)
        :param latent_dim: dimension of the latent space
        :param kwargs: other params, may or may not used
        """
        super(ExtendedEncoder, self).__init__()

        self.base_net = base_net
        self.out_dim = out_dim
        self.latent_dim = latent_dim

        # convert base_net to object if not yet, type (list, tuple, dict, str)
        self.base_net = lazy_object_fn_call(self.base_net, **kwargs)

        # make the latent mu and var
        self.mu = nn.Linear(self.out_dim, self.latent_dim)
        self.var = nn.Linear(self.out_dim, self.latent_dim)

    def forward(self, *x, **kwargs):
        base_out, *other = self.base_net(*x)
        base_out_flat = base_out.view(base_out.size(0), -1)

        base_out_flat = F.relu(base_out_flat)

        z_mu = self.mu(base_out_flat)
        z_var = self.var(base_out_flat)
        x_ret = x if len(other) == 0 else other[-1]
        return z_mu, z_var, x_ret


class DNNDecoder(nn.Module):
    """
    Implement simple DNN Decoder for VAE
    Input: [bs, latent_dim]
    Output: [bs, output_dim]
    """

    def __init__(self, latent_dim, hidden_dim, output_dim, is_relu=True, **kwargs):
        """
        :param latent_dim: integer number indicating the latent size
        :param hidden_dim: number or list of number, the size of hidden(s)
        :param output_dim: dim of output space
        :param kwargs:
        """
        super(DNNDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim]
        self.output_dim = output_dim
        self.is_relu = is_relu

        self.seqs = [nn.Linear(self.latent_dim, self.hidden_dim[0])]
        if self.is_relu:
            self.seqs.append(nn.ReLU(inplace=False))
        for pos in range(1, len(self.hidden_dim)):
            current_hidden_size = self.hidden_dim[pos]
            previous_hidden_size = self.hidden_dim[pos - 1]
            current_hidden = nn.Linear(previous_hidden_size, current_hidden_size)
            self.seqs.append(current_hidden)
            if self.is_relu:
                self.seqs.append(nn.ReLU(inplace=False))

        self.hiddens = nn.Sequential(*self.seqs)

        self.output_layer = nn.Linear(self.hidden_dim[-1], self.output_dim)
        self.last = nn.Sigmoid() if kwargs.get('is_sigmoid_last', True) else None

    def forward(self, x, **kwargs):
        hidden_val = self.hiddens(x)
        output = self.output_layer(hidden_val)
        output = self.last(output) if self.last is not None else output
        return output


class GeneralVAE(nn.Module):
    """
    Implement the General VAE (vanillavae)
    Ref: https://graviraja.github.io/vanillavae
    In most case, loss object is from `prlab.losses.VAESimpleLoss`
    """
    TRAIN_MODE = 1
    TEST_MODE = 2

    def __init__(self, encoder, decoder, output_mode=None, **kwargs):
        super(GeneralVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_mode = output_mode

        # convert encoder/decoder to object if not yet (lazy)
        self.encoder = lazy_object_fn_call(self.encoder, **kwargs)
        self.decoder = lazy_object_fn_call(self.decoder, **kwargs)

    def layer_groups(self):
        return [self.encoder, self.decoder]

    def forward(self, *x, **kwargs):
        z_mu, z_var, *other = self.encoder(*x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        # decode
        predicted = self.decoder(x_sample)

        x_ret = x if len(other) == 0 else other[-1]
        if self.output_mode is None:
            return predicted, z_mu, z_var, x_ret
        elif self.output_mode == self.TRAIN_MODE:
            return predicted, z_mu, z_var, x_ret
        elif self.output_mode == self.TEST_MODE:
            return predicted
        else:
            raise Exception("Wrong output_mode, please have a check, it should be None/Train(1)/Test(2)")


class MultiDecoderVAE(GeneralVAE):
    """
    Multi Decoder VAE, support more than 1 decoder
    """

    def __init__(self, encoder, decoder, output_mode=None,
                 second_decoder=[],
                 is_sep_sample=True,
                 is_testing=False,
                 **kwargs):
        super(MultiDecoderVAE, self).__init__(encoder=encoder, decoder=decoder,
                                              output_mode=output_mode,
                                              **kwargs)
        self.second_decoder = second_decoder if isinstance(second_decoder, list) else [second_decoder]
        # second_decoder support lazy calc to make object if not yet
        self.second_decoder = [lazy_object_fn_call(o, **kwargs) for o in self.second_decoder]
        self.second_decoder_ = nn.Sequential(*self.second_decoder)  # for store purpose

        self.is_sep_sample = is_sep_sample
        self.is_testing = is_testing

    def layer_groups(self):
        return [self.encoder, nn.Sequential(self.decoder, *self.second_decoder)]

    def forward(self, *x, **kwargs):
        z_mu, z_var, *other = self.encoder(*x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)

        def f_sample():
            eps = torch.randn_like(std)
            x_sample_i = eps.mul(std).add_(z_mu)
            return x_sample_i

        # decode
        x_sample = f_sample()
        predicted = self.decoder(x_sample)
        others = []
        for dec in self.second_decoder:
            x_sample = f_sample() if self.is_sep_sample else x_sample
            others.append(dec(x_sample))

        x_ret = x if len(other) == 0 else other[-1]
        if self.is_testing and len(others) > 0:
            # TODO when more than one others? Fastai does not support tuple or list when TTA, preds
            return others[0]

        if self.output_mode is None:
            return predicted, z_mu, z_var, others, x_ret
        elif self.output_mode == self.TRAIN_MODE:
            return predicted, z_mu, z_var, others, x_ret
        elif self.output_mode == self.TEST_MODE:
            return predicted, others
        else:
            raise Exception("Wrong output_mode, please have a check, it should be None/Train(1)/Test(2)")

    def sample(self, latent_dim):
        """
        Sample from random of latent
        :return:
        """
        device = next(self.parameters()).device
        x_sample = torch.randn(1, latent_dim).to(device)
        predicted = self.decoder(x_sample)
        return predicted
