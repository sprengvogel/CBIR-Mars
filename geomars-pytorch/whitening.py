
"""
This code has been developed by Roy, Subhankar and Siarohin, Aliaksandr and Sangineto, Enver and Bulo, Samuel Rota and Sebe,
Nicu and Ricci, Elisa or is heavily based upon their implementation and can be found here:
https://github.com/roysubhankar/dwt-domain-adaptation.
More details on the implementation are described in their paper: https://arxiv.org/abs/1903.03215
"""

import torch
import torch.nn as nn
from torch.nn.functional import conv2d, conv1d, softmax, log_softmax

class EntropyLoss(nn.Module):
		''' Module to compute entropy loss '''
		def __init__(self):
				super(EntropyLoss, self).__init__()

		def forward(self, x):
				p = softmax(x, dim=1)
				q = log_softmax(x, dim=1)
				b = p * q
				b = -1.0 * b.sum(-1).mean()
				return b

class _Whitening(nn.Module):

	def __init__(self, num_features, group_size, running_m=None, running_var=None, momentum=0.1, track_running_stats=True, eps=1e-3, alpha=1):
		super(_Whitening, self).__init__()
		self.num_features = num_features
		self.momentum = momentum
		self.track_running_stats = track_running_stats
		self.eps = eps
		self.alpha = alpha
		self.group_size = min(self.num_features, group_size)
		self.num_groups = self.num_features // self.group_size
		self.running_m = running_m
		self.running_var = running_var

		if self.track_running_stats and self.running_m is not None:
			self.register_buffer('running_mean', self.running_m)
			self.register_buffer('running_variance', self.running_var)
		else:
			self.register_buffer('running_mean', torch.zeros([1, self.num_features, 1, 1], out=torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()))
			self.register_buffer('running_variance', torch.ones([self.num_groups, self.group_size, self.group_size], out=torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()))

	def _check_input_dim(self, input):
		raise NotImplementedError

	def _check_group_size(self):
		raise NotImplementedError

	def forward(self, x):
		self._check_input_dim(x)
		self._check_group_size()

		m = x.mean(0).view(self.num_features, -1).mean(-1).view(1, -1, 1, 1)
		if not self.training and self.track_running_stats: # for inference
			m = self.running_mean
		xn = x - m

		T = xn.permute(1,0,2,3).contiguous().view(self.num_groups, self.group_size,-1)
		f_cov = torch.bmm(T, T.permute(0,2,1)) / T.shape[-1]
		f_cov_shrinked = (1-self.eps) * f_cov + self.eps * torch.eye(self.group_size, out=torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()).repeat(self.num_groups, 1, 1)

		if not self.training and self.track_running_stats: # for inference
			f_cov_shrinked = (1-self.eps) * self.running_variance + self.eps * torch.eye(self.group_size, out=torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()).repeat(self.num_groups, 1, 1)

		inv_sqrt = torch.inverse(torch.cholesky(f_cov_shrinked)).contiguous().view(self.num_features, self.group_size, 1, 1)

		decorrelated = conv2d(xn, inv_sqrt, groups = self.num_groups)

		if self.training and self.track_running_stats:
			self.running_mean = torch.add(self.momentum * m.detach(), (1 - self.momentum) * self.running_mean, out=self.running_mean)
			self.running_variance = torch.add(self.momentum * f_cov.detach(), (1 - self.momentum) * self.running_variance, out=self.running_variance)

		return decorrelated

class WTransform2d(_Whitening):
	def _check_input_dim(self, input):
		if input.dim() != 4:
			raise ValueError('expected 4D input (got {}D input)'. format(input.dim()))

	def _check_group_size(self):
		if self.num_features % self.group_size != 0:
			raise ValueError('expected number of channels divisible by group_size (got {} group_size\
				for {} number of features'.format(self.group_size, self.num_features))

class WTransform1D(_Whitening):

	def __init__(self, num_features, group_size, running_m=None, running_var=None, momentum=0.1, track_running_stats=True, eps=1e-3, alpha=1):
		super(_Whitening, self).__init__()
		self.num_features = num_features
		self.momentum = momentum
		self.track_running_stats = track_running_stats
		self.eps = eps
		self.alpha = alpha
		self.group_size = min(self.num_features, group_size)
		self.num_groups = self.num_features // self.group_size
		self.running_m = running_m
		self.running_var = running_var

		if self.track_running_stats and self.running_m is not None:
			self.register_buffer('running_mean', self.running_m)
			self.register_buffer('running_variance', self.running_var)
		else:
			self.register_buffer('running_mean', torch.zeros([1, self.num_features], out=torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()))
			self.register_buffer('running_variance', torch.ones([self.num_groups, self.group_size, self.group_size], out=torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()))

	def _check_input_dim(self, input):
		if input.dim() != 2:
			raise ValueError('expected 2D input (got {}D input)'. format(input.dim()))

	def _check_group_size(self):
		if self.num_features % self.group_size != 0:
			raise ValueError('expected number of channels divisible by group_size (got {} group_size\
				for {} number of features'.format(self.group_size, self.num_features))

	def forward(self, x):
		if x.dim() == 1:
			x = x.view(1, -1)
		self._check_input_dim(x)
		self._check_group_size()

		m = x.mean(0).view(self.num_features, -1).mean(-1).view(1, -1)
		if not self.training and self.track_running_stats: # for inference
			m = self.running_mean
		xn = x - m


		T = xn.permute(1,0).contiguous().view(self.num_groups, self.group_size,-1)

		f_cov = torch.bmm(T, T.permute(0,2,1)) / T.shape[-1]
		f_cov_shrinked = (1-self.eps) * f_cov + self.eps * torch.eye(self.group_size, out=torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()).repeat(self.num_groups, 1, 1)

		if not self.training and self.track_running_stats: # for inference
			f_cov_shrinked = (1-self.eps) * self.running_variance + self.eps * torch.eye(self.group_size, out=torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()).repeat(self.num_groups, 1, 1)

		inv_sqrt = torch.inverse(torch.cholesky(f_cov_shrinked)).contiguous().view(self.num_features, self.group_size, 1)

		decorrelated = conv1d(xn.view(-1, self.num_features, 1), inv_sqrt, groups = self.num_groups)
		decorrelated = decorrelated.view(-1, self.num_features)

		if self.training and self.track_running_stats:
			self.running_mean = torch.add(self.momentum * m.detach(), (1 - self.momentum) * self.running_mean, out=self.running_mean)
			self.running_variance = torch.add(self.momentum * f_cov.detach(), (1 - self.momentum) * self.running_variance, out=self.running_variance)

		return decorrelated
