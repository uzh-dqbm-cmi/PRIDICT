## inspired by the Attributionprior repository <link>
## developed by orizuru (= orisenbazuru 😬)
import torch
from torch.autograd import grad
from torch.utils.data import DataLoader


class IntegratedGradExplainer():
    def __init__(self, background_dataset, device, k=50, scale_by_inputs=True, fdtype=torch.float32):
        self.k = k
        self.scale_by_inputs = scale_by_inputs
        # orizuru: i think the assumption here is that batch_size of input should be equivalent 
        # to the batch_size of the returned reference batch
        self.bg_set = background_dataset
        self.bg_sampler = None
        self.device = device
        self.fdtype = fdtype
        self.random_alpha = False

    def create_bgset_sampler(self, batch_size):
        if batch_size > len(self.bg_set):
            raise f"batch_size is > the number of samples (n={len(self.bg_set)}) in the bg_set!!"

        if self.bg_sampler is None:
            self.bg_sampler = DataLoader(dataset=self.bg_set, 
                                        batch_size=batch_size, 
                                        shuffle=True, 
                                        drop_last=True)
        elif self.bg_sampler.batch_size != batch_size:
            self.bg_sampler = DataLoader(dataset=self.bg_set, 
                                        batch_size=batch_size, 
                                        shuffle=True, 
                                        drop_last=True)

    def get_bgset_batch(self):
        return next(iter(self.bg_sampler))

    def expand_baseline(self, baseline_input, n):
        # baseline_input [bsize, ...] where ... is the rest of dimensions
        num_input_dims = len(list(baseline_input.shape)[1:])
        keep_fixed = [-1] * num_input_dims
        return baseline_input.unsqueeze(1).expand(-1, n, *keep_fixed)

    def get_samples_input(self, input_tensor, baseline_tensor):
        '''
        calculate interpolation points
        Args:
            input_tensor: Tensor of shape (batch, ...), where ... indicates
                          the input dimensions. 
            baseline_tensor: A tensor of shape (batch, k, ...) where ... 
                indicates dimensions, and k represents the number of background 
                reference samples to draw per input in the batch.
        Returns: 
            samples_input: A tensor of shape (batch, k, ...) with the 
                interpolated points between input and ref.
        '''
        device = self.device
        fdtype = self.fdtype

        input_dims = list(input_tensor.shape[1:])
        num_input_dims = len(input_dims)
            
        batch_size = baseline_tensor.shape[0]
        k_ = baseline_tensor.shape[1]

        # Grab a [batch_size, k]-sized interpolation sample
        if self.random_alpha:
            t_tensor = torch.tensor(batch_size, k_).uniform_(0,1).type(fdtype).to(device)
        else: # get linearly spaced points from 0 - 1
            t_tensor = torch.cat([torch.linspace(start=0.,end=1.,steps=k_) for __ in range(batch_size)]).to(device)

        # print('t_tensor.shape:', t_tensor.shape)
        # print('t_tensor:\n', t_tensor)

        shape = [batch_size, k_] + [1] * num_input_dims
        # interp_coef represent alphas
        # [batch_size, k, ...]
        interp_coef = t_tensor.view(*shape)
        # print('interp_coef.shape:', interp_coef.shape)
        # print('interp_coef:\n', interp_coef)

        # Evaluate the end points
        end_point_ref = (1.0 - interp_coef) * baseline_tensor

        # print('end_point_ref.shape:',end_point_ref.shape)
        # print('end_point_ref:\n',end_point_ref)

        input_expand_mult = input_tensor.unsqueeze(1)
        # print('input_expand_mult.shape:',input_expand_mult.shape)
        # print('input_expand_mult:\n',input_expand_mult)


        end_point_input = interp_coef * input_expand_mult

        # print('end_point_input.shape:',end_point_input.shape)
        # print('end_point_input:\n',end_point_input)

        ## orizuru:
        # alpha*input_tensor + (1-alpha)*ref_tensor
        # => ref_tensor + alpha * (input_tensor - ref_tensor)
        # => x' + alpha * (x - x')   numerator in IG and EG formula
        # => x' is baseline tensor (background sample chosen as baseline)
        # => x is the input tensor that we are trying to explain corresponding output prediction
        samples_input = end_point_input + end_point_ref
        # print('samples_input.shape:',samples_input.shape)
        # print('samples_input:\n',samples_input)

        
        return samples_input
    
    def get_samples_delta(self, input_tensor, baseline_tensor):
        # orizuru: this is the distance between the input tensor and the baseline samples 
        # input_expand_mult.shape = (bsize, 1, ...)
        input_expand_mult = input_tensor.unsqueeze(1)
        # x - x'
        # reference_tensor.shape = (bsize, k, ...)
        sd = input_expand_mult - baseline_tensor
        # print('sd.shape:',sd.shape)
        # print('sd:\n',sd)
        return sd
    
    def sum_ig_contrib(self, ig_tensor, seqlen):
        """
        Args:
            ig_tensor: torch tensor, [num_baselines, max_seqlen, featdim]
            seqlen: list [num_baselines], length of the sequences 
        Returns:
            tensor [num_baselines]
        
        """
        # (bsize, max_seqlen)
        collapsed_t = torch.zeros(ig_tensor.shape[0]).type(self.fdtype).to(self.device)
        t = ig_tensor.sum(axis=-1)
        for b, slen in enumerate(seqlen):
            collapsed_t[b] =  t[b, :slen].sum(axis=-1).item()
        return collapsed_t

    def compute_riemann_trapezoidal_approximation(self, gradients):
        """
        Args:
            gradients: tensor, (num_baselines, m_steps, maxseqlen, embed_dim)
        Returns:
            integrated gradient: tensor (num_baselines, maxseqlen, embed_dim)
        """
        # sum across the m_steps dimension
        grads = (gradients[:,:-1] + gradients[:,1:]) / 2.
        # # average across m_steps dimension 
        # # TODO: test summing across m_steps axis and then dividing by m_steps
        integrated_gradients = torch.mean(grads, axis=1)

        # integrated_gradients = torch.mean(gradients, axis=1)
        return integrated_gradients