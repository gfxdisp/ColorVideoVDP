from pycvvdp import cvvdp
import torch

"""
ColourVideoVDP metric. Refer to pytorch_examples for examples on how to use this class. 
"""
class cvvdp_rnn(cvvdp):
    input_dims = 36     # 9 bands x 4 bands per channel
    rho_dims = 1        # Condition on base rho band

    def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, color_space="sRGB", foveated=False, heatmap=None, quiet=False, device=None, temp_padding="replicate", use_checkpoints=False,
                 hidden_dims=8, num_layers=2, dropout=0.2, pooling='gru', pretrained_net=None):
        super().__init__(display_name, display_photometry, display_geometry, color_space, foveated, heatmap, quiet, device, temp_padding, use_checkpoints)
        assert pooling in ('base', 'lstm', 'gru')
        self.pooling = pooling
        rnn_type = {'lstm': torch.nn.LSTM, 'gru': torch.nn.GRU}
        if pooling in ('lstm', 'gru'):
            recurrent_net = rnn_type[pooling](self.input_dims, hidden_dims, num_layers, dropout=dropout)
            linear = torch.nn.Sequential(
                torch.nn.Linear(hidden_dims + self.rho_dims, 1),   # rho_band appended here
                torch.nn.Sigmoid()
            )
            self.pooling_net = torch.nn.Sequential(recurrent_net, linear)

            # Load network weights
            if pretrained_net is not None:
                checkpoint = torch.load(pretrained_net)
                self.pooling_net.load_state_dict(checkpoint['state_dict'])

            self.pooling_net.to(self.device)
            self.pooling_net.eval()

    '''
    The same as `predict` but takes as input fvvdp_video_source_* object instead of Numpy/Pytorch arrays.
    '''
    def predict_video_source(self, vid_source, features_provided=False):
        if not features_provided:
            return super().predict_video_source(vid_source)
        else:
            Q_jod = self.do_pooling_and_jods(*vid_source)

            stats = {}
            return (Q_jod.squeeze(), stats)

    # Perform pooling with per-band weights and map to JODs
    def do_pooling_and_jods(self, Q_per_ch, base_rho_band):
        if self.pooling == 'base':
            return super().do_pooling_and_jods(Q_per_ch, base_rho_band)
        # Q_per_ch[channel,frame,sp_band]
        feat_in = Q_per_ch.permute(1, 0, 2).flatten(start_dim=1)
        feat_intermediate, _ = self.pooling_net[0](feat_in)
        feat_intermediate = torch.cat((feat_intermediate[-1], base_rho_band.unsqueeze(0)))
        Q = self.pooling_net[1](feat_intermediate).squeeze() * 10
        return Q

    def short_name(self):
        return "cvvdp_rnn"

    def quality_unit(self):
        return "JOD"

    def get_info_string(self):
        if self.display_name.startswith('standard_'):
            #append this if are using one of the standard displays
            standard_str = ', (' + self.display_name + ')'
        else:
            standard_str = ''
        fv_mode = 'foveated' if self.foveated else 'non-foveated'
        return '"ColourVideoVDP with {} v{}, {:.4g} [pix/deg], Lpeak={:.5g}, Lblack={:.4g} [cd/m^2], {}{}"'.format(self.pooling, self.version, self.pix_per_deg, self.display_photometry.get_peak_luminance(), self.display_photometry.get_black_level(), fv_mode, standard_str)
