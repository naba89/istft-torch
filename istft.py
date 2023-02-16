import torch
import torch.nn.functional as F


def istft(input_tensor, n_fft, hop_length=None, win_length=None, window=None,
          normalized=False, center=True, length=None, onesided=None, return_complex=False, skip_nola=True):
    r"""    Quick and naive translation of torch.istft() to python with option to skip NOLA check.
            I don't handle the return complex case for now
            and the code might not be optimized and handle all corner cases.
            https://github.com/pytorch/pytorch/blob/e790281a85fe3693fc1d38bf0e2c6e874d5e10b0/aten/src/ATen/native/SpectralOps.cpp#L1002-L1175

    """

    if return_complex:
        raise NotImplementedError("return_complex=True is not implemented")

    if hop_length is None:
        hop_length = int(n_fft // 4)
    if win_length is None:
        win_length = n_fft

    input_tensor = torch.view_as_real(input_tensor.resolve_conj())

    input_dim = input_tensor.dim()
    n_frames = input_tensor.size(-2)
    fft_size = input_tensor.size(-3)

    expected_output_signal_len = n_fft + hop_length * (n_frames - 1)

    onesided = onesided if onesided else fft_size != n_fft

    if window is None:
        window = torch.ones(n_fft, dtype=input_tensor.dtype, device=input_tensor.device)

    if win_length != n_fft:
        # center window by padding zeros on right and left side
        left = (n_fft - win_length) / 2
        window = F.pad(window, (left, n_fft - win_length - left), mode='constant', value=0)

    if input_dim == 3:
        input_tensor = input_tensor.unsqueeze(0)

    input_tensor = torch.view_as_complex(input_tensor.transpose(1, 2))
    if not onesided:
        input_tensor = input_tensor[..., :fft_size // 2 + 1]

    norm_mode = 'ortho' if normalized else 'backward'

    input_tensor = torch.fft.irfft(input_tensor, dim=input_tensor.dim() - 1, norm=norm_mode)

    assert input_tensor.size(2) == n_fft

    y_tmp = input_tensor * window.view(1, 1, n_fft)

    fold_params = {
        'kernel_size': (n_fft, 1),
        'stride': (hop_length, 1),
    }

    y_tmp = y_tmp.transpose(1, 2)
    y = F.fold(y_tmp, output_size=(expected_output_signal_len, 1), **fold_params)
    y = y.reshape(y.size(0), -1)
    window = window.pow(2).expand(1, n_frames, n_fft)

    window = window.transpose(1, 2)

    window_envelop = F.fold(window, output_size=(expected_output_signal_len, 1), **fold_params)
    window_envelop = window_envelop.reshape(window_envelop.size(0), -1)
    assert window_envelop.size(1) == expected_output_signal_len
    assert y.size(1) == expected_output_signal_len

    start = n_fft // 2 if center else 0

    end = expected_output_signal_len
    if length is not None:
        end = start + length
    elif center:
        end = -(n_fft // 2)

    y = y[..., start:end]
    window_envelop = window_envelop[..., start:end]

    if not skip_nola:
        window_envelop_lowest = window_envelop.abs().min().lt(1e-11)
        window_envelop_lowest = torch.equal(window_envelop_lowest,
                                            window_envelop_lowest.new_ones(window_envelop_lowest.shape,
                                                                           device=window_envelop_lowest.device))
        if window_envelop_lowest:
            raise ValueError("window overlap add min: {}".format(window_envelop_lowest))
        
    y = y / window_envelop
    if input_dim == 3:
        y = y.squeeze(0)

    if end > expected_output_signal_len:
        y = F.pad(y, (0, end - expected_output_signal_len), mode='constant', value=0)

    return y
