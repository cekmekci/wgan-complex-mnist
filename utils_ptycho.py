import numpy as np
# import cupy
import tike
import tike.ptycho
import torch

def ptycho_forward_op(input, scan, probe):
    # input is a tensor with shape (1,2,H1,W1)
    # scan is a np array with shape (S,2)
    # probe is a tensor with shape (1,2,H2,W2)
    # farplane is a tensor with (1,S,2,H2,W2)
    device = input.device
    # convert input into a complex array
    input = np.squeeze(input.detach().cpu().numpy(), 0)
    input = input[0,:,:] + 1j * input[1,:,:]
    # convert probe into a complex array
    probe = np.squeeze(probe.detach().cpu().numpy(), 0)
    probe = probe[0,:,:] + 1j * probe[1,:,:]
    probe = np.expand_dims(probe, (0,1,2)) # expand probe dims to make it compatible with tike (1,1,1,H2,W2)
    # forward operator
    with tike.operators.Ptycho(probe_shape=probe.shape[-1], detector_shape=int(probe.shape[-1]), nz=input.shape[-2], n=input.shape[-1]) as operator:
        scan = operator.asarray(scan, dtype=tike.precision.floating)
        psi = operator.asarray(input, dtype=tike.precision.cfloating)
        probe = operator.asarray(probe, dtype=tike.precision.cfloating)
        # farplane is (scan.shape[0], 1, 1, probe.shape[0], probe.shape[1])
        farplane = operator.fwd(probe=tike.ptycho.probe.get_varying_probe(probe, None, None), scan=scan, psi=psi)
        # get rid of the squeezable dims. farplane is (scan.shape[0], probe.shape[0], probe.shape[1]) now
        farplane = np.squeeze(farplane, (1,2))
        # convert it back to a normal numpy array
        farplane = operator.asnumpy(farplane)
    # convert the result into a tensor with shape (1,S,2,H2,W2)
    farplane = np.expand_dims(farplane, 0)
    farplane = np.stack((np.real(farplane), np.imag(farplane)), 2)
    farplane = torch.from_numpy(farplane).float().to(device)
    return farplane


def ptycho_adjoint_op(input, scan, probe, object_size):
    # input is a tensor with shape (1,S,2,H2,W2)
    # scan is a numpy array with shape (S, 2)
    # probe is a tensor with shape (1,2,H2,W2)
    # object size is a tuple: (H1,W1)
    device = input.device
    # convert input into a complex array (S,H2,W2)
    input = np.squeeze(input.detach().cpu().numpy(), 0)
    input = input[:,0,:,:] + 1j * input[:,1,:,:]
    # convert probe into a complex array
    probe = np.squeeze(probe.detach().cpu().numpy(), 0)
    probe = probe[0,:,:] + 1j * probe[1,:,:]
    probe = np.expand_dims(probe, (0,1,2)) # expand probe dims to make it compatible with tike (1,1,1,H2,W2)
    with tike.operators.Ptycho(probe_shape=probe.shape[-1], detector_shape=int(probe.shape[-1]), nz=input.shape[-2], n=input.shape[-1]) as operator:
        farplane = operator.asarray(np.expand_dims(input, (1,2)), dtype=tike.precision.cfloating) #cfloating??
        scan = operator.asarray(scan, dtype=tike.precision.floating)
        probe = operator.asarray(probe, dtype=tike.precision.cfloating)
        psi = operator.asarray(np.zeros(object_size), dtype=tike.precision.cfloating)
        output = operator.adj(farplane=farplane, probe=probe, scan=scan, psi=psi)
        output = operator.asnumpy(output)
    # convert the result into a tensor with shape (1,2,H1,W1)
    output = np.expand_dims(output, 0)
    output = np.stack((np.real(output), np.imag(output)), 1)
    output = torch.from_numpy(output).float().to(device)
    return output


def cartesian_scan_pattern(object_size, probe_shape, step_size = 25, sigma = 1):
    scan = []
    for y in range(0, object_size[0] - probe_shape[2] + 1, step_size):
        for x in range(0, object_size[1] - probe_shape[3] + 1, step_size):
            y_perturbation = sigma * np.random.randn()
            x_perturbation = sigma * np.random.randn()
            y_new = 1 + y + y_perturbation
            x_new = 1 + x + x_perturbation
            if x_new <= 1:
                x_new = 1 + x + np.abs(x_perturbation)
            if y_new <= 1:
                y_new = 1 + y + np.abs(y_perturbation)
            if x_new >= object_size[1] - probe_shape[3] + 1:
                x_new = 1 + x - x_perturbation
            if y_new >= object_size[0] - probe_shape[2] + 1:
                y_new = 1 + y - y_perturbation
            scan.append((y_new, x_new))
    scan = np.array(scan, dtype=np.float32)
    return scan

def l2_error(true_object, reconstructed_object):
    # true object is (1,2,H,W)
    # reconstructed_object object is (1,2,H,W)
    term1 = np.vdot(reconstructed_object, reconstructed_object)
    term2 = np.vdot(true_object, true_object)
    term3 = - 2 * np.abs(np.vdot(reconstructed_object, true_object))
    l2_error = np.real(np.sqrt(term1 + term2 + term3))

    return l2_error


def rPIE(measurement, object_size, scan, probe):
    # object_size is a tuple, e.g., (H1,W1)
    # scan is a np array with shape (S,2)
    # probe is a tensor with shape (1,2,H2,W2)
    # measurement is a tensor with (1,S,2,H2,W2)
    device = measurement.device
    # convert probe into a complex array (1,1,1,H2,W2)
    probe = np.squeeze(probe.detach().cpu().numpy(), 0)
    probe = probe[0,:,:] + 1j * probe[1,:,:]
    probe = np.expand_dims(probe, (0,1,2)) # expand probe dims to make it compatible with tike (1,1,1,H2,W2)
    # convert measurement into a complex array (S,H2,W2)
    measurement = np.squeeze(measurement.detach().cpu().numpy(), 0)
    measurement = measurement[:,0,:,:]
    # initial estimate of the object
    psi = 0.5 * np.ones(object_size) + 1j * 0.0
    # RPIE
    parameters = tike.ptycho.PtychoParameters(
        # Provide initial guesses for parameters that are updated
        probe = probe,
        scan = scan,
        psi = psi,
        # Probe options
        probe_options = tike.ptycho.ProbeOptions(), # uses default settings for probe recovery # None yapinca probe u update etmiyor
        object_options = tike.ptycho.ObjectOptions(
            # The object will be updated.
            use_adaptive_moment = True,  # smoothness constraint will use our provided setting
            # other object options will be default values
        ),
        position_options = None, # indicates that positions will not be updated
        algorithm_options = tike.ptycho.RpieOptions(
            num_iter = 250,
            num_batch = 1))
    result = tike.ptycho.reconstruct(
        data = measurement,
        parameters = parameters)
    # convert the result into a tensor with shape (1,2,H2,W2)
    result = result.psi
    result = np.expand_dims(result, 0)
    result = np.stack((np.real(result), np.imag(result)), 1)
    result = torch.from_numpy(result).float().to(device)
    return result


# Perform the adjoint test here.
if __name__ == '__main__':

    object_size = (512, 512)
    probe_size = (1, 2, 128, 128)

    probe = torch.randn(*probe_size)
    scan = cartesian_scan_pattern(object_size, probe_size, step_size = 32, sigma = 1)

    x = torch.randn(1, 2, *object_size)
    y_tilde = ptycho_forward_op(x, scan, probe)
    y_1 = torch.randn_like(y_tilde)
    x_tilde = ptycho_adjoint_op(y_1, scan, probe, object_size)

    # Convert everything back to complex tensors
    y_tilde = torch.complex(y_tilde[0,:,0,:,:], y_tilde[0,:,1,:,:])
    y_1 = torch.complex(y_1[0,:,0,:,:], y_1[0,:,1,:,:])
    x = torch.complex(x[0,0,:,:], x[0,1,:,:])
    x_tilde = torch.complex(x_tilde[0,0,:,:], x_tilde[0,1,:,:])

    print(torch.sum(torch.conj(y_1) * y_tilde))
    print(torch.sum(torch.conj(x) * x_tilde))
