import numpy as np
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from abtem.multislice import _generate_potential_configurations
from abtem.antialias import AntialiasAperture
from ase import units
from functools import partial

def move_probe(probe, new_pos):
    """
    Move the probe by a given shift.

    Args:
        probe: The probe to move.
        shift: The shift to apply to the probe.

    Returns:
        The probe after the shift.
    """
    current_pos_row = probe.shape[0]//2 #probe coordinates are in y, x
    current_pos_col = probe.shape[1]//2 #probe coordinates are in y, x
    new_pos_row = new_pos[0] #scan coordinates are given in x, y
    new_pos_col = new_pos[1] #scan coordinates are given in x, y

    # if probe.shape[0] % 2 == 1:
    #     current_pos_row += 1

    shift_to_row = new_pos_row - current_pos_row
    shift_to_col = new_pos_col - current_pos_col
    shift = jnp.array([shift_to_row, shift_to_col])

    return jnp.roll(probe, shift, axis=(0, 1))


def get_frequencies(n, m, ps):
    fx = jnp.fft.fftfreq(n, ps[0])
    fy = jnp.fft.fftfreq(m, ps[1])
    Fx, Fy = jnp.meshgrid(fx, fy, indexing='ij')
    return Fx, Fy


@jax.jit
def shift_kernel(x0, y0, Fx, Fy):
    return jnp.exp(-1j * 2 * jnp.pi * (Fx * x0 + Fy * y0))


def fresnel_propagation_kernel(n: int,
                               m: int,
                               ps: float,
                               z: float,
                               energy: float):

    wavelength = energy2wavelength(energy)
    Fx, Fy = get_frequencies(n, m, ps)

    H = jnp.exp(1j * (
        2 * jnp.pi / wavelength) * z) * jnp.exp(
        -1j * jnp.pi * wavelength * z * (Fx**2 + Fy**2))

    return H


def angular_spectrum_propagation_kernel(n: int,
                                        m: int,
                                        ps: float,
                                        z: float,
                                        energy: float):
    wavelength = energy2wavelength(energy)
    Fx, Fy = get_frequencies(n, m, ps)
    H = jnp.exp(1j * 2 * jnp.pi * z * jnp.sqrt(
        (1 / wavelength)**2 - Fx**2 - Fy**2))

    return H


def wpm_propagation_kernel(Ek, n_val, k0, k_perp2, dz):
    # Calculate the longitudinal wave vector component
    # kz = sqrt((n * k0)^2 - k_perp^2)
    kz = jnp.sqrt(jnp.array(n_val**2 * k0**2 - k_perp2, dtype=jnp.complex128))
    H = jnp.exp(1j * dz * kz)
    return jnp.fft.ifft2(H * Ek)


def wpm_step(wave, n_map, dz, energy, ps):
    """
    WPM step *without* using unique(n). One ifft2 per pixel's refractive index.

    Args:
        wave: complex array (ny, nx)
        n_map: refractive index map (ny, nx)
        dz: propagation distance
        energy: beam energy in eV
        ps: pixel size (dy, dx)
    """
    ny, nx = wave.shape
    wavelength = energy2wavelength(energy)
    k0 = 2 * jnp.pi / wavelength

    # Frequencies -> k_perp^2
    Fy, Fx = get_frequencies(ny, nx, ps)
    kx = 2 * jnp.pi * Fx
    ky = 2 * jnp.pi * Fy
    k_perp2 = kx**2 + ky**2

    # One FFT of the input wave
    Ek = jnp.fft.fft2(wave)

    # Flatten refractive index map to list of values
    n_flat = n_map.reshape(-1)

    # For each refractive index value, propagate the *whole* field
    # fields has shape (P, ny, nx)
    fields = jax.vmap(
        lambda n_val: wpm_propagation_kernel(Ek, n_val, k0, k_perp2, dz)
    )(n_flat)

    P = n_flat.size
    p_indices = jnp.arange(P)
    iy, ix = jnp.divmod(p_indices, nx) # each p -> (iy[p], ix[p])

    # For each pixel p, pick the value at (iy[p], ix[p]) from its field
    def pick_pixel(field, y, x):
        return field[y, x]

    new_wave_flat = jax.vmap(pick_pixel)(fields, iy, ix)   # (P,)

    # Reshape back to (ny, nx)
    new_wave = new_wave_flat.reshape(ny, nx)

    return new_wave


@jax.jit
def Propagator(u, H):
    ufft = jnp.fft.fft2(u)
    return jnp.fft.ifft2(H * ufft)


@jax.jit
def transmission_function(potential, energy):
    sigma = energy2sigma(energy)
    return jnp.exp(1j * sigma * potential)


def electron_refractive_index(potential, energy):
    E0 = electron_rest_energy()
    E = energy

    # Convert electrostatic potential (V) -> potential energy V (eV)
    # Electron charge = -1, so potential energy = -phi
    V = -potential

    EminusV = E - V

    numerator = 2 * EminusV * E0 + EminusV**2
    denominator = 2 * E * E0 + E**2

    n = jnp.sqrt(numerator / denominator)

    return n


def get_abtem_transmit(potential, energy):
    t_functions = []
    for _, potential_configuration in _generate_potential_configurations(
        potential
    ):
        for potential_slice in potential_configuration.generate_slices():
            transmission_function = potential_slice.transmission_function(
                energy=energy
            )
            transmission_function = AntialiasAperture().bandlimit(
                transmission_function, in_place=False
            )
            t_functions.append(transmission_function.array)

    return np.concatenate(t_functions, axis=0)


@jax.jit
def relativistic_mass_correction(energy: float) -> float:
    return 1 + units._e * energy / (units._me * units._c**2)


@jax.jit
def energy2mass(energy: float) -> float:
    """
    Calculate relativistic mass from energy.

    Parameters
    ----------
    energy: float
        Energy [eV].

    Returns
    -------
    float
        Relativistic mass [kg]
    """
    return relativistic_mass_correction(energy) * units._me


@jax.jit
def energy2sigma(energy: float) -> float:
    """
    Calculate interaction parameter from energy.

    Parameters
    ----------
    energy: float
        Energy [eV].

    Returns
    -------
    float
        Interaction parameter [1 / (Å * eV)].
    """
    return (
        2 * jnp.pi * energy2mass(energy) * units.kg * units._e * units.C *
        energy2wavelength(energy) /
        (units._hplanck * units.s * units.J) ** 2
    )


@jax.jit
def energy2wavelength(energy: float) -> float:
    """
    Calculate relativistic de Broglie wavelength from energy.

    Parameters
    ----------
    energy: float
        Energy [eV].

    Returns
    -------
    float
        Relativistic de Broglie wavelength [Å].
    """
    return (
        units._hplanck
        * units._c
        / jnp.sqrt(energy * (2 * units._me * units._c**2 / units._e + energy))
        / units._e
        * 1.0e10
    )


def electron_rest_energy():
    """
    Return the electron rest energy E0 = m_e c^2 in eV.
    """
    m_e = units._me
    c = units._c
    eV = units._e

    return m_e * c**2 / eV


@jdc.pytree_dataclass
class ProbeParamsFixed:
    wavelength: jdc.Static[float]
    alpha: jnp.array
    phi: jnp.array
    aperture: jnp.array


@jdc.pytree_dataclass
class ProbeParamsVariable:
    defocus: float = 0.
    astigmatism: float = 0.
    astigmatism_angle: float = 0.
    Cs: float = 0.
    coma: float = 0.
    coma_angle: float = 0.
    trefoil: float = 0.
    trefoil_angle: float = 0.


@jax.jit
def make_probe_fft(pp: ProbeParamsVariable, fpp: ProbeParamsFixed):
    alpha = fpp.alpha
    phi = fpp.phi
    aperture = fpp.aperture

    aberrations = jnp.zeros(alpha.shape, dtype=jnp.float32)
    aberrations += ((1 / 2) * alpha**2 * pp.defocus)
    aberrations += ((1 / 2) * alpha**2 * pp.astigmatism * jnp.cos(2 * (phi - pp.astigmatism_angle)))
    aberrations += ((1 / 3) * alpha**3 * pp.coma * jnp.cos(phi - pp.coma_angle))
    aberrations += ((1 / 3) * alpha**3 * pp.trefoil * jnp.cos(3 * (phi - pp.trefoil_angle)))
    aberrations += ((1 / 4) * alpha**4 * pp.Cs)
    aberrations *= (2 * jnp.pi / fpp.wavelength)
    # aberrations = jnp.exp(-1j * aberrations)
    aberrations = jnp.cos(-aberrations) + 1.0j * jnp.sin(-aberrations)

    probe_fft = jnp.ones(alpha.shape, dtype=jnp.complex64)
    probe_fft *= aperture
    probe_fft *= aberrations
    probe_fft /= jnp.linalg.norm(probe_fft)
    return probe_fft

