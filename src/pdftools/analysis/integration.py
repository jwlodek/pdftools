from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum, StrEnum
import sys
import typing as tp

import numpy as np
from numpy import ndarray
from pyFAI import AzimuthalIntegrator
from pyFAI.geometry import Geometry

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy import ndarray
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from dataclasses import dataclass
import numpy as np

from skbeam.core.accumulators.binned_statistic import BinnedStatistic1D
from skbeam.core.mask import margin
from .jittools import mask_ring_median, mask_ring_mean


@dataclass
class IntegrationSettings:
    npt: int = 1480
    correctSolidAngle: bool = False
    method: str = 'splitpixel'
    unit: str = 'q_A^-1'
    safe: bool = False

# default visualization setting
_LABEL = {
    "q_A^-1": r"Q ($\mathrm{\AA}^{-1}$)",
    "q_nm^-1": r"Q ({nm}$^{-1}$)",
    "2th_deg": r"2$\theta$ (deg)",
    "2th_rad": r"2$\theta$ (rad)",
    "r_mm": r"radius (mm)"
}


def subtract_background(raw_image: np.ndarray, background_image: np.ndarray, background_scale: float = 1.0) -> np.ndarray:
    """Subtract the background image from the data image inplace.

    Parameters
    ----------
    raw_image : np.ndarray
        The 2D diffraction image array.
    background_image : np.ndarray
        The 2D background image array.
    background_scale : float
        The scale of the the background image.
    
    Returns
    -------
    bg_subtracted_image : np.ndarray
        The background subtracted image.

    Raises
    ------
    ValueError
        If the shape of the background image does not match the shape of the data image.
    """

    if background_image.shape != raw_image.shape:
        raise ValueError(f"Unmatched shape between two images: {background_image.shape}, {raw_image.shape}.")
    return raw_image - background_scale * background_image

# TODO, should just remove this.
def average_images(images: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Average the 2D images.

    Parameters
    ----------
    images : list[np.ndarray]
        The 2D array of diffraction images.
    weights : list[float] | None
        The weights for the images. If None, images will not be weighted when averaged.

    Returns
    -------
    avg_img : np.ndarray
        The averaged 2D image array.
    """

    return np.average(np.stack(tuple(images), axis=0), axis=0, weights=weights)


class MaskingMethod(Enum):
    MEDIAN = "median"
    MEAN = "mean"

def mask_image(
    image: np.ndarray[tuple[int, int]],
    binner: BinnedStatistic1D,
    edge: int = 30,
    lower_thresh: float = 0.0,
    upper_thresh: float | None = None,
    alpha: float = 2.0,
    auto_type: MaskingMethod = MaskingMethod.MEDIAN,
    initial_mask: np.ndarray[tuple[int, int]] | None = None,
    pool=None,
) -> np.ndarray[tuple[int, int]]:
    """Mask an image based off of various methods

    Parameters
    ----------
    image: np.ndarray
        The image to be masked
    binner : BinnedStatistic1D instance
        The binned statistics information
    edge: int, optional
        The number of edge pixels to mask. Defaults to 30. If None, no edge
        mask is applied
    lower_thresh: float, optional
        Pixels with values less than or equal to this threshold will be masked.
        Defaults to 0.0. If None, no lower threshold mask is applied
    upper_thresh: float, optional
        Pixels with values greater than or equal to this threshold will be
        masked.
        Defaults to None. If None, no upper threshold mask is applied.
    alpha: float, optional
        Then number of acceptable standard deviations, if tuple then we use
        a linear distribution of alphas from alpha[0] to alpha[1], if array
        then we just use that as the distribution of alphas. Defaults to 3.
        If None, no outlier masking applied.
    auto_type: MaskingMethod, optional
        The type of binned outlier masking to be done, 'median' is faster,
        where 'mean' is more accurate, defaults to 'median'.
    initial_mask: np.ndarray[tuple[int, int]] | None, optional
        The starting mask to be compounded on. Defaults to None. If None mask
        generated from scratch.
    pool : Executor instance
        A pool against which jobs can be submitted for parallel processing

    Returns
    -------
    working_mask: np.ndarray[tuple[int, int]]
        The mask as a boolean array. True pixels are good pixels, False pixels
        are masked out.

    """

    if initial_mask is None:
        working_mask = np.ones(np.shape(image)).astype(bool)
    else:
        working_mask = initial_mask.copy()
    if edge:
        working_mask *= margin(np.shape(image), edge)
    if lower_thresh is not None:
        working_mask *= (image >= lower_thresh).astype(bool)
    if upper_thresh is not None:
        working_mask *= (image <= upper_thresh).astype(bool)
    if alpha:
        working_mask *= binned_outlier(
            image,
            binner,
            alpha=alpha,
            initial_mask=working_mask,
            mask_method=auto_type,
            pool=pool,
        )
    working_mask = working_mask.astype(bool)
    return working_mask


def binned_outlier(image: np.ndarray[tuple[int, int]], binner: BinnedStatistic1D, initial_mask: np.ndarray[tuple[int, int]], alpha: float = 3.0, mask_method: MaskingMethod = MaskingMethod.MEDIAN, pool=None):
    """Mask outliers based on sigma clipping.

    Parameters
    ----------
    image : np.ndarray
        The image
    binner : BinnedStatistic1D instance
        The binned statistics information
    alpha : float, optional
        The number of standard deviations to clip, defaults to 3
    initial_mask : np.ndarray[tuple[int, int]] | None, optional
        Prior mask. If None don't use a prior mask, defaults to None.
    mask_method : MaskingMethod, optional
        The method to use for creating the mask, median is faster, mean is more
        accurate. Defaults to median.
    pool : Executor instance
        A pool against which jobs can be submitted for parallel processing

    Returns
    -------
    np.ndarray:
        The mask
    """
    if pool is None:
        pool = ThreadPoolExecutor(max_workers=20)
    # skbeam 0.0.12 doesn't have argsort_index cached
    idx = binner.argsort_index
    flattened_initial_mask = initial_mask.flatten()
    tmsk2 = flattened_initial_mask[idx]
    vfs = image.flatten()[idx]
    pfs = np.arange(np.size(image))[idx]
    t = []
    i = 0
    for k in binner.flatcount:
        m = tmsk2[i : i + k]
        vm = vfs[i : i + k][m]
        if k > 0 and len(vm) > 0:
            t.append((vm, (pfs[i : i + k][m]), alpha))
        i += k
    p_err = np.seterr(all="ignore")
    # only run tqdm on mean since it is slow
    with pool as p:
        futures = [p.submit(getattr(sys.modules[__name__], f"mask_ring_{mask_method.value}"), *x) for x in t]
    removals = []
    for f in as_completed(futures):
        removals.extend(f.result())
    np.seterr(**p_err)
    flattened_initial_mask[removals] = False
    initial_mask = flattened_initial_mask.reshape(np.shape(image))
    return initial_mask.astype(bool)


def generate_map_bin(geo: Geometry, image_shape: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Create a q map and the pixel resolution bins

    Parameters
    ----------
    geo : pyFAI.geometry.Geometry instance
        The calibrated geometry
    img_shape : tuple[int, int] | None, optional
        The shape of the image, if None pull from the mask. Defaults to None.

    Returns
    -------
    q : np.ndarray
        The q map
    qbin : np.ndarray
        The pixel resolution bins
    """
    r: np.ndarray = geo.rArray(image_shape)  # type: ignore
    q: np.ndarray = geo.qArray(image_shape) / 10  # type: np.ndarray
    q_dq: np.ndarray = geo.deltaQ(image_shape) / 10  # type: np.ndarray

    pixel_size = (geo.pixel1, geo.pixel2)
    if not all(isinstance(ps, (int, float)) for ps in pixel_size):
        raise ValueError(f"Pixel size must be a number, got {pixel_size}")

    rres = np.hypot(*pixel_size)
    rbins = np.arange(np.min(r) - rres / 2., np.max(r) + rres / 2., rres / 2.)
    rbinned = BinnedStatistic1D(r.ravel(), statistic=np.max, bins=rbins) # type: ignore

    qbin_sizes = rbinned(q_dq.ravel())
    qbin_sizes = np.nan_to_num(qbin_sizes)
    qbin = np.cumsum(qbin_sizes)
    qbin[0] = np.min(q_dq)
    if np.max(q) > qbin[-1]:
        qbin[-1] = np.max(q)
    return q, qbin


def auto_mask(
    img: np.ndarray,
    ai: AzimuthalIntegrator,
    user_mask: np.ndarray | None = None,
    mask_setting: dict = {}
) -> tuple[np.ndarray, dict]:
    """Automatically generate the mask of the image.

    Parameters
    ----------
    img : np.ndarray
        The 2D diffraction image array.
    ai : AzimuthalIntegrator
        The AzimuthalIntegrator instance.
    mask_setting : dict
        The user's modification to auto-masking settings.
    user_mask : np.ndarray
        A mask provided by user. It is an integer array. 0 are good pixels, 1 are masked out.

    Returns
    -------
    mask : np.ndarray
        The mask as an integer array. 0 are good pixels, 1 are masked out.

    _mask_setting : dict
        The whole mask_setting.
    """
    if mask_setting is not None:
        _mask_setting = mask_setting
    else:
        _mask_setting = dict()
    pixel_map, bins = generate_map_bin(ai, img.shape)
    mask = user_mask.flatten() if user_mask is not None else None
    binner = BinnedStatistic1D(pixel_map.flatten(), bins=bins, mask=mask)  # type: ignore
    tmsk = np.invert(user_mask.astype(bool)) if user_mask is not None else None
    mask = mask_image(img, binner, tmsk=tmsk, **_mask_setting)
    mask = np.invert(mask).astype(int)
    return mask, _mask_setting


def mask_image_pyfai(img: np.ndarray[tuple[int, int]], binner: BinnedStatistic1D, tmsk: np.ndarray | None = None):
    """
    Mask an image based off of various methods

    Parameters
    ----------
    img: np.ndarray
        The image to be masked
    binner : BinnedStatistic1D instance
        The binned statistics information
    tmsk: np.ndarray, optional
        The starting mask to be compounded on. Defaults to None. If None mask
        generated from scratch.

    Returns
    -------
    tmsk: np.ndarray
        The mask as a int array. 0 pixels are good pixels, 1 pixels
        are masked out.
    """
    mask = np.invert(tmsk.astype(bool)) if tmsk is not None else None
    mask = mask_image(img, binner, tmsk=mask)
    mask = np.invert(mask).astype(int)
    return mask



def integrate(
    image: np.ndarray, ai: AzimuthalIntegrator, mask: np.ndarray | None = None, integration_settings: dict = {}
) -> tuple[np.ndarray, IntegrationSettings]:
    """Use AzimuthalIntegrator to integrate the image.

    Parameters
    ----------
    image : np.ndarray
        The 2D diffraction image array.
    ai : AzimuthalIntegrator
        The AzimuthalIntegrator instance.
    mask : np.ndarray, optional
        The mask as a 0 and 1 array. 0 pixels are good pixels, 1 pixels are masked out.
    integration_settings : dict, default {}
        The user's modification to integration settings.

    Returns
    -------
    chi : np.ndarray
        The chi data. The first row is bin centers and the second row is the average intensity in bins.

    settings: IntegrationSettings
        The whole integration setting.
    """

    settings = IntegrationSettings(**integration_settings)
    xy = ai.integrate1d(image, mask=mask, **settings.__dict__)
    chi = np.stack(xy)
    return chi, settings


def visualize_image(image: np.ndarray, mask: np.ndarray | None = None, matshow_settings: dict = {}, show: bool = True) -> Axes:
    """Visualize the processed image. The color map will be determined by statistics of the pixel values. The color map
    is determined by mean +/- z_score * std.

    Parameters
    ----------
    image : np.ndarray
        The 2D diffraction image array.
    mask: np.ndarray, optional
        The mask as a 0 and 1 array. 0 pixels are good pixels, 1 pixels are masked out.
    matshow_settings : dict
        The user's modification to imshow kwargs except a special key 'z_score'.
    show : bool, default True
        If True, show the figure.

    Returns
    -------
    ax : Axes
        The axes with the image shown.
    """
    fig = plt.figure()
    ax: Axes = fig.add_subplot(111)
    if mask is not None:
        img = np.ma.masked_array(image, mask)
    mean, std = image.mean(), image.std()
    z_score = matshow_settings.pop('z_score', 2.0)
    kwargs = {
        'vmin': mean - z_score * std,
        'vmax': mean + z_score * std
    }
    kwargs.update(**matshow_settings)
    img_obj = ax.matshow(image, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # color bar with magical settings to make it same size as the plot
    plt.colorbar(img_obj, ax=ax, fraction=0.046, pad=0.04)
    if show:
        plt.show(block=False)
    return ax


def vis_chi(chi: ndarray, plot_setting: dict = None, unit: str = None, show: bool = True) -> Axes:
    """Visualize the chi curve.

    Parameters
    ----------
    chi : ndarray
        The chi data. The first row is bin centers and the second row is the average intensity in bins.

    plot_setting : dict
        The kwargs for the plot function.

    unit: str
        The unit of the chi data. It affects the label of the plot. If None, no unit.

    show : bool
        If True, show the figure.

    Returns
    -------
    ax : Axes
        The axes with the curve plotted.
    """
    if plot_setting is None:
        plot_setting = dict()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(chi[0], chi[1], **plot_setting)
    if unit:
        ax.set_xlabel(_LABEL.get(unit))
    ax.set_ylabel('I (A. U.)')
    if show:
        plt.show(block=False)
    return ax




def get_chi(
    ai: AzimuthalIntegrator,
    img: ndarray,
    dk_img: ndarray = None,
    bg_img: ndarray = None,
    mask: ndarray = None,
    bg_scale: float = None,
    mask_setting: tp.Union[str, dict] = None,
    integ_setting: dict = None,
    img_setting: tp.Union[str, dict] = None,
    plot_setting: tp.Union[str, dict] = None,
) -> tp.Tuple[
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    tp.Union[None, ndarray],
    dict,
    tp.Union[str, dict]
]:
    """Process the diffraction image to get I(Q).

    The image will be subtracted by the background image and then auto masked. The results will be
    binned on the azimuthal direction and the average value of the intensity in the bin and their
    corresponding Q will be returned. The I(Q) and background subtracted masked image will
    be visualized.

    Parameters
    ----------
    ai : AzimuthalIntegrator
        The AzimuthalIntegrator.

    img : ndarray
        The of the 2D array of the image.

    dk_img : ndarray
        The dark frame image. The image will be subtracted by it if provided.

    bg_img : ndarray
        The 2D array of the background image. If None, no background subtraction.

    mask : ndarray
        A mask provided by user. The auto generated mask will be multiplied by this mask.

    bg_scale : float
        The scale for background subtraction. If None, use 1.

    mask_setting : dict
        The auto final_mask setting.
        If None, use _AUTOMASK_SETTING in the tools module. To turn off the auto masking, use "OFF".

    integ_setting : dict
        The integration setting.
        If None, use _INTEG_SETTING in the tools module.

    img_setting : dict
        The user's modification to imshow kwargs except a special key 'z_score'. If None, use use empty dict.
        To turn off the imshow, use "OFF".

    plot_setting : dict
        The kwargs for the plot function. If None, use empty dict.

    Returns
    -------
    chi : ndarray
        The 2D array of integrated results. The first row is the Q and the second row is the I.

    bg_sub_img : ndarray
        The background subtracted image. If no background subtraction, it is the dk_sub_image.

    dk_sub_img : ndarray
        The dark subtracted image. If no dark subtraction, it is the input img.

    img : ndarray
        The input image.

    final_mask : ndarray or None
        The final_mask array. If no auto_masking, return None.

    _integ_setting: dict
        The integration setting used.

    _mask_setting : dict or str
        The auto masking setting.
    """
    if dk_img is not None:
        dk_sub_img = bg_sub(img, dk_img, bg_scale=1.)
    else:
        dk_sub_img = img
    if bg_img is not None:
        bg_sub_img = bg_sub(dk_sub_img, bg_img, bg_scale=bg_scale)
    else:
        bg_sub_img = dk_sub_img
    if mask_setting != "OFF":
        final_mask, _mask_setting = auto_mask(bg_sub_img, ai, user_mask=mask, mask_setting=mask_setting)
    elif mask is not None:
        final_mask, _mask_setting = mask, {}
    else:
        final_mask, _mask_setting = None, None
    if img_setting != "OFF":
        vis_img(bg_sub_img, final_mask, img_setting=img_setting)
    chi, _integ_setting = integrate(bg_sub_img, ai, mask=final_mask, integ_setting=integ_setting)
    if plot_setting != "OFF":
        vis_chi(chi, plot_setting=plot_setting, unit=_integ_setting.get('unit'))
    return chi, bg_sub_img, dk_sub_img, img, final_mask, _integ_setting, _mask_setting

