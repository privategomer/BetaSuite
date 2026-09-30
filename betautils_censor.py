"""
betautils_censor.py - Turning a tracked box into censored pixels.

Two entry points:

    process_raw_box( raw, vid_w, vid_h )
        One raw detection -> one censorable box dict, or None if the
        label is not censored or the score does not clear its gate.
        Runs once per detection, before tracking.

    censor_img_for_boxes( image, boxes )
        One frame -> that frame with every box rendered onto it. Runs
        once per rendered frame, so everything it touches is on the
        hottest path in the codebase.

THREAD SAFETY
-------------
censor_img_for_boxes is called concurrently by parallel render workers
(see betautils_render). It therefore must not mutate the box dicts it is
given, and it does not: it sorts a copy, and the overlap-merge
strategies build new dicts rather than editing the originals. That was
also a correctness bug in its own right - the pre-2.1 'single-pass'
merge widened the caller's box dicts in place, and because merges only
ever expand, a censor region ratcheted outward frame after frame and
never shrank back.

The shape-mask cache and the sticker cache are both safe to share.
"""

import functools
import glob
import math
import os
import random
import threading

import cv2
import numpy as np

import betaconfig
import betautils_config as bu_config


# Largest blur kernel, as a fraction of the region's shorter side, that
# still averages real pixels rather than BORDER_DEFAULT reflection. A
# correctness guard, not a style control - see _clamp_kernel_to_region.
# Per method, because a flat box window reaches about twice as far as a
# Gaussian of the same kernel size (sigma k/sqrt(12) vs OpenCV's
# ~0.15k), so it saturates at a smaller fraction of the region.
BLUR_KERNEL_MAX_REGION_FRACTION = { 'gaussian': 1.0, 'box': 0.5, 'triple_box': 1.0 }


# Number of box passes 'triple_box' makes. Three is the point where the
# central limit theorem has done essentially all the work it is going to:
# repeated box convolution converges to a Gaussian, and by the third pass
# the shape is within a few percent of one (Kovesi, "Fast Almost-Gaussian
# Filtering"). A fourth pass costs another full pass for a difference
# measured below the noise floor of the render bench.
TRIPLE_BOX_PASSES = 3


# ---------------------------------------------------------------------------
# Strength scaling
# ---------------------------------------------------------------------------

def censor_scale_for_image_box( image, feature_w, feature_h ):
    """
    Multiplier applied to a blur/pixel style's 'strength'.

    Per betaconfig.censor_scale_strategy:

      'feature'  (recommended) min(feature_w, feature_h) / 100
          Scales with the censored thing's own size, so the same
          strength reads as similarly strong whether the subject is
          close to camera or far away.
      'image'    max(img_h, img_w) / 1000
          Scales with the frame instead, so strength is constant
          relative to the picture rather than the subject.
      'none'     1
          strength is used as a raw pixel count.

    An unrecognised strategy returns 1 rather than None. The pre-2.1
    version fell off the end of the function and returned None, which
    became `strength * None` and a TypeError deep inside a render chunk.

    THE SIZE PASSED IN MUST BE STABLE ACROSS A TRACK'S FRAMES
    ---------------------------------------------------------
    Under 'feature' the multiplier is min(w, h)/100, so the blur kernel
    and the pixel block size are recomputed from whatever box size this
    frame happens to have. A tracked box jitters: measured on 640m,
    exposed_vulva's smoothed frame-to-frame jump has a median of 7.6px
    and a p90 of 40.8px. At strength 54 an 18px wobble moves the blur
    kernel by 10px, frame after frame - and a kernel that changes every
    frame changes the visible amount of blur every frame.

    That is what reads as "the blur flickers". It is not the style
    switching; it is one style whose strength is being recomputed from a
    moving measurement. Callers therefore pass the TRACK's stable
    reference size (see censor_image) rather than the live box, so a
    continuous censor keeps a constant kernel while its position and
    extent still follow the subject.
    """
    strategy = getattr( betaconfig, 'censor_scale_strategy', 'feature' )
    if strategy == 'feature':
        return min( feature_w, feature_h ) / 100
    if strategy == 'image':
        img_h, img_w = image.shape[:2]
        return max( img_h, img_w ) / 1000
    return 1


# ---------------------------------------------------------------------------
# Style and shape resolution
# ---------------------------------------------------------------------------
#
# Anywhere a censor_style is configured it may be either a single style
# dict (always used) or a list of style dicts (one chosen at random,
# weighted by each dict's optional 'weight', default 1). The choice is
# made ONCE per tracked instance, in betautils_track.smooth_boxes, and
# then held for that instance's whole lifetime - otherwise a continuous
# censor region flickers between styles frame to frame.

def resolve_censor_style( style_config ):
    """
    Pick one concrete style dict from what is configured.

    Args:
        style_config: A style dict, or a list of them to choose from.

    Returns:
        One concrete style dict.
    """
    if isinstance( style_config, dict ):
        return style_config
    weights = [ one_style.get( 'weight', 1 ) for one_style in style_config ]
    return random.choices( style_config, weights=weights, k=1 )[0]


def resolve_censor_shape( resolved_style, item_default_shape ):
    """
    What shape a resolved style renders as.

    Precedence, most to least specific:
      1. the style dict's own 'shape' - a deliberate per-variant
         override, e.g. one bar variant wanting a rectangle while its
         sibling blur variants stay elliptical.
      2. betaconfig.type_default_shapes[style['type']] - a structural
         default for types that only make sense as one shape. 'bar' is
         a rectangle by definition, and the span/single-pass merges only
         fire for shape == 'box', so a bar style that forgets to set
         'shape' would otherwise inherit the item's shape and silently
         never merge.
      3. item_default_shape - the item's censor_shape, or
         default_censor_shape.

    Args:
        resolved_style: One concrete style dict.
        item_default_shape: The item's own shape, as the last fallback.

    Returns:
        'box', 'circle' or 'ellipse'.
    """
    if 'shape' in resolved_style:
        return resolved_style['shape']
    type_default = getattr( betaconfig, 'type_default_shapes', {} ).get( resolved_style['type'] )
    if type_default:
        return type_default
    return item_default_shape


# ---------------------------------------------------------------------------
# Shape masks (cached)
# ---------------------------------------------------------------------------
#
# A mask depends only on (shape, width, height, feather) - never on
# pixel content - so it is a pure function and can be memoised. It was
# rebuilt from scratch for every box on every frame before 2.1, and the
# feather step is a Gaussian blur whose kernel scales with the box:
# feather 0.55 on a 300px box is a 167-tap blur, 17.6ms, once per frame.
#
# The cache stores the finished float32 alpha so the per-frame uint8 ->
# float conversion is also avoided. Entries are marked read-only; every
# consumer only ever reads them.

_MASK_CACHE_ENTRIES = 128


def make_shape_mask( shape, w, h ):
    """
    A binary 0/255 mask of the given shape at the given size.

    Args:
        shape: 'circle', 'ellipse', or anything else (treated as 'box').
        w: Width in pixels.
        h: Height in pixels.

    Returns:
        An HxW uint8 array: 255 inside the shape, 0 outside. A 'box'
        mask is entirely 255, which is the same as no masking at all.
    """
    if shape == 'circle':
        mask = np.zeros( ( h, w ), dtype=np.uint8 )
        radius = min( w, h ) // 2
        cv2.circle( mask, ( w//2, h//2 ), radius, 255, cv2.FILLED )
        return mask
    if shape == 'ellipse':
        mask = np.zeros( ( h, w ), dtype=np.uint8 )
        cv2.ellipse( mask, ( w//2, h//2 ), ( max( w//2, 1 ), max( h//2, 1 ) ),
                     0, 0, 360, 255, cv2.FILLED )
        return mask
    return np.full( ( h, w ), 255, dtype=np.uint8 )


def feather_mask( mask, feather, w, h ):
    """
    Soften a binary mask's edge into a gradient.

    Implemented by Gaussian-blurring the mask itself, so instead of a
    step at the boundary there is a ramp: edge pixels blend
    proportionally between censored and original rather than jumping.

    Args:
        mask: A binary 0/255 mask.
        feather: 0-1, the fraction of the shape's shorter dimension used
            as the transition width. 0 returns mask unchanged.
        w: mask's width.
        h: mask's height.

    Returns:
        The mask, blurred, or unchanged when feather is falsy.
    """
    if not feather:
        return mask
    feather_radius_px = max( 1, round( feather * min( w, h ) / 2 ) )
    kernel_size = 2*feather_radius_px + 1
    return cv2.GaussianBlur( mask, ( kernel_size, kernel_size ), 0 )


@functools.lru_cache( maxsize=_MASK_CACHE_ENTRIES )
def _alpha_for( shape, w, h, feather ):
    """
    Cached float32 alpha plane for one (shape, size, feather).

    Returns:
        A read-only (h, w, 1) float32 array in [0, 1].
    """
    mask = feather_mask( make_shape_mask( shape, w, h ), feather, w, h )
    alpha = ( mask.astype( np.float32 ) / 255.0 )[ :, :, None ]
    alpha.flags.writeable = False
    return alpha


def mask_cache_info():
    """Cache statistics for the shape-mask cache, for the benchmark harness."""
    return _alpha_for.cache_info()


def clear_mask_cache():
    """Drop every cached mask. Test-support and benchmark-support only."""
    _alpha_for.cache_clear()


def apply_masked_region( image, x, y, w, h, censored_region, shape, feather=0 ):
    """
    Composite an already-censored region back into image.

    Args:
        image: The frame to composite into. Mutated in place and
            returned, for chaining.
        x, y: Top-left of the region in image pixel coordinates.
        w, h: Region size in pixels.
        censored_region: An HxWx3 array of the finished censored pixels.
        shape: 'box', 'circle' or 'ellipse'.
        feather: 0-1 soft-edge amount.

    Returns:
        image.
    """
    if shape == 'box' and not feather:
        # The common fast path: a plain rectangular overwrite needs no
        # mask at all.
        image[ y:y+h, x:x+w ] = censored_region
        return image

    alpha = _alpha_for( shape, w, h, feather )
    original_region = image[ y:y+h, x:x+w ].astype( np.float32 )
    blended = alpha*censored_region.astype( np.float32 ) + ( 1 - alpha )*original_region
    image[ y:y+h, x:x+w ] = blended.astype( image.dtype )
    return image


# ---------------------------------------------------------------------------
# Pixelation
# ---------------------------------------------------------------------------

def pixelate_block_grid( region, w, h, factor ):
    """
    Square-grid pixelation: downsample to a coarse grid, then upsample
    nearest-neighbour so each cell renders as one flat block.

    What most tools call either "pixelation" or "mosaic" - the same
    technique under two names, which is why both patterns route here.

    Args:
        region: HxWx3 region to pixelate.
        w, h: region's size.
        factor: Block size in source pixels. Larger means chunkier.

    Returns:
        An HxWx3 array the same size as region.
    """
    downsampled_w = max( 1, math.ceil( w/factor ) )
    downsampled_h = max( 1, math.ceil( h/factor ) )
    small = cv2.resize( region, ( downsampled_w, downsampled_h ), interpolation=cv2.INTER_LINEAR )
    return cv2.resize( small, ( w, h ), interpolation=cv2.INTER_NEAREST )


# Hex cell-label grids, keyed on (w, h, quantised radius). Purely
# geometric, so a tracked box of stable size reuses one grid for its
# whole lifetime.
_HEX_GRID_CACHE_ENTRIES = 64
_SQRT3 = math.sqrt( 3.0 )


@functools.lru_cache( maxsize=_HEX_GRID_CACHE_ENTRIES )
def _hex_cell_labels( w, h, radius_milli ):
    """
    Map every pixel of a w x h region to a flat-top hexagon cell index.

    Analytic, and vectorised over the whole region at once: convert
    pixel coordinates to fractional axial hex coordinates, cube-round to
    the nearest hex centre, then pack the axial pair into a dense
    integer label.

    This replaces a loop that, for every hexagon in the grid, allocated
    a full region-sized mask, fillPoly'd one hexagon into it, and
    np.where'd over the entire region to find that hexagon's pixels -
    O(cells x w x h) where the work is O(w x h). On a 600x600 box that
    was 174ms per box per frame, or roughly 4.4 seconds of render time
    per second of video for one box.

    The tiling is also now a correct, non-overlapping hexagonal
    tessellation. The old generator offset alternate rows by a full hex
    width while stepping columns by 1.5 radii, which produced
    overlapping hexagons that partially overwrote each other. The
    rendered pattern therefore looks slightly different: regular where
    it used to be subtly irregular.

    Args:
        w: Region width in pixels.
        h: Region height in pixels.
        radius_milli: Hex radius (centre to corner) in thousandths of a
            pixel. Quantised so nearly-identical radii share a grid.

    Returns:
        A read-only (h, w) int32 array of dense cell labels, and the
        label count, as a (labels, count) pair.
    """
    radius = max( 1e-3, radius_milli / 1000.0 )

    ys, xs = np.mgrid[ 0:h, 0:w ]
    # Flat-top axial coordinates (Amit Patel's hexagon reference).
    frac_q = ( 2.0/3.0 * xs ) / radius
    frac_r = ( -1.0/3.0 * xs + _SQRT3/3.0 * ys ) / radius

    # Cube-round: round all three cube coordinates, then correct the one
    # that moved furthest so the triple stays on the x+y+z=0 plane.
    cube_x, cube_z = frac_q, frac_r
    cube_y = -cube_x - cube_z
    round_x, round_y, round_z = np.rint( cube_x ), np.rint( cube_y ), np.rint( cube_z )
    diff_x, diff_y, diff_z = ( np.abs( round_x - cube_x ),
                               np.abs( round_y - cube_y ),
                               np.abs( round_z - cube_z ) )
    fix_x = ( diff_x > diff_y ) & ( diff_x > diff_z )
    fix_y = ( ~fix_x ) & ( diff_y > diff_z )
    fix_z = ( ~fix_x ) & ( ~fix_y )
    axial_q = np.where( fix_x, -round_y - round_z, round_x ).astype( np.int64 )
    axial_r = np.where( fix_z, -round_x - round_y, round_z ).astype( np.int64 )

    q_min, r_min = axial_q.min(), axial_r.min()
    r_span = int( axial_r.max() - r_min ) + 1
    labels = ( ( axial_q - q_min ) * r_span + ( axial_r - r_min ) ).astype( np.int32 )

    # Densify: cube rounding can leave gaps in the packed index space,
    # and bincount allocates by maximum label, so remap to 0..n-1.
    unique_labels, dense = np.unique( labels, return_inverse=True )
    dense = dense.astype( np.int32 ).reshape( h, w )
    dense.flags.writeable = False
    return dense, len( unique_labels )


def pixelate_hex_grid( region, w, h, factor ):
    """
    Hexagonal-grid pixelation: fill each hex cell with the mean colour
    of the source pixels under it.

    Genuinely a different look from the square grid, not an alias.

    Args:
        region: HxWx3 region to pixelate.
        w, h: region's size.
        factor: Controls cell size; the hex radius is factor/2, floored
            at 2 source pixels.

    Returns:
        An HxWx3 array the same size as region.
    """
    hex_radius = max( 2.0, factor / 2.0 )
    labels, label_count = _hex_cell_labels( w, h, int( round( hex_radius * 1000 ) ) )

    flat_labels = labels.reshape( -1 )
    pixel_counts = np.bincount( flat_labels, minlength=label_count ).astype( np.float64 )
    pixel_counts[ pixel_counts == 0 ] = 1.0

    source = region.reshape( -1, region.shape[2] ).astype( np.float64 )
    result = np.empty_like( region )
    for channel in range( region.shape[2] ):
        channel_sums = np.bincount( flat_labels, weights=source[ :, channel ],
                                    minlength=label_count )
        channel_means = channel_sums / pixel_counts
        result[ :, :, channel ] = channel_means[ labels ].astype( region.dtype )
    return result


def pixelate_image( image, x, y, w, h, style, shape='box', feather=0,
                    scale_w=None, scale_h=None ):
    """
    Apply a 'pixel' censor style to one region.

    Args:
        image: The frame to censor. Mutated in place and returned.
        x, y, w, h: The region, in image pixel coordinates.
        style: The resolved style dict. Reads 'strength' (default 10)
            and 'pattern' ('square' | 'mosaic' | 'hex', default 'square').
        shape: Passed through to apply_masked_region.
        feather: Passed through to apply_masked_region.

    Returns:
        image.
    """
    block_factor = style.get( 'strength', 10 ) * censor_scale_for_image_box(
        image, scale_w if scale_w else w, scale_h if scale_h else h )
    region = image[ y:y+h, x:x+w ]
    pattern = style.get( 'pattern', 'square' )
    if pattern == 'hex':
        censored_region = pixelate_hex_grid( region, w, h, block_factor )
    else:
        censored_region = pixelate_block_grid( region, w, h, block_factor )
    return apply_masked_region( image, x, y, w, h, censored_region, shape, feather )


# ---------------------------------------------------------------------------
# Blur
# ---------------------------------------------------------------------------

def _gaussian_sigma_for_kernel( kernel_size ):
    """OpenCV's own sigma-from-kernel formula, so approximations match."""
    return 0.3 * ( ( kernel_size - 1 ) * 0.5 - 1 ) + 0.8


def _clamp_kernel_to_region( kernel_size, region_w, region_h, method='gaussian' ):
    """
    Hold a blur kernel to something the region can actually support.

    WHY THIS EXISTS
    ---------------
    A Gaussian kernel wider than the region it covers has no real pixels
    left to average at the centre, so cv2 fills the window from
    BORDER_DEFAULT edge reflection instead. Past that point the output
    stops being "this region, blurred" and becomes a function of
    whatever sits at the region's BORDER - and a tracked box's border
    moves and resizes every frame even when its centre sits still.

    That is what reads as a blur "pulsing in strength" while staying in
    place. Measured on a 180px box with the kernel held perfectly
    constant, mean brightness inside the censor swung:

        strength  50 -> kernel  91 (0.51x box) -> 0.08 levels
        strength  70 -> kernel 127 (0.71x box) -> 0.54 levels
        strength  96 -> kernel 175 (0.97x box) -> 1.80 levels
        strength 130 -> kernel 235 (1.31x box) -> 4.13 levels

    The onset is monotonic from about 0.8x. Note the kernel was NOT
    changing in that test: holding the kernel steady (style_scale_w/h)
    does not help here, because the varying input is the border, not the
    kernel. The two fixes are independent and both are needed.

    WHAT THE LIMIT IS, AND WHAT IT IS NOT
    -------------------------------------
    BLUR_KERNEL_MAX_REGION_FRACTION is a correctness guard at 1.0, not a
    style control. At 1.0 it only clamps kernels that were already past
    the point of meaning something, so a normally-configured style never
    reaches it and nothing about the intended look changes. Tuning the
    amount of blur is what 'strength' is for; lowering this fraction to
    get a softer censor would silently cap every style at once.

    Clamping REDUCES blur, so it can only ever reveal more than an
    unclamped kernel would have. That is why the limit sits at 1.0 and
    not lower: below 1.0 it would start trading real obscuring power for
    stability on styles that were working correctly.

    Args:
        kernel_size: The odd kernel the strength calculation produced.
        region_w, region_h: The region's real pixel extent this frame.
        method: 'gaussian' or 'box'. A flat box window reaches further
            than a Gaussian of the same kernel size, so it gets a
            tighter fraction.

    Returns:
        An odd kernel size >= 3, no larger than the region's shorter
        side times this method's BLUR_KERNEL_MAX_REGION_FRACTION.
    """
    shorter_side = min( region_w, region_h )
    if shorter_side <= 0:
        return kernel_size

    fraction = BLUR_KERNEL_MAX_REGION_FRACTION.get(
        method, BLUR_KERNEL_MAX_REGION_FRACTION['gaussian'] )
    limit = int( shorter_side * fraction )
    # cv2 requires odd; step DOWN so the result never exceeds the limit.
    if limit % 2 == 0:
        limit -= 1
    limit = max( 3, limit )

    return min( kernel_size, limit )


def _approximate_gaussian_blur( region, kernel_size ):
    """
    A heavy Gaussian blur, computed at reduced resolution.

    Downsample by an integer factor, blur with the proportionally
    smaller sigma, upsample back. For an obscuring blur the result is
    visually indistinguishable from the exact kernel, because the
    detail the approximation loses is exactly the detail the blur is
    there to destroy.

    Measured against the exact kernel:
        200x200 region, k=101    6.73ms -> 0.11ms   (61x)
        300x300 region, k=121   13.52ms -> 0.41ms   (33x)

    Args:
        region: HxWx3 region to blur.
        kernel_size: The odd kernel size the exact blur would have used.

    Returns:
        A blurred copy of region, same shape and dtype.
    """
    h, w = region.shape[:2]
    sigma = _gaussian_sigma_for_kernel( kernel_size )

    # Keep at least ~8 pixels of blur radius at the reduced resolution,
    # so the small blur still has enough support to look smooth.
    downscale = max( 1, int( sigma // 4 ) )
    small_w = max( 1, w // downscale )
    small_h = max( 1, h // downscale )
    if downscale <= 1 or small_w < 4 or small_h < 4:
        return cv2.GaussianBlur( region, ( kernel_size, kernel_size ), 0,
                                 borderType=cv2.BORDER_DEFAULT )

    small = cv2.resize( region, ( small_w, small_h ), interpolation=cv2.INTER_AREA )
    small_sigma = sigma / downscale
    small_kernel = 2 * max( 1, int( math.ceil( 3 * small_sigma ) ) ) + 1
    small = cv2.GaussianBlur( small, ( small_kernel, small_kernel ), small_sigma,
                              borderType=cv2.BORDER_DEFAULT )
    return cv2.resize( small, ( w, h ), interpolation=cv2.INTER_LINEAR )


def _box_width_for_sigma( sigma, passes=TRIPLE_BOX_PASSES ):
    """
    The box width whose repeated application approximates one Gaussian.

    A box window of width w has variance (w^2 - 1)/12, so `passes` of
    them contribute passes*(w^2 - 1)/12. Setting that equal to sigma^2
    and solving for w gives the form below (Kovesi, "Fast Almost-Gaussian
    Filtering"). Forced odd so the window is centred; an even window
    shifts the image by half a pixel per pass, which over three passes is
    a visible drift of the censored area.

    The algebra matters: writing this as sigma*sqrt(12/passes) + 1 - the
    same symbols in the wrong order - comes out about two pixels wide at
    every kernel, which over three passes is a systematically stronger
    blur than the Gaussian it claims to match.
    """
    width = int( round( math.sqrt( 12.0 * sigma * sigma / passes + 1.0 ) ) )
    if width % 2 == 0:
        width += 1
    return max( 3, width )


def _triple_box_blur( region, kernel_size ):
    """
    A Gaussian-shaped blur built from repeated box passes.

    WHY THIS EXISTS ALONGSIDE 'gaussian'
    ------------------------------------
    cv2.blur is a running-sum filter: its cost per pixel is independent
    of the window width, while cv2.GaussianBlur's grows with the kernel.
    At the kernel sizes censoring uses, three box passes cost a fraction
    of one true Gaussian and land within a few percent of it.

    Measured on synthetic detail at matched obscuring power (400x400
    region): exact Gaussian k=101 took 12.53ms, this took 1.02ms at k=87
    for the same residual detail, with a mean absolute difference from
    the exact result of 0.38 levels out of 255.

    Unlike _approximate_gaussian_blur this never resamples, so its output
    does not depend on an integer downscale factor - the thing that made
    the blur visibly step between strengths when a box jittered.

    REJECTED ALTERNATIVE: cv2.stackBlur
    -----------------------------------
    Tried and deliberately not used. It is fast, but its blur strength is
    NOT monotonic in kernel size at the sizes censoring needs. Measured
    residual detail on the same fixture: k=31 gave 11.3, k=101 gave 15.0,
    k=401 gave 72.8 - i.e. past about k=31 a larger kernel obscures LESS.
    A censor style whose strength silently reverses above a threshold is
    the wrong shape of risk, whatever it costs.
    """
    sigma = _gaussian_sigma_for_kernel( kernel_size )
    width = _box_width_for_sigma( sigma )
    blurred = region
    for _ in range( TRIPLE_BOX_PASSES ):
        blurred = cv2.blur( blurred, ( width, width ), borderType=cv2.BORDER_DEFAULT )
    return blurred


def _blur_region_with_margin( image, x, y, w, h, kernel_size ):
    """
    The region to blur, widened into the real pixels around the box.

    WHAT THIS FIXES
    ---------------
    The residual blur flicker that survived stabilising the kernel. A
    blur whose kernel is a large fraction of its region reads mostly
    BORDER_DEFAULT edge reflection rather than real content, and that
    reflection is a function of where the box's edge happens to sit. So
    the blurred result changed every time the box resized by a pixel,
    even with the kernel held perfectly constant.

    Measured on synthetic detail, box jittering +/-4px with a constant
    kernel, mean level swing inside a fixed crop:

        kernel/region   no margin   with margin
            0.30          1.31         0.00
            0.51          2.19         0.00
            0.76          2.74         0.00
            0.99          3.04         0.00

    Reading real neighbours instead of reflected ones removes the
    dependence entirely, because those neighbours do not move when the
    box resizes.

    The margin is only ever READ. What gets composited back is still
    exactly the box, so this cannot change what is censored.

    Returns:
        A (region, offset_x, offset_y) triple. The offsets locate the
        box's own top-left corner inside the returned region, which is
        where the caller crops the blurred result back to.
    """
    margin = kernel_size // 2
    image_h, image_w = image.shape[:2]
    x0 = max( 0, x - margin )
    y0 = max( 0, y - margin )
    x1 = min( image_w, x + w + margin )
    y1 = min( image_h, y + h + margin )
    return image[ y0:y1, x0:x1 ], x - x0, y - y0


def blur_image( image, x, y, w, h, style, shape='box', feather=0,
                scale_w=None, scale_h=None ):
    """
    Apply a 'blur' censor style to one region.

    Args:
        image: The frame to censor. Mutated in place and returned.
        x, y, w, h: The region, in image pixel coordinates.
        style: The resolved style dict. Reads 'strength' (default 20)
            and 'method' ('gaussian' | 'box', default 'gaussian').
        shape: Passed through to apply_masked_region.
        feather: Passed through to apply_masked_region.

    Returns:
        image.

    Note:
        'box' is cv2.blur, a flat average over a square window. It is
        marginally cheaper but tends to look smeary at the edges of the
        blurred area; 'gaussian' weights the window's centre more and
        generally reads as smoother at the same radius. It is kept as an
        option for the look, not for the speed - see
        blur_fast_approximation for the speed.

        The kernel is clamped so it cannot exceed the region it blurs -
        see _clamp_kernel_to_region for why an oversized one flickers.
    """
    # Both the strength scaling AND the region clamp read the track's
    # stable reference size, never this frame's box.
    #
    # Only the first half of that was true before. The kernel was
    # computed from style_scale and then re-clamped against the LIVE
    # box, which jitters - so the clamp handed back the per-frame
    # variation the stable size had just removed. It is silent at low
    # strength, because a small kernel never approaches the limit, and
    # it bites exactly where a censor is being pushed hardest.
    #
    # Measured on a real 640m cache (28 tracks, exposed_breast) the
    # per-track kernel swing was 0 at strength 60, 18px median and 110px
    # peak at strength 140, and the downscale factor inside
    # _approximate_gaussian_blur changed mid-track on 14 of 21 tracks.
    # That factor is int(sigma // 4), so a few pixels of box jitter steps
    # it 10 -> 9 and re-samples the region on a different grid: the blur
    # visibly drops from strong to weak and back. Clamping against the
    # reference size instead takes all three of those to zero.
    scale_ref_w = scale_w if scale_w else w
    scale_ref_h = scale_h if scale_h else h
    kernel_size = 2*math.ceil( style.get( 'strength', 20 )
                               * censor_scale_for_image_box(
                                   image, scale_ref_w, scale_ref_h ) / 2 ) + 1
    method = style.get( 'method', 'gaussian' )
    kernel_size = _clamp_kernel_to_region( kernel_size, scale_ref_w, scale_ref_h, method )

    # Deliberately NOT re-clamped against the live w/h afterwards.
    #
    # The obvious-looking min(reference, live) only gets part way: on a
    # real cache it left 7 of 21 tracks still changing downscale factor,
    # because the live box is smaller than the reference on 52% of boxes
    # (median 0.89x), so the live term keeps winning and keeps jittering.
    #
    # Dropping it is safe in the direction that matters. Clamping only
    # ever REDUCES a kernel, so declining to clamp can only blur more,
    # never less - it cannot under-censor. The cost is that a box much
    # smaller than its track's reference gets a kernel larger than its
    # own region, which reads as a flat wash rather than as detail. That
    # is a look tradeoff, and the stable version is the one that matches
    # what a person actually sees as "strong blur".
    # Read a margin of real neighbouring pixels when asked to, so the
    # blur is not a function of where the box's edge sits. The margin is
    # read only; what gets composited back is still exactly the box.
    use_margin = getattr( betaconfig, 'blur_edge_margin', True )
    if use_margin:
        region, offset_x, offset_y = _blur_region_with_margin(
            image, x, y, w, h, kernel_size )
    else:
        region, offset_x, offset_y = image[ y:y+h, x:x+w ], 0, 0

    if method == 'box':
        censored_region = cv2.blur( region, ( kernel_size, kernel_size ),
                                    borderType=cv2.BORDER_DEFAULT )
    elif method == 'triple_box':
        censored_region = _triple_box_blur( region, kernel_size )
    else:
        # The downscale approximation is skipped when a margin is in use:
        # it resamples on a grid derived from the region's size, which
        # reintroduces exactly the size dependence the margin removes.
        # triple_box is the fast path that is compatible with a margin.
        approximate = ( getattr( betaconfig, 'blur_fast_approximation', True )
                        and not use_margin )
        approximation_min_kernel = getattr( betaconfig, 'blur_approximation_min_kernel', 21 )
        if approximate and kernel_size >= approximation_min_kernel:
            censored_region = _approximate_gaussian_blur( region, kernel_size )
        else:
            censored_region = cv2.GaussianBlur( region, ( kernel_size, kernel_size ), 0,
                                                borderType=cv2.BORDER_DEFAULT )

    # Crop the blurred result back to the box itself.
    censored_region = censored_region[ offset_y:offset_y+h, offset_x:offset_x+w ]

    return apply_masked_region( image, x, y, w, h, censored_region, shape, feather )


# ---------------------------------------------------------------------------
# Bar
# ---------------------------------------------------------------------------

def bar_image( image, x, y, w, h, style, shape='box', feather=0 ):
    """
    Apply a 'bar' censor style: a solid-colour rectangle.

    Args:
        image: The frame to censor. Mutated in place and returned.
        x, y, w, h: The detection box, in image pixel coordinates. y and
            h are adjusted internally when thickness != 1.0.
        style: The resolved style dict. Reads 'color' (RGB as
            configured, default black) and 'thickness'.
        shape: Passed through to apply_masked_region.
        feather: Passed through to apply_masked_region.

    Returns:
        image.

    Note:
        thickness is the fraction of the box's height the bar covers,
        centred within it. 1.0 fills the box. Below 1.0 draws a thinner
        strip and leaves the box's top and bottom as original pixels,
        which is what makes a bar read as a censor strip rather than a
        filled block. Above 1.0 grows past the box (clamped to the
        frame), for guaranteeing a thin item stays fully covered.
    """
    color_bgr = tuple( reversed( style.get( 'color', ( 0, 0, 0 ) ) ) )
    thickness = style.get( 'thickness', 1.0 )
    if thickness != 1.0:
        img_h = image.shape[0]
        bar_h = max( 1, round( h * max( 0.01, thickness ) ) )
        y = max( 0, min( y + ( h - bar_h )//2, max( 0, img_h - bar_h ) ) )
        h = min( bar_h, img_h - y )
    image = np.ascontiguousarray( image )
    censored_region = np.full( ( h, w, 3 ), color_bgr, dtype=image.dtype )
    return apply_masked_region( image, x, y, w, h, censored_region, shape, feather )


# ---------------------------------------------------------------------------
# Sticker
# ---------------------------------------------------------------------------
#
# No new ML models and no live emoji-glyph rendering: colour emoji needs
# either a colour-capable font run through a rendering stack that
# supports colour glyphs (plain PIL/cv2 text drawing renders monochrome
# outlines only) or a bundled image set. The dependency-free route is to
# point config at a folder of PNGs - ideally with alpha, for a clean
# silhouette - and composite them directly. OpenMoji and Twemoji both
# publish plain PNG sets that drop straight in.

_sticker_cache = {}
_sticker_cache_lock = threading.Lock()


def _load_stickers( style ):
    """
    Load and cache every usable sticker PNG a style points at.

    Thread-safe: parallel render workers share one cache, and loading
    the same folder twice concurrently would otherwise duplicate the
    decode work and the warning.

    Args:
        style: The resolved style dict. Reads 'dir' (globbed for
            '*.png') and/or 'images' (an explicit path list).

    Returns:
        A list of HxWx4 BGRA arrays, one per image that loaded. Images
        with no alpha channel are given a fully opaque one. Empty when
        nothing loaded; sticker_image falls back to a bar in that case.
    """
    cache_key = style.get( 'dir' ) or tuple( style.get( 'images', [] ) )
    with _sticker_cache_lock:
        if cache_key in _sticker_cache:
            return _sticker_cache[cache_key]

        candidate_paths = list( style.get( 'images', [] ) )
        if style.get( 'dir' ):
            candidate_paths += sorted( glob.glob( os.path.join( style['dir'], '*.png' ) ) )

        loaded_images = []
        for path in candidate_paths:
            img = cv2.imread( path, cv2.IMREAD_UNCHANGED )
            if img is None:
                bu_config.warn_once( "sticker image not found or unreadable: %s"%(path) )
                continue
            if img.shape[2] == 3:
                opaque_alpha = np.full( img.shape[:2], 255, dtype=np.uint8 )
                img = np.dstack( [ img, opaque_alpha ] )
            loaded_images.append( img )

        _sticker_cache[cache_key] = loaded_images
        if not loaded_images:
            bu_config.warn_once(
                "sticker style configured with no usable images (dir=%r, images=%r) - "
                "falling back to a bar censor for these boxes."%(
                    style.get( 'dir' ), style.get( 'images' ) ) )
        return loaded_images


def sticker_image( image, x, y, w, h, style, shape='box', feather=0, sticker_seed=None ):
    """
    Apply a 'sticker' censor style: composite a PNG over the box.

    Args:
        image: The frame to censor. Mutated in place and returned.
        x, y, w, h: The box, in image pixel coordinates.
        style: The resolved style dict. Reads 'dir'/'images' and 'scale'
            (default 1.0).
        shape: Unused when a sticker is drawn - the PNG's own alpha is
            its silhouette. Honoured by the bar fallback.
        feather: Softens the sticker's alpha edge further, on top of
            whatever the source PNG already has. Also passed to the
            fallback.
        sticker_seed: A float in [0, 1) that deterministically selects
            WHICH sticker to draw, so one tracked instance keeps one
            sticker. This function runs once per rendered frame, not
            once per track, so without a stable seed a sticker style
            flips through the whole folder frame by frame. None picks
            randomly each call, for box dicts predating this field.

    Returns:
        image.
    """
    stickers = _load_stickers( style )
    if not stickers:
        return bar_image( image, x, y, w, h, { 'color': ( 0, 0, 0 ) }, shape, feather )

    if sticker_seed is None:
        sticker_index = random.randrange( len( stickers ) )
    else:
        sticker_index = min( len( stickers ) - 1, int( sticker_seed * len( stickers ) ) )
    sticker = stickers[ sticker_index ]

    scale = style.get( 'scale', 1.0 )
    target_w = max( 1, round( w * scale ) )
    target_h = max( 1, round( h * scale ) )
    resized_sticker = cv2.resize( sticker, ( target_w, target_h ), interpolation=cv2.INTER_AREA )

    origin_x = x + w//2 - target_w//2
    origin_y = y + h//2 - target_h//2
    img_h, img_w = image.shape[:2]
    src_x0, src_y0 = max( 0, -origin_x ), max( 0, -origin_y )
    dst_x0, dst_y0 = max( 0, origin_x ), max( 0, origin_y )
    dst_x1 = min( img_w, origin_x + target_w )
    dst_y1 = min( img_h, origin_y + target_h )
    composite_w, composite_h = dst_x1 - dst_x0, dst_y1 - dst_y0
    if composite_w <= 0 or composite_h <= 0:
        return image

    # cv2.imread always returns BGR(A) whatever the source PNG's own
    # channel order, and the frame buffer is BGR too, so no reversal is
    # needed here. A previous version assumed RGB and reversed, which
    # rendered every yellow-toned emoji blue.
    sticker_rgb = resized_sticker[ src_y0:src_y0+composite_h, src_x0:src_x0+composite_w, :3 ]
    sticker_alpha = resized_sticker[
        src_y0:src_y0+composite_h, src_x0:src_x0+composite_w, 3:4 ].astype( np.float32 ) / 255.0
    if feather:
        feathered = feather_mask( ( sticker_alpha[ :, :, 0 ]*255 ).astype( np.uint8 ),
                                  feather, composite_w, composite_h )
        sticker_alpha = ( feathered.astype( np.float32 )/255.0 )[ :, :, None ]

    destination = image[ dst_y0:dst_y1, dst_x0:dst_x1 ].astype( np.float32 )
    blended = sticker_alpha*sticker_rgb.astype( np.float32 ) + ( 1 - sticker_alpha )*destination
    image[ dst_y0:dst_y1, dst_x0:dst_x1 ] = blended.astype( image.dtype )
    return image


# ---------------------------------------------------------------------------
# Debug overlay and watermark
# ---------------------------------------------------------------------------

def debug_image( image, box ):
    """
    Draw a labelled debug rectangle for one box instead of censoring it.

    Active when betaconfig.debug_mode's bit 0 is set. Useful for
    eyeballing detection quality without touching items_to_censor.

    Args:
        image: The frame to draw on. Mutated in place and returned.
        box: The box being visualised.

    Returns:
        image.
    """
    x, y, w, h = box['x'], box['y'], box['w'], box['h']
    color_bgr = tuple( reversed( box['censor_style'].get( 'color', ( 255, 0, 255 ) ) ) )
    image = np.ascontiguousarray( image )
    cv2.rectangle( image, ( x, y ), ( x+w, y+h ), color_bgr, 3 )
    cv2.putText( image, '(%d,%d)'%(x, y),     ( x+10, y+20 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1 )
    cv2.putText( image, '(%d,%d)'%(x+w, y+h), ( x+10, y+40 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1 )
    cv2.putText( image, box['label'],         ( x+10, y+60 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1 )
    cv2.putText( image, '%.2f %.1f %.1f'%( box['score'], box['start'], box['end'] ),
                 ( x+10, y+80 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1 )
    return image


def watermark_image( image ):
    """Draw the BetaSuite watermark, if betaconfig enables it."""
    if not getattr( betaconfig, 'enable_betasuite_watermark', False ):
        return image
    image = np.ascontiguousarray( image )
    h, w = image.shape[:2]
    text_scale = max( min( w/750, h/750 ), 1 )
    cv2.putText( image, 'Censored with betasuite.net', ( 20, math.ceil( 20*text_scale ) ),
                 cv2.FONT_HERSHEY_PLAIN, text_scale, ( 0, 0, 255 ), math.floor( text_scale ) )
    return image


def annotate_image_shape( image ):
    """Draw image.shape as text in the corner. Debugging aid."""
    image = np.ascontiguousarray( image )
    cv2.putText( image, str( image.shape ), ( 20, 20 ),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 0, 255 ), 2 )
    return image


# ---------------------------------------------------------------------------
# Raw detection -> censorable box
# ---------------------------------------------------------------------------

def compute_safe_geometry( raw_x, raw_y, raw_w, raw_h, vid_w, vid_h,
                           x_area_safety, y_area_safety ):
    """
    Pad a raw box by the given area-safety fractions and clamp it to the
    frame.

    Split out of process_raw_box so betautils_track can re-run the
    identical arithmetic with a STYLE's own area safety once a track's
    style is resolved, rather than being stuck with the item default.

    Args:
        raw_x, raw_y, raw_w, raw_h: The unpadded box, in pixels.
        vid_w, vid_h: Frame size, in pixels.
        x_area_safety: Fractional horizontal padding. Negative shrinks.
        y_area_safety: Fractional vertical padding. Negative shrinks.

    Returns:
        A (safe_x, safe_y, safe_w, safe_h) tuple, always inside the
        frame, always at least 1x1.

    Note:
        The START coordinate is clamped BEFORE width and height are
        computed. A raw model box can land past the frame edge (a
        detection straddling into the black padding the preprocessor
        adds, scaled back up by a large picture size). Without that
        clamp, safe_x could reach or exceed vid_w; the w/h clamp still
        forces w >= 1, but image[y:y+h, x:x+w] with x >= vid_w returns an
        EMPTY array rather than raising, which is what crashed
        cv2.GaussianBlur with "!_src.empty()" deep inside a render chunk.
    """
    safe_x = math.floor( max( 0, raw_x - raw_w*x_area_safety/2 ) )
    safe_y = math.floor( max( 0, raw_y - raw_h*y_area_safety/2 ) )
    safe_x = min( safe_x, max( 0, vid_w-1 ) )
    safe_y = min( safe_y, max( 0, vid_h-1 ) )
    safe_w = max( 1, math.ceil( min( vid_w-safe_x, raw_w*( 1+x_area_safety ) ) ) )
    safe_h = max( 1, math.ceil( min( vid_h-safe_y, raw_h*( 1+y_area_safety ) ) ) )
    return safe_x, safe_y, safe_w, safe_h


def process_raw_box( raw, vid_w, vid_h, parts_to_blur=None ):
    """
    Turn one raw detection into a censorable box, or reject it.

    Args:
        raw: A raw box dict from a detector adapter:
            {'x','y','w','h','class_id','score','t','size'}.
        vid_w, vid_h: Frame size, in pixels.
        parts_to_blur: The resolved per-label settings. Passed in by the
            caller so the whole table is built once per video instead of
            once per detection - before 2.1 this function called
            get_parts_to_blur() itself, rebuilding and re-resolving the
            entire table for every one of tens of thousands of raw
            detections.

    Returns:
        A censorable box dict, or None when the label is not censored or
        the score does not clear the label's entry gate.

    Note on hysteresis:
        A detection scoring below min_prob but at or above
        min_prob_continue is admitted and marked 'provisional'. A
        provisional box may only CONTINUE an already-established track,
        never start one - see betautils_track.smooth_boxes. When
        min_prob_continue is unset the two gates are identical and
        nothing is ever provisional, which is exactly the pre-2.1
        behaviour.
    """
    if parts_to_blur is None:
        parts_to_blur = bu_config.get_parts_to_blur()

    label = raw['class_id']
    settings = parts_to_blur.get( label )
    if settings is None:
        return None

    min_prob = settings['min_prob']
    min_prob_continue = settings.get( 'min_prob_continue' )
    entry_floor = min_prob if min_prob_continue is None else min( min_prob, min_prob_continue )
    if raw['score'] <= entry_floor:
        return None

    x_area_safety = settings['width_area_safety']
    y_area_safety = settings['height_area_safety']
    time_safety = settings['time_safety']
    safe_x, safe_y, safe_w, safe_h = compute_safe_geometry(
        raw['x'], raw['y'], raw['w'], raw['h'], vid_w, vid_h, x_area_safety, y_area_safety )

    return {
        'start': max( raw['t'] - time_safety/2, 0 ),
        'end':   raw['t'] + time_safety/2,
        't': raw['t'],
        'x': safe_x, 'y': safe_y, 'w': safe_w, 'h': safe_h,
        # Raw, pre-safety geometry plus the frame size, kept so
        # betautils_track can recompute padding from a per-style area
        # safety once this box's track has a resolved style. Leading
        # underscore: internal, not part of the user-facing box schema.
        '_raw_x': raw['x'], '_raw_y': raw['y'], '_raw_w': raw['w'], '_raw_h': raw['h'],
        '_vid_w': vid_w, '_vid_h': vid_h,
        # Still the CONFIGURED value here (a dict, or a list to choose
        # from) - not yet resolved to one concrete style. Resolution
        # happens once per tracked instance in betautils_track, so a
        # continuous detection does not flicker between random styles.
        'censor_style': settings['censor_style'],
        'censor_shape': settings['censor_shape'],
        # A stable per-box variant pick, currently read only by
        # sticker_image. Rolled once here rather than inside
        # sticker_image so a box rendered across many frames keeps one
        # choice. Harmless and unread for every other style type.
        'censor_sticker_seed': random.random(),
        'label': label,
        'score': raw['score'],
        'size': raw.get( 'size', 0 ),
        'provisional': ( min_prob_continue is not None and raw['score'] <= min_prob ),
    }


# ---------------------------------------------------------------------------
# Per-frame rendering
# ---------------------------------------------------------------------------

def rectangles_intersect( box1, box2 ):
    """Whether two axis-aligned boxes overlap. Touching edges count."""
    if box1['x']+box1['w'] < box2['x']:
        return False
    if box1['y']+box1['h'] < box2['y']:
        return False
    if box1['x'] > box2['x']+box2['w']:
        return False
    if box1['y'] > box2['y']+box2['h']:
        return False
    return True


def censor_style_key( censor_style ):
    """
    A hashable identity for a resolved style, for grouping before merge.

    Two style dicts with identical contents produce equal keys even when
    they are different objects.
    """
    return tuple( sorted( censor_style.items() ) )


def censor_style_sort( censor_style ):
    """
    Sort key that puts visually similar styles adjacent.

    Styles of the same 'type' sort together (blur < bar < pixel <
    sticker < debug), with a secondary ordering inside blur/bar/pixel on
    strength or colour, so two randomised variants of one type still end
    up next to each other and can be considered for merging.
    """
    style_type = censor_style['type']
    if style_type == 'blur':
        return 1 + 1/max( censor_style.get( 'strength', 20 ), 1 )
    if style_type == 'bar':
        color = censor_style.get( 'color', ( 0, 0, 0 ) )
        return 2 + 1/( 2 + 255*3 - sum( color ) )
    if style_type == 'pixel':
        return 3 + censor_style.get( 'strength', 10 )
    if style_type == 'sticker':
        return 4
    if style_type == 'debug':
        return 99
    return 50


def collapse_boxes_for_style( piece, img_w=None, img_h=None ):
    """
    Merge same-label, same-style boxes per that style's overlap strategy.

    Args:
        piece: A non-empty list of boxes sharing one label and one
            resolved style, as grouped by censor_img_for_boxes.
        img_w: Frame width, needed by 'span' extension.
        img_h: Frame height. Accepted for symmetry; unused.

    Returns:
        A list of boxes to actually render. NEVER the caller's own dicts
        when a merge happens - merged results are fresh dicts. See this
        module's docstring for the in-place-mutation bug that rule fixes.
    """
    resolved_style = piece[0]['censor_style']
    style_type = resolved_style['type']
    shape = piece[0].get( 'censor_shape', 'box' )
    if shape != 'box':
        return piece

    # A style dict's own 'merge' overrides the type-level default, which
    # is what lets two 'bar' variants behave differently even though
    # censor_overlap_strategy only has one entry per style TYPE.
    strategy = resolved_style.get(
        'merge', betaconfig.censor_overlap_strategy.get( style_type, 'none' ) )
    if strategy == 'none':
        return piece

    if strategy == 'single-pass':
        merged_segments = []
        for box in piece:
            absorbed = False
            for segment in merged_segments:
                if rectangles_intersect( box, segment ):
                    merged_x = min( segment['x'], box['x'] )
                    merged_y = min( segment['y'], box['y'] )
                    segment['w'] = max( segment['x']+segment['w'], box['x']+box['w'] ) - merged_x
                    segment['h'] = max( segment['y']+segment['h'], box['y']+box['h'] ) - merged_y
                    segment['x'] = merged_x
                    segment['y'] = merged_y
                    absorbed = True
                    break
            if not absorbed:
                # A COPY. Seeding a segment with the caller's own dict
                # and then widening it in place permanently enlarged that
                # box for every later frame, and because merges only
                # expand, the region ratcheted outward and never shrank.
                merged_segments.append( dict( box ) )
        return merged_segments

    if strategy == 'span':
        # Bridge two same-style, same-label boxes into ONE covering
        # rectangle even when they do not overlap - a single bar across
        # both breast detections rather than two separate ones, as if a
        # line were drawn between them.
        #
        # Restricted to exactly 2 boxes on purpose: with three or more
        # same-label boxes live in one frame this is far more likely two
        # people than one person with an unusual body-part count, and
        # spanning a bar between two different people reads as one
        # region and is worse than doing nothing.
        if len( piece ) != 2:
            return piece
        box_a, box_b = piece[0], piece[1]
        span_x0 = min( box_a['x'], box_b['x'] )
        span_y0 = min( box_a['y'], box_b['y'] )
        span_x1 = max( box_a['x']+box_a['w'], box_b['x']+box_b['w'] )
        span_y1 = max( box_a['y']+box_a['h'], box_b['y']+box_b['h'] )

        # span_extend reaches PAST the tight two-box rectangle toward the
        # frame edges: 0.0 is exactly the tight rectangle, 1.0 is full
        # frame width. Horizontal only - "a line drawn between them" is
        # a horizontal idea; the vertical extent always just covers both.
        span_extend = resolved_style.get( 'span_extend', 0.0 )
        if span_extend and img_w:
            span_x0 = span_x0 + ( 0 - span_x0 ) * span_extend
            span_x1 = span_x1 + ( img_w - span_x1 ) * span_extend

        merged_box = dict( box_a )
        merged_box['x'] = int( span_x0 )
        merged_box['y'] = int( span_y0 )
        merged_box['w'] = max( 1, int( span_x1 - span_x0 ) )
        merged_box['h'] = max( 1, int( span_y1 - span_y0 ) )
        return [ merged_box ]

    # Unknown strategy: fail safe by censoring every box independently.
    return piece


def censor_image( image, box ):
    """
    Dispatch one resolved, collapsed box to its style's renderer.

    Args:
        image: The frame to censor. Mutated in place and returned.
        box: The box to render.

    Returns:
        image.

    Raises:
        ValueError: box['censor_style']['type'] is not a known type.
    """
    shape = box.get( 'censor_shape', 'box' )
    style = box['censor_style']
    feather = style.get( 'feather', 0 )
    style_type = style['type']

    # Every renderer below builds something sized from box w/h (a
    # feather mask, a hex cell grid, a resized sticker) and then combines
    # it with image[y:y+h, x:x+w]. numpy silently truncates that slice at
    # the frame edge, so a box hanging off the frame gives a region
    # smaller than w/h and the two no longer line up. That crashed whole
    # render chunks as "operands could not be broadcast" (blur, mosaic)
    # and "weights and list don't have the same length" (hex bincount).
    #
    # Tracking clamps boxes to the frame, but this is the one place every
    # style passes through, so it is the right place to guarantee it.
    # A COPY, never the caller's dict: render workers share one box list.
    img_h, img_w = image.shape[:2]
    x = max( 0, min( int( box['x'] ), img_w - 1 ) )
    y = max( 0, min( int( box['y'] ), img_h - 1 ) )
    w = max( 1, min( int( box['w'] ), img_w - x ) )
    h = max( 1, min( int( box['h'] ), img_h - y ) )
    if ( x, y, w, h ) != ( box['x'], box['y'], box['w'], box['h'] ):
        box = dict( box, x=x, y=y, w=w, h=h )

    # Strength scaling reads the track's stable reference size, not this
    # frame's box, so a continuous censor keeps one kernel while its
    # position and extent still follow the subject. See
    # censor_scale_for_image_box for why a per-frame size flickers.
    # Falls back to the live box for anything that has no track behind it
    # (betastare's stills, betavision's live capture, a hand-built box in
    # a bench or test), which is the pre-2.6 behaviour.
    scale_w = box.get( 'style_scale_w' )
    scale_h = box.get( 'style_scale_h' )

    if style_type == 'blur':
        return blur_image( image, box['x'], box['y'], box['w'], box['h'], style, shape, feather,
                           scale_w=scale_w, scale_h=scale_h )
    if style_type == 'pixel':
        return pixelate_image( image, box['x'], box['y'], box['w'], box['h'], style, shape, feather,
                               scale_w=scale_w, scale_h=scale_h )
    if style_type == 'bar':
        return bar_image( image, box['x'], box['y'], box['w'], box['h'], style, shape, feather )
    if style_type == 'sticker':
        return sticker_image( image, box['x'], box['y'], box['w'], box['h'], style, shape,
                              feather, box.get( 'censor_sticker_seed' ) )
    if style_type == 'debug':
        return debug_image( image, box )
    raise ValueError( "Unknown censor_style type %r"%(style_type,) )


def censor_img_for_boxes( image, boxes ):
    """
    Render every box onto one frame.

    Groups same-label / same-style boxes into pieces, merges each piece
    per its overlap strategy, draws the results, then watermarks.

    Args:
        image: The frame to censor.
        boxes: Boxes to render, each with a resolved censor_style and
            censor_shape. NOT mutated, and not reordered - this runs
            concurrently across render workers that share one box list.

    Returns:
        The censored frame.
    """
    if not boxes:
        return watermark_image( image )

    img_h, img_w = image.shape[:2]
    ordered = sorted( boxes, key=lambda box: (
        box['label'], censor_style_sort( box['censor_style'] ) ) )

    grouped_pieces = []
    for box in ordered:
        previous = grouped_pieces[-1][0] if grouped_pieces else None
        if ( previous is not None
                and previous['label'] == box['label']
                and censor_style_key( previous['censor_style'] ) == censor_style_key( box['censor_style'] ) ):
            grouped_pieces[-1].append( box )
        else:
            grouped_pieces.append( [ box ] )

    for piece in grouped_pieces:
        for collapsed_box in collapse_boxes_for_style( piece, img_w, img_h ):
            image = censor_image( image, collapsed_box )

    return watermark_image( image )
