"""Okabe-Ito color palette + cividis colormap.

Color-blind-safe defaults shared by every figure in the paper.
The 8-color Okabe-Ito set is recommended by Wong (2011) Nature Methods
and is designed to be discriminable under deuteranopia, protanopia, and
tritanopia. cividis (Nuñez et al. 2018) is a sequential colormap with
near-monochromatic appearance for color-vision-deficient viewers.

Use:
    from _palette import OI, set_rcparams
    set_rcparams()
    ax.bar(..., color=OI['blue'])
"""

import matplotlib as mpl

OI = {
    'black':   '#000000',
    'orange':  '#E69F00',
    'sky':     '#56B4E9',
    'green':   '#009E73',
    'yellow':  '#F0E442',
    'blue':    '#0072B2',
    'red':     '#D55E00',
    'purple':  '#CC79A7',
    'gray':    '#999999',
}

# Categorical assignment for the five models.
# Pull uses purple (not sky) so it does not compress against the
# Hybrid v4 blue under deuteranopia. Every figure that names a model
# must pull from this dict so colors stay consistent across the paper.
MODEL_COLORS = {
    'Vicsek':    OI['orange'],
    'Pull':      OI['purple'],
    'Hybrid v3': OI['gray'],
    'Hybrid v4': OI['blue'],
    'Hybrid v5': OI['green'],
}

# Redundant categorical encoding for grayscale/print reproduction.
MODEL_HATCHES = {
    'Vicsek':    '',
    'Pull':      '//',
    'Hybrid v3': 'xx',
    'Hybrid v4': '..',
    'Hybrid v5': '\\\\',
}

SEQUENTIAL_CMAP = 'cividis'


def set_rcparams():
    mpl.rcParams['font.family']      = 'sans-serif'
    mpl.rcParams['font.sans-serif']  = ['Helvetica', 'Nimbus Sans',
                                        'Liberation Sans', 'Arial',
                                        'DejaVu Sans']
    mpl.rcParams['mathtext.fontset'] = 'stixsans'
    mpl.rcParams['axes.prop_cycle']  = mpl.cycler(
        color=[OI['blue'], OI['red'], OI['green'], OI['orange'],
               OI['purple'], OI['sky'], OI['yellow'], OI['black']])
