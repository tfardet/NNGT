#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Matplotlib customization """

import itertools
import matplotlib.pyplot as plt



#
#---
# Palette
#------------------------

palette = plt.cm.Set1


#
#---
# Customize PyPlot
#------------------------

try:
    import seaborn as sns
    #~ sns.set(style='ticks', palette='Set2')
    sns.set_style("whitegrid")
    def sns_palette(arr):
        return sns.color_palette("Set2", len(arr))
    palette = sns_palette
except:
    try:
        plt.rcParams['font.size'] = 12
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
        plt.rcParams['axes.titlesize'] = 1.2*plt.rcParams['font.size']
        plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
        plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
        plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.format'] = 'pdf'
        plt.rcParams['xtick.major.size'] = 3
        plt.rcParams['xtick.minor.size'] = 3
        plt.rcParams['xtick.major.width'] = 1
        plt.rcParams['xtick.minor.width'] = 1
        plt.rcParams['ytick.major.size'] = 3
        plt.rcParams['ytick.minor.size'] = 3
        plt.rcParams['ytick.major.width'] = 1
        plt.rcParams['ytick.minor.width'] = 1
        plt.rcParams['legend.frameon'] = False
        plt.rcParams['legend.numpoints'] = 1
        plt.rcParams['axes.linewidth'] = 1
        plt.rcParams['axes.grid'] = True
        plt.rcParams['path.simplify'] = True
    except:
        pass


def format_exponent(ax, axis='y', pos=(1.,0.), valign="top", halign="right"):
    # Change the ticklabel format to scientific format
    ax.ticklabel_format(axis=axis, style='sci', scilimits=(-3, 2))
    # Get the appropriate axis
    if axis == 'y':
        ax_axis = ax.yaxis
    else:
        ax_axis = ax.xaxis
    # Run plt.tight_layout() because otherwise the offset text doesn't update
    plt.tight_layout()
    ##### THIS IS A BUG 
    ##### Well, at least it's sub-optimal because you might not
    ##### want to use tight_layout(). If anyone has a better way of 
    ##### ensuring the offset text is updated appropriately
    ##### please comment!

    # Get the offset value
    offset = ax_axis.get_offset_text().get_text()
    if len(offset) > 0:
        # Get that exponent value and change it into latex format
        minus_sign = u'\u2212'
        expo = float(offset.replace(minus_sign, '-').split('e')[-1])
        offset_text = r'x$\mathregular{10^{%d}}$' %expo
        # Turn off the offset text that's calculated automatically
        ax_axis.offsetText.set_visible(False)
        ax.text(pos[0], pos[1], offset_text, transform=ax.transAxes,
               horizontalalignment=halign,
               verticalalignment=valign)
    return ax
