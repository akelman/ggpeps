figsize = (4.14, 2.66)
fontstyle = {"family": "serif", "size": 10, "sans-serif": ["Palatino"]}
fontstyle_legend = {"fontsize": 8}
lw = 1
markersize = 3


def create_markers(color):
    return {
        "markersize": markersize,
        "markeredgecolor": color,
        "markeredgewidth": 0.75,
        "markerfacecolor": adjust_lightness(color, 1.50),
    }


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
