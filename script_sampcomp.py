"""Script for sampcomp"""
import sampcomp


def script_plot_1():
    """Single edge with autoregulation"""
    diagonal = 0.8
    sampcomp.plot_bounds(
        saveas="bhatta-bound-a{}-s0-d0.1.eps".format(diagonal),
        diagonal=diagonal,
    )
    sampcomp.plot_bounds(
        saveas="bhatta-bound-a{}-s0-d0.5.eps".format(diagonal),
        start_delta=0.5,
        diagonal=diagonal,
    )
    sampcomp.plot_bounds(
        saveas="bhatta-bound-a{}-s1-d0.5.eps".format(diagonal),
        start_delta=0.5,
        sigma_te_sq=1,
        diagonal=diagonal,
    )
    sampcomp.plot_bounds(
        saveas="bhatta-bound-a{}-s1-d0.1.eps".format(diagonal),
        sigma_te_sq=1,
        diagonal=diagonal,
    )


def script_plot_2():
    sampcomp.plot_bounds(saveas="bhatta-bound-s0-d0.1.eps")
    sampcomp.plot_bounds(
        saveas="bhatta-bound-s0-d0.5.eps", start_delta=0.5
    )
    sampcomp.plot_bounds(
        saveas="bhatta-bound-s1-d0.5.eps", start_delta=0.5, sigma_te_sq=1
    )
    sampcomp.plot_bounds(saveas="bhatta-bound-s1-d0.1.eps", sigma_te_sq=1)
