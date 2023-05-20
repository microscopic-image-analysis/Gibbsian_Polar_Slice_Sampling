"""
Provides various MCMC-related plotting functions with many options and 
reasonable defaults wherever possible

Available functions:
    plot_2d_samples
    plot_3d_samples
    plot_samples
    contour_precalc
    plot_contour_and_samples
    trace_plot
    trace_plot_row
    trace_plot_radii
    trace_plot_log_den_vals
    plot_acf
    plot_acfs
    plot_accuracies
    dim_dep_plot
    plot_step_hist
    plot_step_hist_row
    plot_data_and_dec_bnds
    plot_trace_and_acf
    plot_trace_and_step_hists
    plot_traces_2_col
"""

import numpy as np
import numpy.linalg as alg
import matplotlib.pyplot as plt
import mcmc_utils as uti

################################################################################
###################### Auxiliary Variables and Functions #######################

markers = ["o", "s", "^", "v", "P", "d"]
nmarkers = len(markers)

def initiate(figsize, dpi, title=None):
    """Auxiliary function, not to be called by the user"""

    plt.figure(figsize=figsize, dpi=dpi)
    if title != None:
        plt.title(title)

def initiate_overview(figsize, dpi, nsub):
    """Auxiliary function, not to be called by the user"""

    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    subfigs = fig.subfigures(nrows=nsub, ncols=1)
    return subfigs

def wrapup(filepath=None):
    """Auxiliary function, not to be called by the user"""

    plt.tight_layout()
    if filepath != None:
        plt.savefig(filepath)
    plt.show()

def wrapup_overview(filepath=None):
    """Auxiliary function, not to be called by the user"""

    if filepath != None:
        plt.savefig(filepath)
    plt.show()

def size_gen(size, n):
    """Auxiliary function, not to be called by the user"""

    if size != None:
        return size
    return np.min([1e3/n, 1.0])

def bin_gen(nbins, nvals):
    """Auxiliary function, not to be called by the user"""

    if nbins != None:
        return nbins
    return np.max([nvals//100, 1])

################################################################################
################################ User Functions ################################

################ Plotting of 2d and 3d Samples and 2d Contours #################

def plot_2d_samples(
        samples, 
        figsize=(5,5),
        dpi=100,
        title=None,
        filepath=None,
        size=None
    ):
    """Creates a simple scatter plot of 2-dimensional samples

        Args:
            samples: np array of shape (nsamples,2)
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title, by default there is none
            filepath: location to save plot to, leave default to not save it
            size: marker size to be used by scatter(), by default the
                size used is 1e3/nsamples
    """

    initiate(figsize, dpi, title)
    size = size_gen(size, samples.shape[0])
    plt.scatter(samples[:,0], samples[:,1], s=size)
    wrapup(filepath)

def plot_3d_samples(
        samples,
        figsize=(5,5),
        dpi=100,
        title=None,
        filepath=None,
        size=None
    ):
    """Creates a simple scatter plot of 3-dimensional samples

        Args:
            samples: np array of shape (nsamples,3)
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title, by default there is none
            filepath: location to save plot to, leave default to not save it
            size: marker size to be used by scatter(), by default the
                size used is 1e3/nsamples
    """

    fig = plt.figure(figsize=figsize, dpi=dpi)
    size = size_gen(size, samples.shape[0])
    ax = fig.add_subplot(projection="3d")
    if title != None:
        ax.set_title(title)
    ax.scatter(samples[:,0], samples[:,1], samples[:,2], s=size)
    wrapup(filepath)

def plot_samples(
        samples,
        figsize=(5,5),
        dpi=100,
        title=None,
        filepath=None,
        size=None
    ):
    """Creates a simple scatter plot of 2- or 3-dimensional samples, simply
        terminates if sample dimension is not 2 or 3

        Args:
            samples: np array of shape (nsamples,d) with d in [2,3]
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title, by default there is none
            filepath: location to save plot to, leave default to not save it
            size: marker size to be used by scatter(), by default the
                size used is 1e3/nsamples
    """

    d = samples.shape[1]
    if d == 2:
        plot_2d_samples(samples, figsize, dpi, title, filepath, size)
    elif d == 3:
        plot_3d_samples(samples, figsize, dpi, title, filepath, size)

def contour_precalc(
        reso,
        x1min,
        x1max,
        x2min,
        x2max,
        fct
    ):
    """Computes quantities required for contour and 3d plots

        Args:
            reso: resolution (total number of grid points) in each
                coordinate direction
            x1min: minimal value along x1-axis
            x1max: maximal value along x1-axis
            x2min: minimal value along x2-axis
            x2max: maximal value along x2-axis
            fct: function to be evaluated, should take 1d np arrays
                of length 2 as input
        Returns:
            G1, G2, vals: arguments for plt.contour() etc.
    """

    x1s = np.linspace(x1min, x1max, reso)
    x2s = np.linspace(x2min, x2max, reso)
    G1, G2 = np.meshgrid(x1s, x2s)
    X = np.concatenate([G1.reshape(reso,reso,1),G2.reshape(reso,reso,1)],axis=2)
    vals = np.zeros(G1.shape)
    for i in range(reso):
        vals[i] = np.array(list(map(fct, X[i])))
    return G1, G2, vals

def plot_contour_and_samples(
        G1,
        G2,
        vals,
        samples,
        figsize=(5,5),
        dpi=100,
        title=None,
        filepath=None,
        size=None,
        levels=8,
        filled=False
    ):
    """Plots contours of a bivariate target density and given (typically 
        approximate) samples from it in the same figure

        Args:
            G1, G2: grid values for contour plot, like produced by np.meshgrid
            vals: function values of the function to be contour plotted at the 
                grid locations given by G1, G2
            samples: np array of shape (nsamples,2)
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title, by default there is none
            filepath: location to save plot to, leave default to not save it
            size: marker size to be used by plt.scatter, by default the size 
                used is 1e3/nsamples
            levels: number of levels or actual levels for contour plot
            filled: whether space between contour lines should be filled
                with (approximately) continuous contours
    """

    initiate(figsize, dpi, title)
    contfct = plt.contourf if filled else plt.contour
    contfct(G1, G2, vals, levels)
    size = size_gen(size, samples.shape[0])
    plt.scatter(samples[:,0], samples[:,1], s=size, color="red")
    wrapup(filepath)

################################# Trace Plots ##################################

def trace_plot(
        vals,
        figsize=(5,2.5),
        dpi=100,
        title=None,
        filepath=None,
        linewidth=None
    ):
    """Creates a trace plot of the given values

        Args:
            vals: values to be plotted, should be 1d np array
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title, by default there is none
            filepath: location to save plot to, leave default to not save it
            linewidth: linewidth to be used by plt.plot(), by default the width
                used is 1e3/len(vals)
    """

    initiate(figsize, dpi, title)
    nvals = vals.shape[0]
    linewidth = size_gen(linewidth, nvals)
    plt.plot(range(nvals), vals, linewidth=linewidth)
    wrapup(filepath)

def trace_plot_row(
        vals,
        snames,
        spsize=(4,2),
        dpi=200,
        title=None,
        filepath=None,
        linewidth=None
    ):
    """Creates a row of trace plots of given values

        Args:
            vals: values to be plotted, should be 1d np array
            snames: names of the samplers used to be printed in legend
            spsize: 2-tuple giving the size of each subplot
            dpi: dots per inch used for plotting
            title: the figure title, by default there is none
            filepath: location to save plot to, leave default to not save it
            linewidth: linewidth to be used by plt.plot(), by default the width
                used is 1e3/len(vals)
    """

    nsam = len(snames)
    figsize = (nsam * spsize[0], spsize[1])
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    axes = fig.subplots(nrows=1, ncols=nsam)
    if title != None:
        plt.title(title)
    linewidth = size_gen(linewidth, vals[0].shape[0])
    for i in range(nsam):
        axes[i].set_title(snames[i])
        axes[i].plot(vals[i], linewidth=linewidth)
    wrapup_overview(filepath)

def trace_plot_radii(
        radii,
        figsize=(5,2.5),
        dpi=100,
        filepath=None,
        linewidth=None
    ):
    """Wrapper for trace_plot with the title set to 'Chain Iterate Radii' """        

    trace_plot(radii, figsize, dpi, "Chain Iterate Radii", filepath, linewidth)

def trace_plot_log_den_vals(
        samples,
        log_density,
        figsize=(5,2.5),
        dpi=100,
        filepath=None,
        linewidth=None
    ):
    """Wrapper for trace_plot that plots the values of a given log density at
        each of a given set sequence of samples and sets the title to 'Log
        Density Values'
    
        New Args:
            samples: approximate samples from the log density that should be
                plotted, must be np array of shape (nsamples, d)
            log_density: function taking individual samples as inputs and 
                returning float values
    """

    vals = np.array(list(map(log_density, samples)))
    trace_plot(vals, figsize, dpi, "Log Density Values", filepath, linewidth)

############################ Statistical Quantities ############################

def plot_acf(
        vals,
        figsize=(5,2.5),
        dpi=100,
        title=None,
        filepath=None,
        maxl=50
    ):
    """Computes and plots (empirical) autocorrelation function (acf) of the 
        given values

        Args:
            vals: values whose acf is to be plotted, should be 1d np array
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title, by default there is none
            filepath: location to save plot to, leave default to not save it
            maxl: maximum lag for which acf is to be plotted
    """

    acf_vals = uti.acf(vals, maxl)
    initiate(figsize, dpi, title)
    plt.ylim(-0.1,1.1)
    plt.plot(range(maxl+1), acf_vals)
    wrapup(filepath)

def plot_acfs(
        vals,
        snames,
        figsize=(5,2.5),
        dpi=100,
        title=None,
        filepath=None,
        maxl=1000,
        xscale="linear",
    ):
    """Computes and plots (empirical) autocorrelation function (acf) of the 
        given values generated by different samplers and plots their acfs all
        in the same plot 

        Args:
            vals: values whose acf is to be plotted, should be a list of 1d np 
                arrays
            snames: names of the samplers that generated the values
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title, by default there is none
            filepath: location to save plot to, leave default to not save it
            maxl: maximum lag for which acf is to be plotted
            xscale: type of scale to be used for x-axis, e.g. "linear" or "log"
    """

    acf_vals = [uti.acf(vls, maxl) for vls in vals]
    default_cycler = plt.rcParams["axes.prop_cycle"]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(snames)))
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors)
    initiate(figsize, dpi, title)
    plt.xscale(xscale)
    plt.ylim(-0.1,1.1)
    for avs in acf_vals:
        plt.plot(range(maxl+1), avs)
    plt.legend(snames, loc="upper right")
    wrapup(filepath)
    plt.rcParams["axes.prop_cycle"] = default_cycler

def plot_accuracies(
        a_tr,
        b_tr,
        a_te,
        b_te,
        samples,
        pred, 
        figsize=(5,2.5),
        dpi=100,
        title="Progression of Accuracies over Chain Iterations",
        filepath=None,
        linewidth=1
    ):
    """Plots progression of accuracies when using a single sample as estimator 
        for the hidden variables over the course of the generated chain of
        samples
        
        Args:
            a_tr: feature matrix of training data
            b_tr: labels of training data
            a_te: feature matrix of test data
            b_te: labels of test data
            samples: list of samples to be used as estimators
            pred: prediction function, pred(a, x) should yield predicted labels
                for feature matrix a when using x as the hidden variables
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title
            filepath: location to save plot to, leave default to not save it
            linewidth: linewidth to be used by plt.plot()
    """

    inds = range(samples.shape[0])
    tr_acc = lambda sam: np.mean(pred(a_tr, sam) == b_tr)
    te_acc = lambda sam: np.mean(pred(a_te, sam) == b_te)
    tr_accs = np.array(list(map(tr_acc, samples)))
    te_accs = np.array(list(map(te_acc, samples)))
    initiate(figsize, dpi, title)
    plt.plot(inds, tr_accs, linewidth=linewidth)
    plt.plot(inds, te_accs, linewidth=linewidth)
    plt.legend(["train accuracy", "test accuracy"], loc="upper left")
    wrapup(filepath)

def dim_dep_plot(
        ds,
        qs,
        qname,
        snames,
        figsize=(4.5,3),
        dpi=100,
        title=None,
        filepath=None,
        xscale="linear",
        yscale="linear",
        ylim=None
    ):
    """Plots dimension dependent quantities for a set of samplers and dimensions
    
        Args:
            ds: list of dimensions
            qs: matrix of quantities to be plotted, should be 2d np array of 
                shape (nsamplers, len(ds))
            qname: name of the quantity to be plotted (used as ylabel)
            snames: names of the samplers used to be printed in legend
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title
            filepath: location to save plot to, leave default to not save it
            xscale: type of scale to be used for x-axis, e.g. "linear" or "log"
            yscale: type of scale to be used for y-axis, e.g. "linear" or "log"
            ylim: values to be used for plt.ylim, leave None to let matplotlib
                decide this
    """

    default_cycler = plt.rcParams["axes.prop_cycle"]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(snames)))
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors)
    initiate(figsize, dpi, title)
    # don't rearrange, plt.xticks this must come after plt.xscale!
    plt.xscale(xscale)
    plt.yscale(yscale)
    if len(ds) <= 10 and xscale == "linear":
        plt.xticks(ds)
    plt.xlabel("d")
    plt.ylabel(qname)
    if ylim != None:
        plt.ylim(ylim)
    for i, sam_qs in enumerate(qs):
        plt.plot(ds, sam_qs, linestyle="dashed", marker=markers[i % nmarkers])
    plt.legend(snames, loc="upper left")
    wrapup(filepath)
    plt.rcParams["axes.prop_cycle"] = default_cycler

################################ Miscellaneous #################################

def plot_step_hist(
        samples,
        figsize=(5,2.5),
        dpi=100,
        title=None,
        filepath=None,
        nbins=None
    ):
    """Plots a histogram of the distances between consecutive samples

        Args:
            samples: np array of shape (nsamples,d) containing the samples
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title
            filepath: location to save plot to, leave default to not save it
            nbins: number of bins to be used, by default the number will be set
                so that the average bin contains 100 elements
    """

    steps = alg.norm(samples[1:] - samples[:-1], axis=1)
    nbins = bin_gen(nbins, steps.shape[0])
    initiate(figsize, dpi, title)
    plt.hist(steps, bins=nbins)
    wrapup(filepath)

def plot_step_hist_row(
        steps,
        snames,
        spsize=(4,2),
        dpi=200,
        title=None,
        filepath=None,
        nbins=None
    ):
    """Creates a row of step histogram plots corresponding to given step sizes

        Args:
            steps: list of arrays of step sizes for each sampler
            snames: names of the samplers used to be printed in legend
            spsize: 2-tuple giving the size of each subplot
            dpi: dots per inch used for plotting
            title: the figure title, by default there is none
            filepath: location to save plot to, leave default to not save it
            nbins: number of bins to be used, by default the number will be set
                so that the average bin contains 100 elements
    """

    nsam = len(snames)
    figsize = (nsam * spsize[0], spsize[1])
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    axes = fig.subplots(nrows=1, ncols=nsam)
    if title != None:
        plt.title(title)
    maxsteps = np.quantile(steps, 0.999)
    nbins = bin_gen(nbins, steps[0].shape[0])
    for i in range(nsam):
        axes[i].set_title(snames[i])
        axes[i].hist(steps[i], bins=nbins, range=(0,maxsteps))
    wrapup_overview(filepath)

def plot_data_and_dec_bnds(
        a,
        b,
        dec_bnd,
        x_comp,
        x_samples,
        figsize=(5,5),
        dpi=100,
        title=None,
        filepath=None,
        size=None,
        alpha=0.1,
        buf=0.1
    ):
    """Plot 2d data for binary classification as scatter plot, decision
        boundaries corresponding to samples as dashed line plots and a composite
        of samples as regular line plot
        
        Args:
            a: feature matrix of the data, should be np array of shape 
                (ndatapts, 2)
            b: label vector of data, should be 1d np array of size ndatapts
            dec_bnd: function taking hidden variables and the first feature as 
                input and returning the second feature for which the resulting
                feature vector lies exactly on the decision boundary 
                corresponding to the given choice of the hidden variables 
            x_comp: composite (e.g. mean) of samples that is to be used as the
                overall estimator of the hidden variables, should be 1d np array
                of size 3
            x_samples: list of estimators of the hidden variables that are also
                to be displayed (but less prominently), should be np array of
                shape (nsamples, 3)
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title
            filepath: location to save plot to, leave default to not save it
            size: marker size to be used by plt.scatter, by default the size 
                used is 1e3/ndatapts
            alpha: opacity of the sampled decision boundaries
            buf: size buffer around the data points that is still to be plotted
   """

    initiate(figsize, dpi, title)
    # plot data
    size = size_gen(size, a.shape[0])
    plt.scatter(a[b==1][:,0], a[b==1][:,1], s=size)
    plt.scatter(a[b!=1][:,0], a[b!=1][:,1], s=size)
    # plot decision boundaries
    a0s = np.linspace(np.min(a[:,0])-buf, np.max(a[:,0])+buf, 1000)
    a1s = np.array([dec_bnd(x_comp, a0) for a0 in a0s])
    plt.plot(a0s, a1s, color="black")
    for x in x_samples:
        a1s = np.array([dec_bnd(x, a0) for a0 in a0s])
        plt.plot(a0s,a1s, color="k", linestyle="dashed", alpha=alpha, zorder=-2)
    plt.xlim(a0s[0], a0s[-1])
    plt.ylim(np.min(a[:,1])-buf, np.max(a[:,1])+buf)
    wrapup(filepath)

################################ Overview Plots ################################

def plot_trace_and_acf(
        vals,
        snames,
        figsize=(10,10),
        dpi=100,
        title=None,
        filepath=None,
        linewidth=None,
        maxl=50,
    ):
    """Creates a plot that contains (nsam,2) subplots, with each row
        containing a trace plot and the corresponding acf plot for one algorithm

        Args:
            vals: list of size nalg containing the values to be plotted for 
                each sampler
            snames: list of length nsam containing names of the samplers used 
                to be printed as row titles
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title
            filepath: location to save plot to, leave default to not save it
            linewidth: linewidth to be used in trace plot, by default the width
                used in row i is 1e3/len(vals[i])
            maxl: maximum lag to be plotted in acf plot
    """

    nsam = len(snames)
    subfigs = initiate_overview(figsize, dpi, nsam)
    for i, subfig in enumerate(subfigs):
        subfig.suptitle(snames[i])
        axes = subfig.subplots(nrows=1, ncols=2)
        # left column: trace plot
        linewidth = size_gen(linewidth, vals[i].shape[0])
        axes[0].plot(vals[i], linewidth=linewidth)
        # right column: acf line plot
        acf_vals = uti.acf(vals[i], maxl)
        axes[1].set_ylim(-0.1,1.1)
        axes[1].plot(range(maxl+1), acf_vals)
        axes[1].yaxis.tick_right()
    wrapup_overview(filepath)

def plot_trace_and_step_hists(
        vals,
        steps,
        snames,
        figsize=(10,10),
        dpi=100,
        title=None,
        filepath=None,
        linewidth=None,
        nbins=None
    ):
    """Creates a plot that contains (nsam,2) subplots, with each row
        containing a trace plot and a step size histogram for one algorithm

        Args:
            vals: list of size nsam containing the 1d quantities for each 
                sampler that should be trace plotted
            steps: list of size nsam containing the step sizes for each sampler
            snames: list of length nsam containing names of the samplers used 
                to be printed as row titles
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            title: the figure title
            filepath: location to save plot to, leave default to not save it
            linewidth: linewidth to be used in trace plot, by default the width
                used in row i is min(1e3/vals[i].shape[0], 1)
            nbins: number of bins to be used, by default the number will be set
                so that the average bin contains 100 elements
    """

    nsam = len(snames)
    max_step = np.max([np.max(stps) for stps in steps])
    subfigs = initiate_overview(figsize, dpi, nsam)
    for i, subfig in enumerate(subfigs):
        subfig.suptitle(snames[i])
        axes = subfig.subplots(nrows=1, ncols=2)
        # left column: trace plot
        linewidth = size_gen(linewidth, vals[i].shape[0])
        axes[0].plot(vals[i], linewidth=linewidth)
        # right column: step size histogram
        nbins = bin_gen(nbins, steps[i].shape[0])
        axes[1].hist(steps[i], bins=nbins, range=(0,max_step))
        axes[1].set_xlim(0,max_step)
        axes[1].yaxis.tick_right()
    wrapup_overview(filepath)

def plot_traces_2_col(
        vals1,
        vals2,
        snames,
        figsize=(10,10),
        dpi=100,
        filepath=None,
        lw1=None,
        lw2=None
    ):
    """Creates a plot that contains (nsam,2) subplots, with each row
        containing two trace plot for one algorithm

        Args:
            vals1: list of size nalg containing 1d np arrays to be trace-plotted
            vals2: list of size nalg containing 1d np arrays to be trace-plotted
            snames: list of length nsam containing names of the samplers used 
                to be printed as row titles
            figsize: 2-tuple giving the figure's size
            dpi: dots per inch used for plotting
            filepath: location to save plot to, leave default to not save it
            lw1: linewidth to be used in left column of trace plots, by default 
                the width used in row i is min(1e3/vals[i].shape[0], 1)
            lw2: linewidth to be used in right column of trace plots, by default
                the width used in row i is min(1e3/vals[i].shape[0], 1)
    """

    subfigs = initiate_overview(figsize, dpi, len(vals1))
    for i, subfig in enumerate(subfigs):
        subfig.suptitle(snames[i])
        axes = subfig.subplots(nrows=1, ncols=2)
        axes[0].plot(vals1[i], linewidth=lw1)
        axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0,2))
        axes[1].plot(vals2[i], linewidth=lw2)
        axes[1].yaxis.tick_right()
    wrapup_overview(filepath)

