import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from typing import Any
from typing import cast, Union
from typing import Optional  # telling the type checker that either an object of the specific type is required, or None is required
import time

from collections import namedtuple
from scipy.stats import page_trend_test as ptest # "results\n L=%.2f, p-value=%.5f, method=%s" % (r.statistic, r.pvalue, r.method)

class PageTest:
    """Allows Page Trend Test to be performed to compare convegence of two EA's fitnesses.
        - This allows algorithms with different x-axis values to be compared via projection 
        onto a unified x-axis.
        - Algorithm results are added problem by problem. 

    Args:

        num_cuts:
            The number of cuts. Typically half the number of rows/problems considered.
        
        max_x:
            Select the maximum x-axis value to go up to.
        
        invert:
            Binary integer value. Rather than conidering algorith results A-B, consider B-A.

    """

    

    def __init__(
                self,
                num_cuts: int,
                max_x: Optional[float] = None,
                invert: int = 0,
                problem_labels: Optional[list] = None,
                ):
        
        # # Set number of cuts to half the number of problems
        self.num_cuts = num_cuts

        assert invert == 0 or invert == 1, "Invert must be binary integer"
        self.invert = invert
        self.max_x = max_x
        self.problem_labels = problem_labels

        self.results_so_far = []


        return

    @property
    def matrix(self) -> np.ndarray:
        """Fetch the matric of cuts of difference between algorithm convergences"""
        assert len(self.results_so_far) > 0, "Need to add some problem results"
        assert len(self.results_so_far) > self.num_cuts, "Ensure you have considered more problems than the number of cuts"
        return [c.cuts_y for c in self.results_so_far]


    def plot_cuts(self, save: Optional[str] = None, show: int=1):
        """Plot the problem convergence and cuts for the algos added so far"""

        y = [c.y for c in self.results_so_far]
        x = [c.cx for c in self.results_so_far]  # relative x axis in terms of cuts
        cx = [c.cuts_x for c in self.results_so_far]
        cy = self.matrix

        y_flat = np.array(y).flatten()
        ymin = np.min(y_flat)
        ymax = np.max(y_flat)


        fig, ax = plt.subplots()

        for f, arr in enumerate(y):
            
            max = np.max(abs(arr))

            if self.problem_labels is None:
                ax.plot(x[f], y[f]/max, label='f%d' % (f), alpha=1)
            else:
                ax.plot(x[f], y[f]/max, label='f%d (%s)' % (f, self.problem_labels[f]), alpha=1)


            ax.plot(cy[f]/max,'x', color='k', alpha=0.75, markersize=5)

            for xx in range(self.num_cuts):
                # ax.plot([xx,xx],[ymin,ymax], '--', color='k', alpha=0.15)  # plot line
                ax.plot([xx,xx],[-1,1], '--', color='k', alpha=0.15)  # plot line

        ax.set_xlabel('cuts')
        if self.invert == 0:
            ax.set_ylabel('normalised fitness{A-B}')
        elif self.invert == 1:
            ax.set_ylabel('normalised fitness{B-A}')


        handles, labels = ax.get_legend_handles_labels()
        handles.append(mlines.Line2D([], [], color='k', marker='x', markersize=5, label='cuts', linestyle=''))
        ax.legend(handles=handles)


        if show == 1:
            plt.show()

        if save is not None:
            fig.savefig(save, dpi=200)

        plt.close(fig)
        return 

    def test(self):
        """Perform Page Trend Test on the problems added so far"""
        r = ptest(self.matrix)
        print("Page Test (%d problems, %d cuts) results: L=%.2f, p-value=%.5f, method=%s" % (len(self.matrix), self.num_cuts, r.statistic, r.pvalue, r.method))
        print(" > We assume that if p<0.05 algorithm A converges faster than B, or if p>0.95 algorithm B converges faster than A, otherwise we cannot say anything.")
        return


    def add_problem_interp(self, xA: np.ndarray, yA: np.ndarray, xB: np.ndarray, yB: np.ndarray):
        """ Adds the results from two algorithms.
        - Takes the two evolutionary algorithm curves and creates two new
        interpolated arrays with a unified x axis value.
        - These are the "c cut points" used to subtract the results.
        and formulate the A-B trend.
        """

        # # Ensure the interpolation gives values close to what we are looking for
        l = max(len(xA), len(xB))
        nps = (int(l/self.num_cuts)+1)*self.num_cuts*10
        # print("l:", l, ", nps:", nps)

        # # Make sure the max value is the same between al algo piece wise combos
        if self.max_x is None:
            max_x = max([max(xB), max(xA)])
            # max_x = np.around(max_x, decimals=0)
        else:
            max_x = self.max_x

        # # Interpolate to a larger number of points
        new_x = np.linspace(0, max_x, num=nps)
        new_yA = np.interp(new_x, xA, yA)
        new_yB = np.interp(new_x, xB, yB)

        # # Cut down to the newer number of points
        cuts = []
        real_c_locs = []
        c_locs = np.linspace(0, max_x, num=self.num_cuts)
        for c in c_locs:
            idx, val = self.find_nearest(new_x, c)
            if self.invert == 0:
                cuts.append(new_yA[idx]-new_yB[idx])
            elif self.invert == 1:
                cuts.append(new_yB[idx]-new_yA[idx])
            real_c_locs.append(val)

        # # Apply inversion to y values
        if self.invert == 0:
            y = new_yA-new_yB
        elif self.invert == 1:
            y = new_yB-new_yA

        # # 
        rel_x = (new_x/np.max(new_x))*(self.num_cuts-1)


        DataGroup = namedtuple('PageCuts', ['x', 'y', 'cuts_x', 'cuts_y', 'cx'])
        results = DataGroup(new_x, y, c_locs, cuts, rel_x)

        self.results_so_far.append(results)

        return 
    

    def add_problem(self, x: np.ndarray, yA: np.ndarray, yB: np.ndarray):
        """ Adds the results from two algorithms.
        - Takes the two evolutionary algorithm curves.
        - These are the "c cut points" used to subtract the results
        and formulate the A-B trend.
        """

        # # Make sure the max value is the same between al algo piece wise combos
        if self.max_x is None:
            max_x = max(x)
            # max_x = np.around(max_x, decimals=0)
        else:
            max_x = self.max_x

        # # Cut down to the newer number of points
        cuts = []
        real_c_locs = []
        c_locs = np.linspace(0, max_x, num=self.num_cuts)
        for c in c_locs:
            idx, val = self.find_nearest(x,c)

            if self.invert == 0:
                cuts.append(yA[idx]-yB[idx])
            elif self.invert == 1:
                cuts.append(yB[idx]-yA[idx])
            real_c_locs.append(val)

        # # Apply inversion to y values
        if self.invert == 0:
            y = yA-yB
        elif self.invert == 1:
            y = yB-yA

        # # cuts on OG x axis
        rel_x = (x/np.max(x))*(self.num_cuts-1)


        DataGroup = namedtuple('PageCuts', ['x', 'y', 'cuts_x', 'cuts_y', 'cx'])
        results = DataGroup(x, y, c_locs, cuts, rel_x)

        self.results_so_far.append(results)

        return 

    def find_nearest(self, array, location):
        """Find the index & value of the point in an array clossest to the desired cut location """
        array = np.asarray(array)
        idx = (np.abs(array - location)).argmin()
        return idx, array[idx]                      