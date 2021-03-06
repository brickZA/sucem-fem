## Copyright (C) 2011 Stellenbosch University
##
## This file is part of SUCEM.
##
## SUCEM is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## SUCEM is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with SUCEM. If not, see <http://www.gnu.org/licenses/>. 
##
## Contact: cemagga@gmail.com 
# Authors:
# Evan Lezar <mail@evanlezar.com>
# Neilen Marais <nmarais@gmail.com>

""" A collection of linear system solvers """

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from time import time

class SystemSolverBase ( object ):
    """
    A base class for the implementation of various solvers for sparse eigen systems.
    This base class provides logging functionality, but requires an extention to allow for actual solver implementation.
    """
    def __init__ ( self, A, preconditioner_type=None ):
        """
        The constructor for a System Solver
        
        @param A: The matrix for the system that must be solved
        @keyword preconditioner_type: A string indicating the type of preconditioner to be used.
            (default: None)
        """
        self._logging_data = { 'time': [], 'id': [], 'res': [] }
        self._user_callbacks = []
        self._timestamp( 'init' )
        self._A = A
        self.set_preconditioner ( preconditioner_type )
        self._callback_count = 0
    def _call_solver (self):
        """
        A stub routine that is to be implemented by an inherited class depending on the solver being used.
        """
        raise Exception ( "Solver driver not implemented.")

    def add_user_callback(self, cb):
        """
        Add a user callback function cb(self, residual)

        @param cb: User callback function cb(self, residual) where
            self is the current SystemSolver instance and residual is
            the most recently calculated residual.
        """
        self._user_callbacks.append(cb)

    def _callback ( self, xk ):
        """
        Calculate the residual if required, and update the progress of the iterative solver.
        
        @param xk: the solution vector or residual at a step k in the solution process
        """
        self._callback_count += 1;
        if type(xk) is float:
            res = xk
        else:
            res = calculate_residual( self._A, xk, self._b )
        self._timestamp( self._callback_count, res=res )
        
    def _timestamp (self, id, res=np.nan ):
        """
        Add a timestamp/id pair to the logging data used to measure the progress of the solver
        
        @param id: an identifier for the timestamp
        @keyword res: An optional residual parameter to store along with the timestamp info
            (default: NaN)
        """
        self._logging_data['time'].append ( time() )
        self._logging_data['id'].append ( id )
        self._logging_data['res'].append ( res )
        for cb in self._user_callbacks:
            cb(self, res)
    
    def set_b ( self, b ):
        """
        Set the right-hand side vector b in the linear system Ax = b
        
        @param b: the right-hand side vector
        """
        self._b = b

    def set_preconditioner (self, M_type ):
        """
        Set the preconditioner (self._M) used in the solver
        
        @param M_type: A string to specify the preconditioner used
        """
        self._timestamp('preconditioner::start')
        if M_type is None:
            M = None
        elif M_type.lower() == 'diagonal':
            M = scipy.sparse.spdiags(1./self._A.diagonal(), 0, self._A.shape[0], self._A.shape[1])
        elif M_type.lower() == 'ilu':
            self._M_data = scipy.sparse.linalg.spilu (
                self._A.tocsc(), drop_tol=1e-8, fill_factor=1  )
            M = scipy.sparse.linalg.LinearOperator (
                self._A.shape, self._M_data.solve )
        else:
            print "Warning: Preconditioner type '%s' not recognised. Not using preconditioner." % M_type
            M = None 
        
        self._M = M
        self._timestamp('preconditioner::end')
        
        
    def solve ( self, b ):
        """
        Solve the linear system Ax = b for x
        
        @param b: the right-hand side vector
        @return: x, the solution of the linear system Ax = b
        """
        self._timestamp( 'solve::begin' )
        self.set_b(b)
        
        x, info = self._call_solver ()

        if ( info > 0 ):
            "convergence to tolerance not achieved, in %d iterations" % info
        elif ( info < 0 ): 
            "illegal input or breakdown (%d)" % info
            
        self._timestamp( 'solve::end' )
        return x
    
    def get_logging_data (self):
        """Return the timing and residual data for the solver generated by calls to timestamp.
        
        @return: A dicitionary of lists -- { 'time': [], 'id': [], 'res': [] }.
            The items in each list show either the time, identifier, or residual for a particular timestamp.
        """
        return self._logging_data
    
    def plot_convergence (self, x_is_time=False, show_plot=False, label=None, style='-'):
        """Process the logging data and plot the convergence history of the solver.
        
        @keyword x_is_time: The x-asis of the plot must be time and not iterations.
            (default: False)
        @keyword show_plot: Show the plot once the convergence curve has been plotted.
            (default: False)
        @keyword label: The label to associate with the plotted curve in the legend.
            (default: None. The curve is not labelled)
        @keyword style: The style with which to plot the curve.
            (default: '-'. The curve is plotted as a solid line with automatic colouring.)
        """
        import pylab as P
        
        y_data = np.log10( np.array(self._logging_data['res']) )
        y_label = 'Residual [log10]'
        if x_is_time:
            t0 = self._logging_data['time'][0]
            x_data = np.array(self._logging_data['time'], dtype=np.float64) - t0
            x_label = 'Time [s]'
        else:
            index = np.where(np.isfinite(y_data))[0]
            y_data = y_data[index]
            x_data = np.zeros ( index.shape )
            for i in range(len(index)):
                x_data[i] = self._logging_data['id'][index[i]]
            x_label = 'Iterations'
            
        P.plot ( x_data, y_data, style, label=label )
        
        P.xlabel ( x_label )
        P.ylabel ( y_label )
        
        P.grid ( True )
        if show_plot:
            P.legend(loc='upper right')
            P.show()
    
    
    def _get_time_for_id (self, id ):
        """Read the logging data and return the timestamp associated with a specific id
        
        @param id: The id for which the timestamp must be retrieved.
        @return: The timestamp (in seconds) for the specified id.
        """
        return self._logging_data['time'][self._logging_data['id'].index(id)]
    
    def _get_elapsed_time ( self, id0, id1, force_total=False ):
        """Read the logging data and return the time elapsed between two ids.
        
        @param id0: The id of the start of the time interval.
        @param id1: The id of the end of the time interval.
        @keyword force_total: Ignore the values of id0 and id1, and return the total elapsed time for the solver.
        
        @return: The time (in seconds) elapsed between id0 and id1.
        """
        if force_total:
            return self._logging_data['time'][-1] - self._logging_data['time'][0]
         
        return self._get_time_for_id(id1) - self._get_time_for_id(id0)
        
    def print_logging_data ( self ):
        """Print the raw logging data.
        """
        for k in self._logging_data:
            print k
            print self._logging_data[k]
    
    def print_timing_info ( self ):
        """Process the logging data and print timing information associated with the solver.
        """
        solve_time = self._get_elapsed_time('solve::begin', 'solve::end')
        print 'solve time:', solve_time
        preconditioner_time = self._get_elapsed_time('preconditioner::start', 'preconditioner::end')
        print 'preconditioner time:', preconditioner_time
        
        print 'total time:', self._get_elapsed_time(None, None, True)
        
        
        

class BiCGStabSolver ( SystemSolverBase ):
    """
    An iterative Stabilised BICG solver using scipy.
    """
    def _call_solver (self):
        """Solves the linear system (self._A)x = self._b and returns the solution vector.
        
        @return: The solution to the linear system.
        """
        return scipy.sparse.linalg.bicgstab(self._A, self._b, M=self._M, callback=self._callback )

class GMRESSolver ( SystemSolverBase ):
    """
    An iterative GMRES solver using scipy.
    """
    def _call_solver (self):
        """Solves the linear system (self._A)x = self._b and returns the solution vector.
        
        @return: The solution to the linear system.
        """
        return scipy.sparse.linalg.gmres(self._A, self._b, M=self._M, callback=self._callback )

class UMFPACKSolver ( SystemSolverBase ):
    """
    A direct UMFPACK-based solver provided by scipy
    """
    def _call_solver (self):
        """Solves the linear system (self._A)x = self._b and returns the solution vector.
        
        @return: The solution to the linear system.
        """
        scipy.sparse.linalg.use_solver(useUmfpack=True)
        return scipy.sparse.linalg.spsolve ( self._A, self._b ), 0
    
    def plot_convergence (self, x_is_time=False, show_plot=False, label=None, style='-'):
        """Output a string indicating that no convergence history is available.
        """
        print "Direct solver has no convergence history"

class PyAMGSolver ( SystemSolverBase ):
    """
    A PyAMG-based iterative solver.
    """
    def _call_solver (self):
        """Solves the linear system (self._A)x = self._b and returns the solution vector.
        
        @return: The solution to the linear system.
        """
        import pyamg
        x = pyamg.solve ( self._A, self._b, verb=True, tol=1e-8, maxiter=800 )
        return x, 0
    
    def plot_convergence (self, x_is_time=False, show_plot=False, label=None, style='-'):
        """Output a string indicating that no convergence history is available.
        """
        print "PAMG solver convergence display is not yet implemented"

def calculate_residual ( A, x, b ):
    """
    Calculate the residual of the system Ax = b
    
    @param A: a matrix
    @param x: a vector
    @param b: a vector 
    """
    return np.linalg.norm( A*x - b.reshape(x.shape) )


def solve_sparse_system ( A, b, preconditioner_type='ilu' ):
    """
    This function solves the sparse linear system Ax = b for A a scipy sparse matrix, and b a numpy or scipy array
    
    By default an incomplete LU preconditioner is used with the bicgstab iterative solver
    
    @param A: a square matrix 
    @param b: the RHS vector
    @param preconditioner_type: Preconditioner type string
    """
    
    solver = BiCGStabSolver ( A, preconditioner_type )
    x = solver.solve(b)
    return x
    
