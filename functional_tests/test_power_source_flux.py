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
# Neilen Marais <nmarais@gmail.com>
"""
Compare the source energy with the total energy flux out of the
problem domain
"""
from __future__ import division

import unittest
import os
import pickle
import numpy as np
import dolfin
import sucemfem
import sucemfem.BoundaryConditions
import sucemfem.ProblemConfigurations.EMDrivenProblem
import sucemfem.Sources
import sucemfem.Utilities.LinalgSolvers
from sucemfem.Testing import Paths
from sucemfem.Consts import c0
from sucemfem.Utilities.MeshGenerators import get_centred_cube
from sucemfem.PostProcessing import power_flux

class CalcFillament(object):
    def __init__(self):
        ### Problem parameters
        self.freq =  1.0e+9                          # Frequency
        self.lam = c0/self.freq
        self.l = self.lam/10                            # Dipole length
        self.I = 1.0                                 # Dipole current
        self.source_direction = np.array([0,0,1.])    # Source orientation
        self.source_centre = np.array([0,0,0.])        # Position of the source
        self.source_endpoints =  np.array(
            [-self.source_direction*self.l/2, self.source_direction*self.l/2]) \
            + self.source_centre
        ### Discretisation settings
        self.order = 2
        self.domain_size = np.array([self.lam]*3)*0.5
        self.max_edge_len = self.lam/6
        self.mesh = get_centred_cube(self.domain_size, self.max_edge_len)
        ## Set up materials function with all free-space
        material_mesh_func = dolfin.MeshFunction('uint', self.mesh, 3)
        material_mesh_func.set_all(0)
        materials = {0:dict(eps_r=1, mu_r=1),}
        ## Set up 1st-order analytical ABC
        abc = sucemfem.BoundaryConditions.ABC.ABCBoundaryCondition()
        abc.set_region_number(1)
        bcs = sucemfem.BoundaryConditions.container.BoundaryConditions()
        bcs.add_boundary_condition(abc)
        ## Set up high level problem class
        dp = sucemfem.ProblemConfigurations.EMDrivenProblem.DrivenProblemABC()
        dp.set_mesh(self.mesh)
        dp.set_basis_order(self.order)
        dp.set_material_regions(materials)
        dp.set_region_meshfunction(material_mesh_func)
        dp.set_boundary_conditions(bcs)
        ## Set up current fillament source
        current_sources = sucemfem.Sources.current_source.CurrentSources()
        fillament_source = sucemfem.Sources.fillament_current_source.FillamentCurrentSource()
        fillament_source.set_source_endpoints(self.source_endpoints)
        fillament_source.set_value(self.I)
        current_sources.add_source(fillament_source)
        ## Set source in problem container
        dp.set_sources(current_sources)
        self.problem = dp
        self.function_space = dp.function_space

    def solve(self):
        dp = self.problem
        dp.init_problem()
        dp.set_frequency(self.freq)
        ## Get sytem LHS matrix and RHS Vector
        A = dp.get_LHS_matrix()
        b = dp.get_RHS()
        ## Solve. Choose spare solver if UMFPack runsout of memory
        # print 'solve using scipy bicgstab'
        # x = solve_sparse_system ( A, b, preconditioner_type='diagonal')
        # x = solve_sparse_system ( A, b )
        print 'solve using UMFPack'
        umf_solver = sucemfem.Utilities.LinalgSolvers.UMFPACKSolver(A)
        return umf_solver.solve(b)




class test_PowerComparison(unittest.TestCase):
    """Compare powers calculated using source voltage, surface flux
    integral and variational flux integral"""
    def setUp(self):
        self.solver = CalcFillament()
        self.mesh = self.solver.mesh
        self.discretisation_order = self.solver.order
        self.function_space = self.solver.function_space
        self.freq = self.solver.freq
        self.k0 = 2*np.pi*self.freq/c0
        self.source_endpoints = self.solver.source_endpoints
        self.I = self.solver.I

    def test_calc_flux(self):
        surf_flux = power_flux.SurfaceFlux(self.function_space)
        surf_flux.set_k0(self.k0)
        var_flux = power_flux.VariationalSurfaceFlux(self.function_space)
        var_flux.set_k0(self.k0)
        x = self.solver.solve()
        surf_flux.set_dofs(x)
        var_flux.set_dofs(x)
        calc_volts = sucemfem.Sources.PostProcess.ComplexVoltageAlongLine(
            self.function_space)
        calc_volts.set_dofs(x)
        volts = -calc_volts.calculate_voltage(*self.source_endpoints)
        surf_power = np.real(surf_flux.calc_flux())
        var_power = np.real(var_flux.calc_flux())
        volt_power = np.real(volts*self.I)
        # Check that the calculations are within 1% of each other
        self.assertTrue(100*np.abs(surf_power - volt_power)/volt_power < 1)
        self.assertTrue(100*np.abs(var_power - volt_power)/volt_power < 1)
        self.assertTrue(100*np.abs(surf_power - var_power)/var_power < 1)

