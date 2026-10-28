#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/point.h>
#include <deal.II/base/types.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/matrix_tools.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <cmath>

int main() {
  // Physical parameters
  const double k = 52.0;          // Thermal conductivity [W/(m·°C)]
  const double h = 750.0;         // Heat transfer coefficient [W/(m²·°C)]
  const double T_amb = 0.0;       // Ambient temperature [°C]
  const double T_dirichlet = 100.0; // Bottom boundary temperature [°C]

  // Generate structured quadrilateral mesh
  Triangulation<2> triangulation;
  GridGenerator::hyper_rectangle(triangulation, Point<2>(0, 0), Point<2>(0.6, 1.0));
  triangulation.refine_global(5); // 32x64 cells, sufficient for Q2 elements

  // Assign boundary indicators explicitly for robustness
  for (auto &cell : triangulation.active_cell_iterators())
    for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary()) {
        const Point<2> center = cell->face(f)->center();
        if (std::abs(center[1]) < 1e-12)
          cell->face(f)->set_boundary_id(1); // bottom
        else if (std::abs(center[1] - 1.0) < 1e-12)
          cell->face(f)->set_boundary_id(2); // top
        else if (std::abs(center[0] - 0.6) < 1e-12)
          cell->face(f)->set_boundary_id(3); // right
        else if (std::abs(center[0]) < 1e-12)
          cell->face(f)->set_boundary_id(4); // left
      }

  // Finite element and DoF distribution
  FE_Q<2> fe(2);
  DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  const unsigned int n_dofs = dof_handler.n_dofs();

  // Handle Dirichlet boundary conditions on the bottom
  AffineConstraints<double> constraints;
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f) {
      if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == 1) {
        std::vector<unsigned int> face_dofs;
        cell->face(f)->get_dof_indices(face_dofs);
        for (const unsigned int dof : face_dofs)
          constraints.add_line(dof);
      }
    }
  }
  constraints.close();
  for (unsigned int i = 0; i < n_dofs; ++i)
    if (constraints.is_inhomogeneously_constrained(i))
      constraints.set_inhomogeneity(i, T_dirichlet);

  // Assemble system matrix and right-hand side
  SparseMatrix<double> system_matrix;
  Vector<double> system_rhs;
  system_matrix.reinit(n_dofs, n_dofs);
  system_rhs.reinit(n_dofs);

  QGauss<2> q_cell(3);
  QGauss<1> q_face(3);

  FEValues<2> fe_values(fe, q_cell,
                        update_values | update_gradients |
                        update_quadrature_points | update_JxW_values);
  FEFaceValues<2> fe_face_values(fe, q_face,
                                 update_values | update_quadrature_points |
                                 update_JxW_values);

  FullMatrix<double> cell_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
  Vector<double> cell_vector(fe.dofs_per_cell);
  std::vector<unsigned int> dof_indices(fe.dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_vector = 0;

    // Volume integral: k * grad(T) * grad(v)
    for (unsigned int q_point = 0; q_point < q_cell.size(); ++q_point) {
      for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
        for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
          cell_matrix(i, j) += k * fe_values.shape_grad(i, q_point) *
                                   fe_values.shape_grad(j, q_point) *
                                   fe_values.JxW(q_point);
    }

    cell->get_dof_indices(dof_indices);
    system_matrix.add(dof_indices, cell_matrix);
    system_rhs.add(dof_indices, cell_vector);

    // Boundary integrals for Robin conditions (top and right)
    for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
      if (cell->face(f)->at_boundary()) {
        fe_face_values.reinit(cell, f);
        cell_matrix = 0;
        cell_vector = 0;

        if (cell->face(f)->boundary_id() == 2 || cell->face(f)->boundary_id() == 3) {
          for (unsigned int q_point = 0; q_point < q_face.size(); ++q_point) {
            for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
              for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
                cell_matrix(i, j) += h * fe_face_values.shape_value(i, q_point) *
                                         fe_face_values.shape_value(j, q_point) *
                                         fe_face_values.JxW(q_point);
            for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
              cell_vector(i) += h * T_amb * fe_face_values.shape_value(i, q_point) *
                                   fe_face_values.JxW(q_point);
          }
        }

        cell->face(f)->get_dof_indices(dof_indices);
        system_matrix.add(dof_indices, cell_matrix);
        system_rhs.add(dof_indices, cell_vector);
      }
  }

  // Apply constraints to matrix and RHS
  MatrixTools::apply_constraints(constraints, system_matrix);
  MatrixTools::apply_constraints(constraints, system_rhs);

  // Solve linear system
  SolverControl solver_control(10000, 1e-12);
  SolverCG<double> solver(solver_control);
  PreconditionSSOR<double> preconditioner;
  preconditioner.initialize(system_matrix, PreconditionSSOR<double>::AdditionalData(true));

  Vector<double> solution(n_dofs);
  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  // Evaluate solution at reference point
  Point<2> probe_point(0.6, 0.2);
  double probe_value = VectorTools::point_value(fe, solution, probe_point);

  std::cout << "PROBE " << probe_value << std::endl;

  return 0;
}