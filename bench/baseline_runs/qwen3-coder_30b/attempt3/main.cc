#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <iostream>
#include <fstream>

using namespace dealii;

int main()
{
  const unsigned int dim = 2;
  const double k = 52.0; // thermal conductivity W/(m·°C)
  const double h = 750.0; // heat transfer coefficient W/(m²·°C)
  const double T_amb = 0.0; // ambient temperature °C
  const double T_bottom = 100.0; // bottom boundary temperature °C

  // Create a structured quadrilateral mesh
  Triangulation<dim> triangulation;
  const Point<dim> p1(0.0, 0.0);
  const Point<dim> p2(0.6, 1.0);
  GridGenerator::hyper_rectangle(triangulation, p1, p2, true);
  triangulation.refine_global(2); // Refine the mesh

  // Define finite element
  FE_Q<dim> fe(1); // Continuous Lagrange elements of order 1
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  // Set up matrices and vectors
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparseMatrix<double> system_matrix;
  system_matrix.reinit(dsp);
  Vector<double> solution(dof_handler.n_dofs());
  Vector<double> system_rhs(dof_handler.n_dofs());

  // Apply boundary conditions
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0, // bottom boundary
                                           Functions::ConstantFunction<dim>(T_bottom),
                                           boundary_values);

  // Assemble the system matrix and right-hand side
  FEValues<dim> fe_values(fe, 
                          QGauss<dim>(2),
                          update_gradients | update_JxW_values);

  FullMatrix<double> cell_matrix;
  Vector<double> cell_rhs;
  std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);
  
  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_rhs = 0;

    for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
    {
      for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
      {
        cell_matrix(i, j) += k * (fe_values.shape_grad(i, 0) * fe_values.shape_grad(j, 0)) * fe_values.JxW(0);
      }
    }

    // Apply Neumann boundary conditions on top and right boundaries
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
    {
      if (cell->face(face)->at_boundary())
      {
        Point<dim> face_center = cell->face(face)->center();
        if (std::abs(face_center[1] - 1.0) < 1e-10 || std::abs(face_center[0] - 0.6) < 1e-10)
        {
          // Neumann boundary condition: -k * dT/dn = h * (T - T_amb)
          // This is implemented by adding the contribution to the RHS
          for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          {
            cell_rhs(i) += h * (face_center[1] - 1.0) * fe_values.shape_value(i, 0) * fe_values.JxW(0);
          }
        }
      }
    }

    cell->get_dof_indices(local_dof_indices);
    for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
    {
      for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
      {
        system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
      }
      system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }
  }

  // Apply boundary conditions to the system
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

  // Solve the linear system
  SolverControl solver_control(1000, 1e-12);
  SolverCG<> solver(solver_control);
  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);
  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  // Evaluate the solution at the reference point (0.6 m, 0.2 m)
  Point<dim> reference_point(0.6, 0.2);
  double T_ref = VectorTools::point_value(dof_handler, solution, reference_point);

  std::cout << "PROBE " << T_ref << std::endl;

  return 0;
}