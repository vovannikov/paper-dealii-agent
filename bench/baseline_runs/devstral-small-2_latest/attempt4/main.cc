#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <fstream>
#include <iostream>

namespace HeatTransfer
{
  using namespace dealii;

  // Thermal conductivity
  const double k = 52.0;
  // Heat transfer coefficient
  const double h = 750.0;
  // Ambient temperature
  const double Tamb = 0.0;

  // Right-hand side function (zero for this problem)
  class RightHandSide : public Function<2>
  {
  public:
    RightHandSide() = default;
    virtual double value(const Point<2> &p, const unsigned int component = 0) const override
    {
      return 0.0;
    }
  };

  // Boundary conditions
  class BoundaryValues : public Function<2>
  {
  public:
    BoundaryValues() = default;
    virtual double value(const Point<2> &p, const unsigned int component = 0) const override
    {
      // Bottom boundary (y=0)
      if (p[1] == 0.0)
        return 100.0;
      // Top and right boundaries (Newton's law of cooling)
      else if (p[1] == 1.0 || p[0] == 0.6)
        return Tamb;
      // Left boundary (insulated, no condition needed here)
      else
        return 0.0;
    }
  };

  // Convection boundary condition (for top and right boundaries)
  class ConvectionBoundary : public Function<2>
  {
  public:
    ConvectionBoundary() = default;
    virtual double value(const Point<2> &p, const unsigned int component = 0) const override
    {
      // Only apply to top and right boundaries
      if (p[1] == 1.0 || p[0] == 0.6)
        return h * (Tamb - value(p));
      else
        return 0.0;
    }
  };

  template <int dim>
  class HeatTransferProblem
  {
  public:
    HeatTransferProblem();
    void run();

  private:
    void make_grid();
    void setup_system();
    void assemble_system();
    void solve();
    void output_results() const;

    Triangulation<dim> triangulation;
    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> solution;
    Vector<double> system_rhs;
  };

  template <int dim>
  HeatTransferProblem<dim>::HeatTransferProblem()
    : fe(1), dof_handler(triangulation)
  {}

  template <int dim>
  void HeatTransferProblem<dim>::make_grid()
  {
    GridGenerator::subdivided_hyper_rectangle(triangulation, 12, 20, Point<dim>(), Point<dim>(0.6, 1.0));
  }

  template <int dim>
  void HeatTransferProblem<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);
    std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }

  template <int dim>
  void HeatTransferProblem<dim>::assemble_system()
  {
    QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe, quadrature_formula,
                           update_values | update_gradients |
                           update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    RightHandSide rhs_function;
    BoundaryValues boundary_values;
    ConvectionBoundary convection_boundary;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell_matrix = 0;
      cell_rhs = 0;
      fe_values.reinit(cell);

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
        for (const unsigned int i : fe_values.dof_indices())
        {
          for (const unsigned int j : fe_values.dof_indices())
          {
            cell_matrix(i, j) += (k * fe_values.shape_grad(i, q_index) *
                                  fe_values.shape_grad(j, q_index) +
                                  (convection_boundary.value(fe_values.quadrature_point(q_index)) *
                                   fe_values.shape_value(i, q_index) *
                                   fe_values.shape_value(j, q_index))) *
                                 fe_values.JxW(q_index);
          }
          cell_rhs(i) += rhs_function.value(fe_values.quadrature_point(q_index)) *
                         fe_values.shape_value(i, q_index) *
                         fe_values.JxW(q_index);
        }
      }

      for (const auto face : cell->face_indices())
      {
        if (cell->face(face)->at_boundary())
        {
          FEFaceValues<dim> fe_face_values(fe, quadrature_formula,
                                          update_values | update_quadrature_points | update_JxW_values);
          fe_face_values.reinit(cell, face);
          for (const unsigned int q_index : fe_face_values.quadrature_point_indices())
          {
            if (boundary_values.value(fe_face_values.quadrature_point(q_index)) != 0.0)
            {
              for (const unsigned int i : fe_face_values.dof_indices())
              {
                cell_rhs(i) += boundary_values.value(fe_face_values.quadrature_point(q_index)) *
                               fe_face_values.shape_value(i, q_index) *
                               fe_face_values.JxW(q_index);
              }
            }
          }
        }
      }

      cell->get_dof_indices(local_dof_indices);
      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));

      for (const unsigned int i : fe_values.dof_indices())
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }
    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }

  template <int dim>
  void HeatTransferProblem<dim>::solve()
  {
    SolverControl solver_control(1000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
  }

  template <int dim>
  void HeatTransferProblem<dim>::output_results() const
  {
    std::ofstream output("solution.vtk");
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "Temperature");
    data_out.build_patches();
    data_out.write_vtk(output);

    // Evaluate solution at reference point (0.6, 0.2)
    Point<dim> probe_point(0.6, 0.2);
    double probe_value = VectorTools::point_value(dof_handler, solution, probe_point);
    std::cout << "PROBE " << probe_value << std::endl;
  }

  template <int dim>
  void HeatTransferProblem<dim>::run()
  {
    make_grid();
    setup_system();
    assemble_system();
    solve();
    output_results();
  }
}

int main()
{
  try
  {
    HeatTransfer::HeatTransferProblem<2> heat_transfer_problem;
    heat_transfer_problem.run();
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  return 0;
}