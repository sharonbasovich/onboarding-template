#pragma once

#include <cstddef>
#include <vector>
// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid
{
private:
  std::size_t rows_;
  std::size_t cols_;
  // std::vector<std::vector<double>> data;
  std::vector<double> data;

public:
  Grid(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), data(rows * cols)
  {
  }

  double &operator()(std::size_t i, std::size_t j)
  {
    // return data[i][j];
    return data[i * cols_ + j];
  }

  double operator()(std::size_t i, std::size_t j) const
  {
    // return data[i][j];
    return data[i * cols_ + j];
  }

  std::size_t rows() const
  {
    return rows_;
  }

  std::size_t cols() const
  {
    return cols_;
  }

  const double *base() const
  {
    return data.data();
  }

  double *base()
  {
    return data.data();
  }
};

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
// void apply_stencil(const Grid &old_grid, Grid &new_grid)
// {
//   std::size_t rows = old_grid.rows();
//   std::size_t cols = old_grid.cols();
//   for (std::size_t i = 1; i < rows - 1; i++)
//   {
//     for (std::size_t j = 1; j < cols - 1; j++)
//     {
//       new_grid(i, j) = 0.5 * old_grid(i, j) + 0.125 * (old_grid(i - 1, j) + old_grid(i, j - 1) + old_grid(i + 1, j) + old_grid(i, j + 1));

//       // potential scatter strategy to look at later
//       // new_grid(i, j) += old_grid(i, j) * 0.5;
//       // new_grid(i, j + 1) += old_grid(i, j) * 0.125;
//       // new_grid(i, j - 1) += old_grid(i, j) * 0.125;
//       // new_grid(i + 1, j) += old_grid(i, j) * 0.125;
//       // new_grid(i - 1, j) += old_grid(i, j) * 0.125;
//     }
//   }

//   for (std::size_t i = 0; i < rows; i++)
//   {
//     new_grid(i, 0) = old_grid(i, 0);
//     new_grid(i, cols - 1) = old_grid(i, cols - 1);
//   }

//   for (std::size_t i = 0; i < cols; i++)
//   {
//     new_grid(0, i) = old_grid(0, i);
//     new_grid(rows - 1, i) = old_grid(rows - 1, i);
//   }
// }

void apply_stencil(const Grid &old_grid, Grid &new_grid)
{
  const double *old_base = old_grid.base();
  double *new_base = new_grid.base();
  std::size_t rows = old_grid.rows();
  std::size_t cols = old_grid.cols();
  for (std::size_t i = 1; i < rows - 1; i++)
  {
    const double *curr = old_base + i * cols;
    const double *prev = old_base + (i - 1) * cols;
    const double *next = old_base + (i + 1) * cols;
    double *new_cell = new_base + i * cols;
    for (std::size_t j = 1; j < cols - 1; j++)
    {
      new_cell[j] = 0.5 * curr[j] + 0.125 * (curr[j + 1] + curr[j - 1] + prev[j] + next[j]);
    }
  }

  for (std::size_t i = 0; i < rows; i++)
  {
    new_grid(i, 0) = old_grid(i, 0);
    new_grid(i, cols - 1) = old_grid(i, cols - 1);
  }

  for (std::size_t i = 0; i < cols; i++)
  {
    new_grid(0, i) = old_grid(0, i);
    new_grid(rows - 1, i) = old_grid(rows - 1, i);
  }
}