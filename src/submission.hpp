#pragma once

#include <cstddef>
#include <vector>
#include <thread>
// #include <mutex>
// #include <condition_variable>
#include <atomic>
#include <immintrin.h>

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.

inline void process_rows(const double *__restrict__ old_base,
                         double *__restrict__ new_base,
                         std::size_t begin_row, std::size_t end_row,
                         std::size_t cols)
{
  for (std::size_t i = begin_row; i < end_row; ++i)
  {
    const double *curr = old_base + i * cols;
    const double *prev = old_base + (i - 1) * cols;
    const double *next = old_base + (i + 1) * cols;
    double *new_cell = new_base + i * cols;

    for (std::size_t j = 1; j < cols - 1; ++j)
    {
      new_cell[j] = 0.5 * curr[j] + 0.125 * (curr[j + 1] + curr[j - 1] + prev[j] + next[j]);
    }
  }
}

class ThreadPool
{
private:
  std::vector<std::thread> workers_;
  std::size_t worker_count_ = 0;

  const double *old_base_ = nullptr;
  double *new_base_ = nullptr;

  std::size_t rows_ = 0;
  std::size_t cols_ = 0;

  // std::mutex mutex_;
  // std::condition_variable cv_;
  std::atomic<std::size_t> gen_{0};
  std::atomic<bool> stop_{false};

  // std::condition_variable done_cv_;
  std::atomic<std::size_t> completed_{0};

  void worker_loop(std::size_t worker_id)
  {
    std::size_t seen_gen = 0;

    while (true)
    {
      // spin waiting
      std::size_t current_gen;
      while (true)
      {
        if (stop_.load(std::memory_order_acquire))
        {
          return;
        }

        current_gen = gen_.load(std::memory_order_acquire);

        if (current_gen != seen_gen)
        {
          break;
        }

        _mm_pause();
      }

      // std::unique_lock<std::mutex> lock(mutex_);

      // cv_.wait(lock, [&]
      //          { return stop_ || gen_ != seen_gen; });

      // if (stop_)
      // {
      //   return;
      // }

      const double *old_base = old_base_;
      double *new_base = new_base_;
      const std::size_t rows = rows_;
      const std::size_t cols = cols_;

      seen_gen = current_gen;

      // lock.unlock();

      // rows - 2 is interior rows, worker_count_ + 1 since main thread also calculates chunk
      const std::size_t begin_row = 1 + (rows - 2) * worker_id / (worker_count_ + 1);
      const std::size_t end_row = 1 + (rows - 2) * (worker_id + 1) / (worker_count_ + 1);

      process_rows(old_base, new_base, begin_row, end_row, cols);

      // {
      // std::lock_guard<std::mutex> lock(mutex_);

      // ++completed_;

      // if (completed_ == worker_count_)
      // {
      //   done_cv_.notify_one();
      // }

      // const std::size_t finished = completed_.fetch_add(1) + 1;

      completed_.fetch_add(1, std::memory_order_acq_rel);

      // if (finished == worker_count_)
      // {
      //   std::lock_guard<std::mutex> lock(mutex_);
      //   done_cv_.notify_one();
      // }
      // }
    }
  }

public:
  ThreadPool()
  {
    worker_count_ = 3;

    workers_.reserve(worker_count_);

    for (std::size_t i = 0; i < worker_count_; ++i)
    {
      workers_.emplace_back([this, i]
                            { worker_loop(i); });
    }
  }

  ~ThreadPool()
  {

    stop_.store(true, std::memory_order_release);

    // {
    //   std::lock_guard<std::mutex> lock(mutex_);
    //   stop_ = true;
    // }

    // cv_.notify_all();

    for (std::thread &worker : workers_)
    {
      if (worker.joinable())
      {
        worker.join();
      }
    }
  }

  void run(const double *old_base, double *new_base, std::size_t rows, std::size_t cols)
  {

    if (worker_count_ == 0)
    {
      process_rows(
          old_base,
          new_base,
          1,
          rows - 1,
          cols);

      return;
    }
    {
      // std::lock_guard<std::mutex> lock(mutex_);

      old_base_ = old_base;
      new_base_ = new_base;
      rows_ = rows;
      cols_ = cols;

      // completed_ = 0;
      completed_.store(0, std::memory_order_relaxed);

      // ++gen_;
      gen_.fetch_add(1, std::memory_order_release);
    }

    // cv_.notify_all();

    // rows - 2 is interior rows, worker_count_ + 1 since main thread also calculates chunk
    const std::size_t begin_row = 1 + (rows - 2) * worker_count_ / (worker_count_ + 1);

    const std::size_t end_row = rows - 1;

    process_rows(old_base, new_base, begin_row, end_row, cols);

    // std::unique_lock<std::mutex> lock(mutex_);

    /* done_cv_.wait(lock, [&]
                  {
                    // return completed_ == worker_count_;
                  return completed_.load() == worker_count_; }); */
    while (completed_.load(std::memory_order_acquire) != worker_count_)
    {
      _mm_pause();
    }
  }
};

inline ThreadPool &get_thread_pool()
{
  static ThreadPool pool;
  return pool;
}

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
    (void)get_thread_pool();
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

void apply_stencil(const Grid &old_grid, Grid &new_grid)
{
  // const double *old_base = old_grid.base();
  // double *new_base = new_grid.base();
  const double *__restrict__ old_base = old_grid.base();
  double *__restrict__ new_base = new_grid.base();
  const std::size_t rows = old_grid.rows();
  const std::size_t cols = old_grid.cols();

  // handle small grids where all cells are boundary
  if (rows < 3 || cols < 3)
  {
    for (std::size_t i = 0; i < rows; ++i)
    {
      for (std::size_t j = 0; j < cols; ++j)
      {
        new_grid(i, j) = old_grid(i, j);
      }
    }

    return;
  }

  // const double *curr = old_base + cols;
  // const double *prev = old_base;
  // const double *next = old_base + 2 * cols;
  // double *new_cell = new_base + cols;

  // non-custom threading
  // #pragma omp parallel for schedule(static)
  // #pragma omp parallel for
  //   for (std::size_t i = 1; i < rows - 1; ++i)
  //   {

  //     process_rows(old_base, new_base, i, cols);
  //   }

  // process_rows(old_base, new_base, 1, rows - 1, cols);

  get_thread_pool().run(old_base, new_base, rows, cols);

  for (std::size_t i = 0; i < rows; ++i)
  {
    new_grid(i, 0) = old_grid(i, 0);
    new_grid(i, cols - 1) = old_grid(i, cols - 1);
  }

  for (std::size_t i = 0; i < cols; ++i)
  {
    new_grid(0, i) = old_grid(0, i);
    new_grid(rows - 1, i) = old_grid(rows - 1, i);
  }
}