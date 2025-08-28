project('CNN', 'cpp', default_options : ['cpp_std=c++20'])

gtest_subproj = subproject('gtest')
gtest_dep = gtest_subproj.get_variable('gtest_main_dep')

CNN_dir = include_directories('../include')

omp_dep = dependency('openmp', required: true)

test_sources = ['helper_test.cpp', 'learning_test.cpp', 'type_test.cpp']
foreach src : test_sources
  exe = executable(src.split('.')[0], src,
      include_directories : CNN_dir,
      dependencies : [gtest_dep, omp_dep]
  )
  test(src, exe)
endforeach
