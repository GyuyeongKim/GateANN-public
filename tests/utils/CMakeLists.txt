set(CMAKE_CXX_STANDARD 17)

add_executable(bin_to_fvecs bin_to_fvecs.cpp)
target_link_libraries(bin_to_fvecs ${PROJECT_NAME})

add_executable(vecs_to_bin vecs_to_bin.cpp)
target_link_libraries(vecs_to_bin ${PROJECT_NAME})

add_executable(gen_random_slice gen_random_slice.cpp)
target_link_libraries(gen_random_slice ${PROJECT_NAME})

add_executable(compute_groundtruth compute_groundtruth.cpp)
target_link_libraries(compute_groundtruth ${PROJECT_NAME})

add_executable(gen_tags gen_tags.cpp)
target_link_libraries(gen_tags ${PROJECT_NAME})

add_executable(gt_update gt_update.cpp)
target_link_libraries(gt_update ${PROJECT_NAME} )

add_executable(change_pts change_pts.cpp)
target_link_libraries(change_pts ${PROJECT_NAME})