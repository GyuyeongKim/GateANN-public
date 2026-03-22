set(CMAKE_CXX_STANDARD 17)

add_executable(build_disk_index build_disk_index.cpp)
target_link_libraries(build_disk_index ${PROJECT_NAME})

add_executable(build_memory_index build_memory_index.cpp)
target_link_libraries(build_memory_index ${PROJECT_NAME})

# GateANN: filtered search experiment binary (modes 0/2/8/9/10)
add_executable(search_disk_index_fa search_disk_index_fa.cpp)
target_link_libraries(search_disk_index_fa ${PROJECT_NAME})

# GateANN: YFCC multi-label experiment binary
add_executable(search_disk_index_yfcc search_disk_index_yfcc.cpp)
target_link_libraries(search_disk_index_yfcc ${PROJECT_NAME})

# In-memory Vamana search with post-filter (ideal baseline)
add_executable(search_mem_fa search_mem_fa.cpp)
target_link_libraries(search_mem_fa ${PROJECT_NAME})
