add_executable(unitTest unitTest.cpp)
target_link_libraries(unitTest neural2d)

add_executable(neural2d-test neural2d.cpp)
target_link_libraries(neural2d-test neural2d)

enable_testing()

add_test(UnitTest unitTest)
add_test(test neural2d-test ${CMAKE_CURRENT_LIST_DIR}/data/topology.txt ${CMAKE_CURRENT_LIST_DIR}/data/inputData.txt weights.txt)
add_test(test-xor neural2d-test ${CMAKE_CURRENT_LIST_DIR}/data/topology-xor.txt ${CMAKE_CURRENT_LIST_DIR}/data/inputData-xor.txt)
