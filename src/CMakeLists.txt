aux_source_directory(. SRC)
list(FILTER SRC EXCLUDE REGEX "main*")
add_library(vgg16_circuit_lib ${SRC})

add_executable(vgg16_circuit_run main.cpp)
target_link_libraries(vgg16_circuit_run vgg16_circuit_lib)