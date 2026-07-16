cmake_minimum_required(VERSION 3.20)
project(FraudNet CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(fraud_net
    src/main.cpp
    src/utec/agent/fraud_agent.cpp
    src/utec/utils/csv_loader.cpp
)

target_include_directories(fraud_net PRIVATE 
    include
    include/utec/algebra
    include/utec/nn
)
