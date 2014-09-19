CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# Set project name
PROJECT(perceptron)

LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/lib 
)
# Set directories
INCLUDE_DIRECTORIES(
	${CMAKE_SOURCE_DIR}/include
	)

ADD_DEFINITIONS(-DOS_LINUX)
# Add Executables
SET(SRCS   
    ${CMAKE_SOURCE_DIR}/src/perceptron.cpp
    ${CMAKE_SOURCE_DIR}/src/perceptron_demo.cpp
    )
ADD_EXECUTABLE(perceptron ${SRCS})

IF(WIN32)
ELSE(UNIX)
    ADD_DEFINITIONS("-g")
   # TARGET_LINK_LIBRARIES()

ENDIF(WIN32)

SET(CMAKE_INSTALL_PREFIX ".")
INSTALL_TARGETS(../bin perceptron)
