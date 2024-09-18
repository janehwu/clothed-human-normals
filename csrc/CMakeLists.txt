CMAKE_MINIMUM_REQUIRED( VERSION 3.1 )
MARK_AS_ADVANCED(CMAKE_BACKWARDS_COMPATIBILITY)
###########################################################################
#Project-specific Configuration
###########################################################################

SET (CMAKE_CXX_STANDARD 11)

#Set a project name
SET(PROJECT_NAME cir)
PROJECT(${PROJECT_NAME})

#Manually list your cpp and h files here - do NOT use GLOB/GLOB_RECURSE
SET(${PROJECT_NAME}_cpp
        ${PROJECT_SOURCE_DIR}/main.cpp
        ${PROJECT_SOURCE_DIR}/External/cnpy.cpp
        ${PROJECT_SOURCE_DIR}/Topology_Based_Geometry_Level_Sets/FAST_MARCHING_METHOD_TETRAHEDRALIZED_VOLUME.cpp
        #${PROJECT_SOURCE_DIR}/Utils/DEBUG_VIZ_UTIL.cpp
)
SET(${PROJECT_NAME}_h
        ${PROJECT_SOURCE_DIR}/Mesh_Tests/MESH_TESTS.h
        ${PROJECT_SOURCE_DIR}/Tsdf_Tests/TSDF_TESTS.h
        ${PROJECT_SOURCE_DIR}/Embed_Tests/EMBED_TESTS.h
        ${PROJECT_SOURCE_DIR}/Rasterization_Tests/RASTERIZATION_TESTS.h
        ${PROJECT_SOURCE_DIR}/Reconstruction_Tests/RECONSTRUCTION_TESTS.h
        ${PROJECT_SOURCE_DIR}/Refinement_Tests/REFINEMENT_TESTS.h
        ${PROJECT_SOURCE_DIR}/Sdf_Regularization_Tests/SDF_REGULARIZATION_TESTS.h
        ${PROJECT_SOURCE_DIR}/Skin_Tests/SKIN_TESTS.h
        ${PROJECT_SOURCE_DIR}/Fit_Tests/FIT_TESTS.h
        ${PROJECT_SOURCE_DIR}/Fit_Tests/Fit.h
        ${PROJECT_SOURCE_DIR}/Fit_Tests/Utilities.h
        #${PROJECT_SOURCE_DIR}/Skin_Tests/SKIN_WEIGHTS_TESTS.h
        ${PROJECT_SOURCE_DIR}/Topology_Based_Geometry_Level_Sets/FAST_MARCHING_METHOD_TETRAHEDRALIZED_VOLUME.h
        ${PROJECT_SOURCE_DIR}/External/cnpy.h
)

#List the libraries you want to link against - including a higher-level
#library automatically includes the lower-level ones
SET(LIBRARIES_TO_LINK PhysBAM_Rendering-OpenGL)
      
INCLUDE(${PROJECT_SOURCE_DIR}/../../Scripts/CMake/Project.cmake)
