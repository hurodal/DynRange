cmake_minimum_required(VERSION 3.16)
project(dynRange CXX)

# Add our local modules directory to CMake's search path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- DEPENDENCY SEARCH ---
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBRAW REQUIRED libraw)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(CLI11 REQUIRED)
find_package(Gettext REQUIRED)
find_package(wxWidgets REQUIRED)


# --- TARGET DEFINITIONS ---

# A variable with all shared logic source files
set(CORE_SOURCES
    src/core/arguments.cpp
    src/core/functions.cpp
    src/core/engine.cpp
)

# TARGET: The Command-Line (CLI) Executable
add_executable(dynRange
    src/dynRange.cpp
    ${CORE_SOURCES}
)
target_link_libraries(dynRange PRIVATE
    ${LIBRAW_LIBRARIES}
    ${OpenCV_LIBS}
    CLI11::CLI11
    ${GETTEXT_LIBRARIES}
)
target_include_directories(dynRange PRIVATE
    ${PROJECT_SOURCE_DIR}
    src/core
    ${LIBRAW_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${GETTEXT_INCLUDE_DIRS}
)

# TARGET: The Graphical User Interface (GUI) Executable
add_executable(dynRangeGui
    src/dynRangeGuiApp.cpp        # NUEVO: Punto de entrada de la aplicación gráfica
    src/gui/DynaRangeBase.cpp     # NUEVO: Código de la GUI generado
    src/gui/DynaRangeFrame.cpp    # NUEVO: Lógica de nuestra GUI
    ${CORE_SOURCES}
)

target_link_libraries(dynRangeGui PRIVATE 
    ${wxWidgets_LIBRARIES}
    ${LIBRAW_LIBRARIES}
    ${OpenCV_LIBS}
    CLI11::CLI11
    ${GETTEXT_LIBRARIES}
)
target_include_directories(dynRangeGui PRIVATE
    ${PROJECT_SOURCE_DIR}
    src/core
    ${wxWidgets_INCLUDE_DIRS}
    ${LIBRAW_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${GETTEXT_INCLUDE_DIRS}
)
target_compile_definitions(dynRangeGui PRIVATE ${wxWidgets_DEFINITIONS})


# --- GETTEXT CONFIGURATION ---
include(UseGettext)
set(GETTEXT_XGETTEXT_FLAGS "--from-code=UTF-8")
set(LANGUAGES "es" "de" "it" "fr" "pt" "ca")

set(I18N_SOURCES
    src/dynRange.cpp
    src/dynRangeGuiApp.cpp      # NUEVO
    src/gui/DynaRangeFrame.cpp  # NUEVO
    ${CORE_SOURCES}
)

GETTEXT_CREATE_POT_FILE(${PROJECT_NAME} ${I18N_SOURCES})
GETTEXT_ADD_PO_FILES(${PROJECT_NAME} ALL ${LANGUAGES})
GETTEXT_PROCESS_PO_FILES(${PROJECT_NAME} ALL INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/locale ${LANGUAGES})


# --- OPTIONAL: INSTALLER CONFIGURATION ---
include(InstallRequiredSystemLibraries)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
install(TARGETS dynRange dynRangeGui DESTINATION bin)
include(CPack)