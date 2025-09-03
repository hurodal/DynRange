#pragma once

#include "arguments.hpp"
#include <string>
#include <vector>
#include <ostream>

// --- NUEVO: Añade la definición de esta estructura aquí ---
struct DynamicRangeResult {
    std::string filename;
    double dr_12db;
    double dr_0db;
    int patches_used;
};

// La declaración de la función principal del motor de análisis
bool run_dynamic_range_analysis(const ProgramOptions& opts, std::ostream& log_stream);