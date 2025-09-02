// core/functions.hpp
#pragma once

#include <string>
#include <vector>
#include <optional>
#include <ostream>
#include "arguments.hpp" // Required to know about the ProgramOptions struct
#include "spline.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp> // Necesario para cv::Mat
#include <Eigen/Dense>

// Data structures used in both main and the functions
struct DynamicRangeResult {
    std::string filename;
    double dr_12db;
    double dr_0db;
    int patches_used;
};

struct PatchAnalysisResult {
    std::vector<double> signal;
    std::vector<double> noise;
    cv::Mat image_with_patches;
};


// --- NUEVO: Estructura para devolver los resultados del cálculo de DR ---
struct DRCalcResult {
    double dr_value = 0.0;
    tk::spline spline_model;
    cv::Mat poly_coeffs;
    std::vector<double> filtered_snr;
    std::vector<double> filtered_signal;
};

// Declarations of image processing functions
Eigen::VectorXd calculate_keystone_params(
    const std::vector<cv::Point2d>& src_points,
    const std::vector<cv::Point2d>& dst_points
);

cv::Mat undo_keystone(const cv::Mat& imgSrc, const Eigen::VectorXd& k);

PatchAnalysisResult analyze_patches(cv::Mat imgcrop, int NCOLS, int NROWS, double SAFE);

// --- CALCULATION AND DATA EXTRACTION FUNCTIONS ---
std::optional<std::vector<double>> extract_raw_pixels(const std::string& filename);
double calculate_mean(const std::vector<double>& data);
double calculate_quantile(std::vector<double>& data, double percentile);
std::optional<double> estimate_mean_brightness(const std::string& filename, float sample_ratio = 0.1f);

// Modified to accept a log stream for GUI output
std::optional<double> process_dark_frame(const std::string& filename, std::ostream& log_stream);
std::optional<double> process_saturation_frame(const std::string& filename, std::ostream& log_stream);

// --- Centralized function for file preparation and sorting ---
bool prepare_and_sort_files(ProgramOptions& opts, std::ostream& log_stream);

void polyfit(const cv::Mat& src_x, const cv::Mat& src_y, cv::Mat& dst, int order);

// Declaración función para generar gráficos
void generate_debug_plot(
    const std::string& output_filename,
    const std::string& plot_title,
    const std::vector<double>& all_snr_db,
    const std::vector<double>& all_signal_ev,
    const DRCalcResult& result
);