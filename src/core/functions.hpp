#pragma once

#include "arguments.hpp"
#include "spline.h"
#include <string>
#include <vector>
#include <optional>
#include <ostream>
#include <opencv2/core.hpp>
#include <Eigen/Dense>

struct PatchAnalysisResult {
    std::vector<double> signal;
    std::vector<double> noise;
};

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsubobject-linkage"
#endif

struct DRCalcResult {
    double dr_value = 0.0;
    tk::spline spline_model;
    cv::Mat poly_coeffs;
    std::vector<double> filtered_snr;
    std::vector<double> filtered_signal;
};

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

// --- Declaraciones de funciones ---
std::optional<double> process_dark_frame(const std::string& dark_file_path, std::ostream& log_stream);
std::optional<double> process_saturation_frame(const std::string& sat_file_path, std::ostream& log_stream);
bool prepare_and_sort_files(ProgramOptions& opts, std::ostream& log_stream);
Eigen::VectorXd calculate_keystone_params(const std::vector<cv::Point2d>& src, const std::vector<cv::Point2d>& dst);
cv::Mat undo_keystone(const cv::Mat& img, const Eigen::VectorXd& k);
PatchAnalysisResult analyze_patches(const cv::Mat& img, int NCOLS, int NROWS, double SAFE);
void polyfit(const cv::Mat& src_x, const cv::Mat& src_y, cv::Mat& dst, int order);

// --- CAMBIO: Nueva firma para la función de gráficos ---
void generate_debug_plot(
    const std::string& output_filename,
    const std::string& plot_title,
    const std::vector<double>& all_snr_db,
    const std::vector<double>& all_signal_ev,
    const DRCalcResult& result,
    const std::optional<cv::Point2d>& x_range = std::nullopt,
    const std::optional<cv::Point2d>& y_range = std::nullopt,
    int width = 2000, // Aumentamos la resolución por defecto
    int height = 1600
);