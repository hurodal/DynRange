// core/functions.cpp
#include "functions.hpp"
#include <libraw/libraw.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <opencv2/imgproc.hpp>   // Para dibujar (líneas, círculos, texto)
#include <opencv2/imgcodecs.hpp> // Para guardar la imagen (imwrite)

namespace fs = std::filesystem;

/**
 * @brief Calculates the parameters of a projective (keystone) transformation.
 * @param src_points Vector with the 4 source points (corners of the distorted object).
 * @param dst_points Vector with the 4 destination points (corners of the desired rectangle).
 * @return An Eigen::VectorXd object with the 8 transformation parameters.
 */
Eigen::VectorXd calculate_keystone_params(
    const std::vector<cv::Point2d>& src_points,
    const std::vector<cv::Point2d>& dst_points
) {
    Eigen::Matrix<double, 8, 8> A;
    Eigen::Vector<double, 8> b;
    for (int i = 0; i < 4; ++i) {
        const auto& xu = src_points[i].x; const auto& yu = src_points[i].y;
        const auto& xd = dst_points[i].x; const auto& yd = dst_points[i].y;
        A.row(2 * i)     << xd, yd, 1, 0,  0,  0, -xd * xu, -yd * xu;
        A.row(2 * i + 1) << 0,  0,  0, xd, yd, 1, -xd * yu, -yd * yu;
        b(2 * i) = xu; b(2 * i + 1) = yu;
    }
    return A.colPivHouseholderQr().solve(b);
}

/**
 * @brief Applies a keystone distortion correction to an image.
 * @param imgSrc Input image (must be of type CV_32FC1).
 * @param k Transformation parameters obtained from calculate_keystone_params.
 * @return A new cv::Mat image with the correction applied.
 */
cv::Mat undo_keystone(const cv::Mat& imgSrc, const Eigen::VectorXd& k) {
    int DIMX = imgSrc.cols; int DIMY = imgSrc.rows;
    cv::Mat imgCorrected = cv::Mat::zeros(DIMY, DIMX, CV_32FC1);
    for (int y = 0; y < DIMY; ++y) {
        for (int x = 0; x < DIMX; ++x) {
            double xd = x + 1.0, yd = y + 1.0;
            double denom = k(6) * xd + k(7) * yd + 1.0;
            double xu = (k(0) * xd + k(1) * yd + k(2)) / denom;
            double yu = (k(3) * xd + k(4) * yd + k(5)) / denom;
            int x_src = static_cast<int>(round(xu)) - 1;
            int y_src = static_cast<int>(round(yu)) - 1;
            if (x_src >= 0 && x_src < DIMX && y_src >= 0 && y_src < DIMY) {
                imgCorrected.at<float>(y, x) = imgSrc.at<float>(y_src, x_src);
            }
        }
    }
    return imgCorrected;
}

/**
 * @brief Analyzes an image by dividing it into patches and calculates the signal and noise for each.
 * @param imgcrop Cropped image to be analyzed.
 * @param NCOLS Number of columns in the patch grid.
 * @param NROWS Number of rows in the patch grid.
 * @param SAFE Safety margin to avoid the edges of each patch.
 * @return A PatchAnalysisResult structure with the signal/noise vectors and a visual image.
 */
PatchAnalysisResult analyze_patches(cv::Mat imgcrop, int NCOLS, int NROWS, double SAFE) {
    std::vector<double> signal_vec, noise_vec;
    for (int j = 0; j < NROWS; ++j) {
        for (int i = 0; i < NCOLS; ++i) {
            int x1 = round((double)i * imgcrop.cols / NCOLS + SAFE);
            int x2 = round((double)(i + 1) * imgcrop.cols / NCOLS - SAFE);
            int y1 = round((double)j * imgcrop.rows / NROWS + SAFE);
            int y2 = round((double)(j + 1) * imgcrop.rows / NROWS - SAFE);
            if (x1 >= x2 || y1 >= y2) continue;

            cv::Mat patch = imgcrop(cv::Rect(x1, y1, x2 - x1, y2 - y1));
            cv::Scalar mean, stddev;
            cv::meanStdDev(patch, mean, stddev);

            double S = mean[0], N = stddev[0];
            int sat_count = cv::countNonZero(patch > 0.9);
            double sat_ratio = (double)sat_count / (patch.rows * patch.cols);

            if (S > 0 && N > 0 && 20 * log10(S / N) >= -10 && sat_ratio < 0.01) {
                signal_vec.push_back(S); noise_vec.push_back(N);
                cv::rectangle(imgcrop, {x1, y1}, {x2, y2}, cv::Scalar(0.0), 1);
                cv::rectangle(imgcrop, {x1 - 1, y1 - 1}, {x2 + 1, y2 + 1}, cv::Scalar(1.0), 1);
            }
        }
    }
    return {signal_vec, noise_vec, imgcrop};
}

/**
 * @brief Extracts all pixel values from a RAW file into a vector of doubles.
 * @param filename Path to the RAW file.
 * @return An std::optional containing a std::vector<double> with the data on success,
 * or std::nullopt on error.
 */
std::optional<std::vector<double>> extract_raw_pixels(const std::string& filename) {
    LibRaw raw_processor;
    if (raw_processor.open_file(filename.c_str()) != LIBRAW_SUCCESS) {
        std::cerr << "Error: Could not open RAW file: " << filename << std::endl;
        return std::nullopt;
    }
    if (raw_processor.unpack() != LIBRAW_SUCCESS) {
        std::cerr << "Error: Could not decode RAW data from: " << filename << std::endl;
        return std::nullopt;
    }

    int width = raw_processor.imgdata.sizes.raw_width;
    int height = raw_processor.imgdata.sizes.raw_height;
    size_t num_pixels = (size_t)width * height;

    if (num_pixels == 0) {
        return std::nullopt;
    }

    std::vector<double> pixels;
    pixels.reserve(num_pixels);

    unsigned short* raw_data = raw_processor.imgdata.rawdata.raw_image;
    for (size_t i = 0; i < num_pixels; ++i) {
        pixels.push_back(static_cast<double>(raw_data[i]));
    }

    return pixels;
}

/**
 * @brief Calculates the mean (average) of the values in a vector.
 * @param data Input vector (const, not modified).
 * @return The mean value as a double.
 */
double calculate_mean(const std::vector<double>& data) {
    if (data.empty()) {
        return 0.0;
    }
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

/**
 * @brief Calculates a specific quantile (percentile) of a dataset.
 * @param data Input vector. IMPORTANT: The contents of the vector will be modified (partially sorted).
 * @param percentile The desired percentile (e.g., 0.05 for 5%, 0.5 for the median).
 * @return The quantile value as a double.
 */
double calculate_quantile(std::vector<double>& data, double percentile) {
    if (data.empty()) {
        return 0.0;
    }
    
    size_t n = static_cast<size_t>(data.size() * percentile);
    n = std::min(n, data.size() - 1);

    std::nth_element(data.begin(), data.begin() + n, data.end());
    
    return data[n];
}

/**
 * @brief Processes a dark frame RAW file to get the black level (mean).
 * @param filename Path to the dark frame RAW file.
 * @param log_stream The output stream for progress messages.
 * @return An optional containing the calculated black level, or nullopt on failure.
 */
std::optional<double> process_dark_frame(const std::string& filename, std::ostream& log_stream) {
    log_stream << "[INFO] Calculating black level from: " << filename << "..." << std::endl;
    auto pixels_opt = extract_raw_pixels(filename);
    if (!pixels_opt) {
        return std::nullopt;
    }
    
    double mean_value = calculate_mean(*pixels_opt);
    log_stream << "[INFO] -> Black level obtained: " 
               << std::fixed << std::setprecision(2) << mean_value << std::endl;
              
    return mean_value;
}

/**
 * @brief Processes a saturation RAW file to get the saturation point (quantile).
 * @param filename Path to the saturation RAW file.
 * @param log_stream The output stream for progress messages.
 * @return An optional containing the calculated saturation point, or nullopt on failure.
 */
std::optional<double> process_saturation_frame(const std::string& filename, std::ostream& log_stream) {
    log_stream << "[INFO] Calculating saturation point from: " << filename << "..." << std::endl;
    auto pixels_opt = extract_raw_pixels(filename);
    if (!pixels_opt) {
        return std::nullopt;
    }

    double quantile_value = calculate_quantile(*pixels_opt, 0.05);
    log_stream << "[INFO] -> Saturation point obtained (5th percentile): " 
               << std::fixed << std::setprecision(2) << quantile_value << std::endl;

    return quantile_value;
}

/**
 * @brief Estimates the mean brightness of a RAW file by reading only a fraction of its pixels.
 * @param filename Path to the RAW file.
 * @param sample_ratio Fraction of pixels to sample (e.g., 0.1 for 10%). The default value is specified in the .hpp.
 * @return An std::optional containing the estimated mean on success.
 */
std::optional<double> estimate_mean_brightness(const std::string& filename, float sample_ratio) {
    LibRaw raw_processor;
    if (raw_processor.open_file(filename.c_str()) != LIBRAW_SUCCESS || raw_processor.unpack() != LIBRAW_SUCCESS) {
        return std::nullopt;
    }

    size_t num_pixels = (size_t)raw_processor.imgdata.sizes.raw_width * raw_processor.imgdata.sizes.raw_height;
    if (num_pixels == 0) {
        return std::nullopt;
    }

    int step = (sample_ratio > 0 && sample_ratio < 1) ? static_cast<int>(1.0f / sample_ratio) : 1;

    unsigned short* raw_data = raw_processor.imgdata.rawdata.raw_image;
    
    double sum = 0.0;
    long long count = 0;

    for (size_t i = 0; i < num_pixels; i += step) {
        sum += static_cast<double>(raw_data[i]);
        count++;
    }

    return (count > 0) ? (sum / count) : 0.0;
}

/**
 * @brief Pre-analyzes and sorts the input files based on their mean brightness.
 * @param opts The ProgramOptions struct, whose input_files member will be sorted in place.
 * @param log_stream The output stream for progress messages.
 * @return True if successful, false if no files could be processed.
 */
bool prepare_and_sort_files(ProgramOptions& opts, std::ostream& log_stream) {
    // Temporary structure to associate each file with its brightness.
    struct FileExposureInfo {
        std::string filename;
        double mean_brightness;
    };

    std::vector<FileExposureInfo> exposure_data;
    log_stream << "Pre-analyzing files to sort by exposure (using fast sampling)..." << std::endl;

    // Pre-analysis loop: calculates the estimated brightness of each file.
    for (const std::string& name : opts.input_files) {
        auto mean_val_opt = estimate_mean_brightness(name, 0.05f);
        if (mean_val_opt) {
            exposure_data.push_back({name, *mean_val_opt});
            log_stream << "  - " << "File: " << fs::path(name).filename().string()
                       << ", " << "Estimated brightness: " << std::fixed << std::setprecision(2) << *mean_val_opt << std::endl;
        }
    }

    if (exposure_data.empty()) {
        log_stream << "Error: None of the input files could be processed." << std::endl;
        return false;
    }

    // Sort the list of files based on mean brightness.
    std::sort(exposure_data.begin(), exposure_data.end(),
        [](const FileExposureInfo& a, const FileExposureInfo& b) {
            return a.mean_brightness < b.mean_brightness;
        }
    );

    // Update the 'opts' file list with the now-sorted list.
    opts.input_files.clear();
    for (const auto& info : exposure_data) {
        opts.input_files.push_back(info.filename);
    }
    
    log_stream << "Sorting finished. Starting Dynamic Range calculation process..." << std::endl;
    return true;
}

// Realiza un ajuste de mínimos cuadrados para encontrar los coeficientes de un polinomio.
void polyfit(const cv::Mat& src_x, const cv::Mat& src_y, cv::Mat& dst, int order)
{
    CV_Assert(src_x.rows > 0 && src_y.rows > 0 && src_x.total() == src_y.total() && src_x.rows >= order + 1);

    cv::Mat A = cv::Mat::zeros(src_x.rows, order + 1, CV_64F);

    for (int i = 0; i < src_x.rows; ++i) {
        for (int j = 0; j <= order; ++j) {
            A.at<double>(i, j) = std::pow(src_x.at<double>(i), j);
        }
    }
    
    // Inversión de columnas para que los coeficientes salgan en el orden esperado (mayor a menor potencia)
    cv::Mat A_flipped;
    cv::flip(A, A_flipped, 1);

    cv::solve(A_flipped, src_y, dst, cv::DECOMP_SVD);
}

void generate_debug_plot(
    const std::string& output_filename,
    const std::string& plot_title,
    const std::vector<double>& all_snr_db,
    const std::vector<double>& all_signal_ev,
const DRCalcResult& result)
{
    // --- 1. Configuración del lienzo y coordenadas ---
    const int width = 1000, height = 800, margin = 60;
    cv::Mat plot_img(height, width, CV_8UC3, cv::Scalar(255, 255, 255)); // Lienzo blanco

    double min_snr = -10, max_snr = 45;
    double min_sig = -12, max_sig = 0;

    auto to_pixel = [&](double snr, double sig) {
        int px = margin + (int)((snr - min_snr) / (max_snr - min_snr) * (width - 2 * margin));
        int py = margin + (int)((max_sig - sig) / (max_sig - min_sig) * (height - 2 * margin));
        return cv::Point(px, py);
    };

    // --- 2. Dibujar ejes y rejilla ---
    cv::line(plot_img, {margin, margin}, {margin, height - margin}, cv::Scalar(0,0,0), 2);
    cv::line(plot_img, {margin, height - margin}, {width - margin, height - margin}, cv::Scalar(0,0,0), 2);
    cv::putText(plot_img, "SNR (dB)", {width/2 - 40, height - 15}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,0,0}, 2);
    cv::putText(plot_img, "Signal (EV)", {10, height/2}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,0,0}, 2, cv::LINE_AA, true);
    cv::putText(plot_img, plot_title, {width/2 - 150, 35}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,0,0}, 2);

    // --- 3. Dibujar los datos ---
    for (size_t i = 0; i < all_snr_db.size(); ++i) {
        cv::circle(plot_img, to_pixel(all_snr_db[i], all_signal_ev[i]), 4, cv::Scalar(200, 200, 200), -1);
    }
    for (size_t i = 0; i < result.filtered_snr.size(); ++i) {
        cv::circle(plot_img, to_pixel(result.filtered_snr[i], result.filtered_signal[i]), 5, cv::Scalar(0, 0, 0), -1);
    }

    // --- 4. Dibujar las curvas de ajuste ---
    const double step = 0.1;
    // Curva Spline (en rojo)
    // --- CORRECCIÓN AQUÍ ---
    if (result.filtered_snr.size() >= 2) {
        for (double x = result.filtered_snr.front(); x <= result.filtered_snr.back(); x += step) {
            cv::Point p1 = to_pixel(x, result.spline_model(x));
            cv::Point p2 = to_pixel(x + step, result.spline_model(x + step));
            cv::line(plot_img, p1, p2, cv::Scalar(0, 0, 255), 2); // Rojo
        }
    }
    // Curva Polinómica (en azul)
    if (!result.poly_coeffs.empty()) {
        double c2 = result.poly_coeffs.at<double>(0);
        double c1 = result.poly_coeffs.at<double>(1);
        double c0 = result.poly_coeffs.at<double>(2);
        for (double x = min_snr; x <= max_snr; x += step) {
            double y = c2 * x * x + c1 * x + c0;
            double y_next = c2 * (x + step) * (x + step) + c1 * (x + step) + c0;
            cv::line(plot_img, to_pixel(x, y), to_pixel(x + step, y_next), cv::Scalar(255, 0, 0), 2); // Azul
        }
    }

    // --- 5. Dibujar leyenda ---
    cv::circle(plot_img, {width - 200, 60}, 5, {0,0,0}, -1);
    cv::putText(plot_img, "Filtered Data", {width - 180, 65}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,0,0}, 1);
    cv::line(plot_img, {width - 200, 90}, {width - 180, 90}, {255,0,0}, 2);
    cv::putText(plot_img, "Polynomial Fit", {width - 170, 95}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,0,0}, 1);
    cv::line(plot_img, {width - 200, 120}, {width - 180, 120}, {0,0,255}, 2);
    cv::putText(plot_img, "Spline", {width - 170, 125}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,0,0}, 1);

    // --- 6. Guardar la imagen ---
    cv::imwrite(output_filename, plot_img);
}