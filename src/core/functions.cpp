#include "functions.hpp"
#include "spline.h" 
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <libraw/libraw.h>
#include <algorithm>
#include <numeric>
#include <filesystem>

/**
 * @brief Processes a RAW file specified as a dark frame to calculate the black level.
 * @param dark_file_path Path to the dark frame RAW file.
 * @param log_stream Stream for logging output.
 * @return An optional containing the calculated black level, or nullopt on failure.
 */
std::optional<double> process_dark_frame(const std::string& dark_file_path, std::ostream& log_stream) {
    LibRaw raw_processor;
    if (raw_processor.open_file(dark_file_path.c_str()) != LIBRAW_SUCCESS) {
        log_stream << "Error: Could not open dark frame file: " << dark_file_path << std::endl;
        return std::nullopt;
    }
    if (raw_processor.unpack() != LIBRAW_SUCCESS) {
        log_stream << "Error: Could not decode dark frame data from: " << dark_file_path << std::endl;
        return std::nullopt;
    }
    cv::Mat raw_image(raw_processor.imgdata.sizes.raw_height, raw_processor.imgdata.sizes.raw_width, CV_16U, raw_processor.imgdata.rawdata.raw_image);
    return cv::mean(raw_image)[0];
}

/**
 * @brief Processes a RAW file specified as a saturation frame to determine the saturation point.
 * @param sat_file_path Path to the saturation frame RAW file.
 * @param log_stream Stream for logging output.
 * @return An optional containing the saturation level, or nullopt on failure.
 */
std::optional<double> process_saturation_frame(const std::string& sat_file_path, std::ostream& log_stream) {
    LibRaw raw_processor;
    if (raw_processor.open_file(sat_file_path.c_str()) != LIBRAW_SUCCESS) {
        log_stream << "Error: Could not open saturation frame file: " << sat_file_path << std::endl;
        return std::nullopt;
    }
    if (raw_processor.unpack() != LIBRAW_SUCCESS) {
        log_stream << "Error: Could not decode saturation frame data from: " << sat_file_path << std::endl;
        return std::nullopt;
    }
    return static_cast<double>(raw_processor.imgdata.color.maximum);
}

/**
 * @brief Sorts the input files based on their average brightness.
 * @param opts ProgramOptions struct containing the list of input files, which will be modified.
 * @param log_stream Stream for logging output.
 * @return True on success, false on failure if a file cannot be read.
 */
bool prepare_and_sort_files(ProgramOptions& opts, std::ostream& log_stream) {
    log_stream << "Sorting input files by brightness..." << std::endl;

    // 1. Crear un vector de pares para almacenar (brillo, nombre_de_fichero)
    std::vector<std::pair<double, std::string>> files_with_brightness;

    // 2. Iterar sobre cada fichero para leer su brillo medio
    for (const auto& filename : opts.input_files) {
        LibRaw raw_processor;
        if (raw_processor.open_file(filename.c_str()) != LIBRAW_SUCCESS) {
            log_stream << "  - Error: Could not open file for sorting: " << filename << std::endl;
            // Opcional: podríamos devolver 'false' aquí si un fichero es crítico
            continue; // Omitir este fichero y continuar con los demás
        }
        if (raw_processor.unpack() != LIBRAW_SUCCESS) {
            log_stream << "  - Error: Could not decode file for sorting: " << filename << std::endl;
            continue;
        }

        // Calcular el brillo medio del sensor completo
        cv::Mat raw_image(raw_processor.imgdata.sizes.raw_height, raw_processor.imgdata.sizes.raw_width, CV_16U, raw_processor.imgdata.rawdata.raw_image);
        double brightness = cv::mean(raw_image)[0];
        
        files_with_brightness.push_back({brightness, filename});
        log_stream << "  - File: " << std::filesystem::path(filename).filename().string() << ", Brightness: " << brightness << std::endl;
    }

    // 3. Ordenar el vector de pares basándose en el brillo (el primer elemento del par)
    std::sort(files_with_brightness.begin(), files_with_brightness.end(), 
              [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // 4. Actualizar la lista de ficheros en opts con la nueva lista ordenada
    opts.input_files.clear();
    for (const auto& pair : files_with_brightness) {
        opts.input_files.push_back(pair.second);
    }

    log_stream << "Input files successfully sorted by brightness." << std::endl;
    return true;
}

/**
 * @brief Calculates keystone correction parameters.
 * @param src Vector of source points.
 * @param dst Vector of destination points.
 * @return Eigen::VectorXd containing the transformation parameters.
 */
Eigen::VectorXd calculate_keystone_params(const std::vector<cv::Point2d>& src, const std::vector<cv::Point2d>& dst) {
    Eigen::MatrixXd A(8, 8);
    Eigen::VectorXd b(8);
    for (int i = 0; i < 4; ++i) {
        A(2 * i, 0) = src[i].x; A(2 * i, 1) = src[i].y; A(2 * i, 2) = 1; A(2 * i, 3) = 0; A(2 * i, 4) = 0; A(2 * i, 5) = 0;
        A(2 * i, 6) = -src[i].x * dst[i].x; A(2 * i, 7) = -src[i].y * dst[i].x;
        b(2 * i) = dst[i].x;
        A(2 * i + 1, 0) = 0; A(2 * i + 1, 1) = 0; A(2 * i + 1, 2) = 0; A(2 * i + 1, 3) = src[i].x; A(2 * i + 1, 4) = src[i].y; A(2 * i + 1, 5) = 1;
        A(2 * i + 1, 6) = -src[i].x * dst[i].y; A(2 * i + 1, 7) = -src[i].y * dst[i].y;
        b(2 * i + 1) = dst[i].y;
    }
    return A.colPivHouseholderQr().solve(b);
}

/**
 * @brief Applies keystone correction to an image.
 * @param img The input image.
 * @param k The keystone parameters.
 * @return The corrected image.
 */
cv::Mat undo_keystone(const cv::Mat& img, const Eigen::VectorXd& k) {
    cv::Mat map_x(img.size(), CV_32FC1);
    cv::Mat map_y(img.size(), CV_32FC1);
    for (int j = 0; j < img.rows; ++j) {
        for (int i = 0; i < img.cols; ++i) {
            double den = k(6) * i + k(7) * j + 1;
            map_x.at<float>(j, i) = (float)((k(0) * i + k(1) * j + k(2)) / den);
            map_y.at<float>(j, i) = (float)((k(3) * i + k(4) * j + k(5)) / den);
        }
    }
    cv::Mat img_remap;
    cv::remap(img, img_remap, map_x, map_y, cv::INTER_LINEAR);
    return img_remap;
}

/**
 * @brief Analyzes patches in the image to find signal and noise values.
 * @param img The input image.
 * @param NCOLS Number of columns in the patch grid.
 * @param NROWS Number of rows in the patch grid.
 * @param SAFE Size of the square patch area.
 * @return A PatchAnalysisResult struct containing signal and noise vectors.
 */
PatchAnalysisResult analyze_patches(const cv::Mat& img, int NCOLS, int NROWS, double SAFE) {
    PatchAnalysisResult result;
    double H = img.rows, W = img.cols;
    for (int j = 0; j < NROWS; ++j) {
        for (int i = 0; i < NCOLS; ++i) {
            double xc = W * (i + 0.5) / NCOLS;
            double yc = H * (j + 0.5) / NROWS;
            cv::Rect roi((int)(xc - SAFE / 2), (int)(yc - SAFE / 2), (int)SAFE, (int)SAFE);
            if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > img.cols || roi.y + roi.height > img.rows) continue;
            
            cv::Mat patch = img(roi);
            cv::Scalar mean, stddev;
            cv::meanStdDev(patch, mean, stddev);
            
            if (mean[0] > 0.95 || mean[0] < 0.001) continue;
            
            result.signal.push_back(mean[0]);
            result.noise.push_back(stddev[0]);
        }
    }
    return result;
}

/**
 * @brief Performs a least-squares polynomial fit.
 * @param src_x Input X-coordinates (must be a single column cv::Mat).
 * @param src_y Input Y-coordinates (must be a single column cv::Mat).
 * @param dst Output cv::Mat for the polynomial coefficients.
 * @param order The order of the polynomial to fit.
 */
void polyfit(const cv::Mat& src_x, const cv::Mat& src_y, cv::Mat& dst, int order)
{
    CV_Assert(src_x.rows > 0 && src_y.rows > 0 && src_x.total() == src_y.total() && src_x.rows >= order + 1);

    cv::Mat A = cv::Mat::zeros(src_x.rows, order + 1, CV_64F);

    for (int i = 0; i < src_x.rows; ++i) {
        for (int j = 0; j <= order; ++j) {
            A.at<double>(i, j) = std::pow(src_x.at<double>(i), j);
        }
    }
    
    cv::Mat A_flipped;
    cv::flip(A, A_flipped, 1);

    cv::solve(A_flipped, src_y, dst, cv::DECOMP_SVD);
}

/**
 * @brief Generates a debug plot comparing data points, spline, and polynomial fit.
 * @param output_filename The path to save the output PNG image.
 * @param plot_title The title to display on the plot.
 * @param all_snr_db Vector containing all SNR data points.
 * @param all_signal_ev Vector containing all Signal data points.
 * @param result The DRCalcResult struct containing filtered data and calculated models.
 * @param x_range Optional cv::Point2d(min, max) for the X-axis range.
 * @param y_range Optional cv::Point2d(min, max) for the Y-axis range.
 * @param width The width of the output image in pixels.
 * @param height The height of the output image in pixels.
 */
void generate_debug_plot(
    const std::string& output_filename,
    const std::string& plot_title,
    const std::vector<double>& all_snr_db,
    const std::vector<double>& all_signal_ev,
    const DRCalcResult& result,
    const std::optional<cv::Point2d>& x_range,
    const std::optional<cv::Point2d>& y_range,
    int width,
    int height)
{
    const int margin = 80;
    cv::Mat plot_img(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    double min_snr, max_snr, min_sig, max_sig;

    if (x_range) {
        min_snr = x_range->x;
        max_snr = x_range->y;
    } else {
        if (all_snr_db.empty()) { min_snr = 0; max_snr = 40; }
        else {
            auto minmax_x = std::minmax_element(all_snr_db.begin(), all_snr_db.end());
            min_snr = *minmax_x.first - 2.0;
            max_snr = *minmax_x.second + 2.0;
        }
    }

    if (y_range) {
        min_sig = y_range->x;
        max_sig = y_range->y;
    } else {
        if (all_signal_ev.empty()) { min_sig = -12; max_sig = 0; }
        else {
            auto minmax_y = std::minmax_element(all_signal_ev.begin(), all_signal_ev.end());
            min_sig = *minmax_y.first - 1.0;
            max_sig = *minmax_y.second + 1.0;
        }
    }

    auto to_pixel = [&](double snr, double sig) {
        int px = margin + (int)((snr - min_snr) / (max_snr - min_snr) * (width - 2 * margin));
        int py = margin + (int)((max_sig - sig) / (max_sig - min_sig) * (height - 2 * margin));
        return cv::Point(px, py);
    };

    cv::line(plot_img, {margin, margin}, {margin, height - margin}, cv::Scalar(150,150,150), 2);
    cv::line(plot_img, {margin, height - margin}, {width - margin, height - margin}, cv::Scalar(150,150,150), 2);
    cv::putText(plot_img, "SNR (dB)", {width/2 - 50, height - 25}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,0,0}, 2);
    cv::putText(plot_img, "Signal (EV)", {20, height/2}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,0,0}, 2, cv::LINE_AA, true);
    cv::putText(plot_img, plot_title, {width/2 - (int)(plot_title.length()*12), 50}, cv::FONT_HERSHEY_SIMPLEX, 1.2, {0,0,0}, 2);

    for (size_t i = 0; i < all_snr_db.size(); ++i) {
        cv::circle(plot_img, to_pixel(all_snr_db[i], all_signal_ev[i]), 5, cv::Scalar(200, 200, 200), -1);
    }
    for (size_t i = 0; i < result.filtered_snr.size(); ++i) {
        cv::circle(plot_img, to_pixel(result.filtered_snr[i], result.filtered_signal[i]), 7, cv::Scalar(0, 0, 0), -1);
    }

    const double step = (max_snr - min_snr) / 2000.0;
    if (result.filtered_snr.size() >= 2) {
        for (double x = result.filtered_snr.front(); x <= result.filtered_snr.back(); x += step) {
            cv::Point p1 = to_pixel(x, result.spline_model(x));
            cv::Point p2 = to_pixel(x + step, result.spline_model(x + step));
            cv::line(plot_img, p1, p2, cv::Scalar(0, 0, 255), 3);
        }
    }
    if (!result.poly_coeffs.empty()) {
        double c2 = result.poly_coeffs.at<double>(0);
        double c1 = result.poly_coeffs.at<double>(1);
        double c0 = result.poly_coeffs.at<double>(2);
        for (double x = min_snr; x <= max_snr; x += step) {
            double y = c2 * x * x + c1 * x + c0;
            double y_next = c2 * (x + step) * (x + step) + c1 * (x + step) + c0;
            cv::line(plot_img, to_pixel(x, y), to_pixel(x + step, y_next), cv::Scalar(255, 0, 0), 3);
        }
    }

    cv::circle(plot_img, {width - 220, 70}, 7, {0,0,0}, -1);
    cv::putText(plot_img, "Filtered Data", {width - 190, 75}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,0,0}, 2);
    cv::line(plot_img, {width - 220, 110}, {width - 200, 110}, {255,0,0}, 3);
    cv::putText(plot_img, "Polynomial Fit", {width - 190, 115}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,0,0}, 2);
    cv::line(plot_img, {width - 220, 150}, {width - 200, 150}, {0,0,255}, 3);
    cv::putText(plot_img, "Spline", {width - 190, 155}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0,0,0}, 2);

    cv::imwrite(output_filename, plot_img);
}
