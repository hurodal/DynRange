#include "engine.hpp"
#include "functions.hpp"
#include "spline.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <filesystem>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <libraw/libraw.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace fs = std::filesystem;

static DRCalcResult calculate_dr_models(double threshold, const std::vector<double>& snr_db, const std::vector<double>& signal_ev, std::ostream& log_stream) {
    const double filter_range = 5.0;
    DRCalcResult result;
    
    // 1. Filtrar los parches según el criterio (umbral +/- 5dB)
    for (size_t i = 0; i < snr_db.size(); ++i) {
        if (snr_db[i] >= (threshold - filter_range) && snr_db[i] <= (threshold + filter_range)) {
            result.filtered_snr.push_back(snr_db[i]);
            result.filtered_signal.push_back(signal_ev[i]);
        }
    }

    log_stream << "  - Info: For " << threshold << "dB threshold, using " << result.filtered_snr.size() << " patches." << std::endl;

    // 2. Calcular el modelo de Spline con los datos filtrados
    if (result.filtered_snr.size() >= 2) {
        // Spline necesita que los puntos X estén ordenados
        std::vector<size_t> p(result.filtered_snr.size());
        std::iota(p.begin(), p.end(), 0);
        std::sort(p.begin(), p.end(), [&](size_t i, size_t j){ return result.filtered_snr[i] < result.filtered_snr[j]; });
        
        std::vector<double> sorted_snr(result.filtered_snr.size()), sorted_sig(result.filtered_signal.size());
        for(size_t idx = 0; idx < p.size(); ++idx) {
            sorted_snr[idx] = result.filtered_snr[p[idx]]; 
            sorted_sig[idx] = result.filtered_signal[p[idx]];
        }
        
        result.spline_model.set_points(sorted_snr, sorted_sig);
    }

    // 3. Calcular el modelo Polinómico con los datos filtrados
    if (result.filtered_snr.size() >= 3) {
        cv::Mat snr_mat(result.filtered_snr.size(), 1, CV_64F, result.filtered_snr.data());
        cv::Mat signal_mat(result.filtered_signal.size(), 1, CV_64F, result.filtered_signal.data());
        polyfit(snr_mat, signal_mat, result.poly_coeffs, 2);
    }
    
    return result;
}


bool run_dynamic_range_analysis(const ProgramOptions& opts, std::ostream& log_stream) {
    const int NCOLS = 11;
    const int NROWS = 7;
    const double SAFE = 50.0;

    std::vector<DynamicRangeResult> all_results;
    Eigen::VectorXd k;
    
    const auto& filenames = opts.input_files;

    for (int i = 0; i < filenames.size(); ++i) {
        const std::string& name = filenames[i];
        log_stream << "\nProcessing \"" << name << "\"..." << std::endl;

        LibRaw raw_processor;
        if (raw_processor.open_file(name.c_str()) != LIBRAW_SUCCESS) {
            log_stream << "Error: Could not open RAW file: " << name << std::endl;
            return false;
        }
        if (raw_processor.unpack() != LIBRAW_SUCCESS) {
            log_stream << "Error: Could not decode RAW data from: " << name << std::endl;
            return false;
        }
        
        log_stream << "  - Info: Black=" << opts.dark_value << ", Saturation=" << opts.saturation_value << std::endl;

        cv::Mat raw_image(raw_processor.imgdata.sizes.raw_height, raw_processor.imgdata.sizes.raw_width, CV_16U, raw_processor.imgdata.rawdata.raw_image);
        
        cv::Mat img_float;
        raw_image.convertTo(img_float, CV_32F);
        img_float = (img_float - opts.dark_value) / (opts.saturation_value - opts.dark_value);

        cv::Mat imgBayer(raw_processor.imgdata.sizes.raw_height / 2, raw_processor.imgdata.sizes.raw_width / 2, CV_32FC1);
        for (int r = 0; r < imgBayer.rows; ++r) {
            for (int c = 0; c < imgBayer.cols; ++c) {
                imgBayer.at<float>(r, c) = img_float.at<float>(r * 2, c * 2);
            }
        }
        if (i == 0) {
            std::vector<cv::Point2d> xu = {{119, 170}, {99, 1687}, {2515, 1679}, {2473, 158}};
            double xtl = (xu[0].x + xu[1].x) / 2.0; double ytl = (xu[0].y + xu[3].y) / 2.0;
            double xbr = (xu[2].x + xu[3].x) / 2.0; double ybr = (xu[1].y + xu[2].y) / 2.0;
            std::vector<cv::Point2d> xd = {{xtl, ytl}, {xtl, ybr}, {xbr, ybr}, {xbr, ytl}};
            k = calculate_keystone_params(xu, xd);
            log_stream << "  - Keystone parameters calculated." << std::endl;
        }
        cv::Mat imgc = undo_keystone(imgBayer, k);
        double xtl = (119.0 + 99.0) / 2.0; double ytl = (170.0 + 158.0) / 2.0;
        double xbr = (2515.0 + 2473.0) / 2.0; double ybr = (1687.0 + 1679.0) / 2.0;
        cv::Rect crop_area(round(xtl), round(ytl), round(xbr - xtl), round(ybr - ytl));
        cv::Mat imgcrop = imgc(crop_area);
        PatchAnalysisResult patch_data = analyze_patches(imgcrop.clone(), NCOLS, NROWS, SAFE);
        
        if (patch_data.signal.empty()) {
            log_stream << "Warning: No valid patches found for " << name << std::endl;
            continue;
        }

        std::vector<double> snr_db, signal_ev;
        for (size_t j = 0; j < patch_data.signal.size(); ++j) {
            snr_db.push_back(20 * log10(patch_data.signal[j] / patch_data.noise[j]));
            signal_ev.push_back(log2(patch_data.signal[j]));
        }
        
        DRCalcResult result12db = calculate_dr_models(12.0, snr_db, signal_ev, log_stream);
        DRCalcResult result0db  = calculate_dr_models(0.0, snr_db, signal_ev, log_stream);

        double dr_12db = 0.0, dr_0db = 0.0;
        if (opts.use_splines) {
            log_stream << "  - Method: Using Spline Interpolation." << std::endl;
            if (result12db.filtered_snr.size() >= 2) dr_12db = -result12db.spline_model(12.0);
            if (result0db.filtered_snr.size() >= 2)  dr_0db  = -result0db.spline_model(0.0);
        } else {
            log_stream << "  - Method: Using Polynomial Fit (Order 2)." << std::endl;
            if (!result12db.poly_coeffs.empty()) {
                double c2 = result12db.poly_coeffs.at<double>(0);
                double c1 = result12db.poly_coeffs.at<double>(1);
                double c0 = result12db.poly_coeffs.at<double>(2);
                dr_12db = -(c2 * 12.0 * 12.0 + c1 * 12.0 + c0);
            }
            if (!result0db.poly_coeffs.empty()) {
                double c2 = result0db.poly_coeffs.at<double>(0);
                double c1 = result0db.poly_coeffs.at<double>(1);
                double c0 = result0db.poly_coeffs.at<double>(2);
                dr_0db = -(c2 * 0.0 * 0.0 + c1 * 0.0 + c0);
            }
        }
        
        all_results.push_back({name, dr_12db, dr_0db, (int)patch_data.signal.size()});

        fs::path input_path(name);
        std::string stem = input_path.stem().string();
        
        // --- CAMBIO: Generamos tres gráficos por fichero ---

        // Gráfico 1: Vista general (auto-escalado)
        std::string plot_full_filename = stem + "_full_range_plot.png";
        generate_debug_plot(plot_full_filename, "Full Range Analysis for " + stem, snr_db, signal_ev, result12db);
        log_stream << "  - Saved full range plot: " << plot_full_filename << std::endl;
        
        // Gráfico 2: Zoom en el umbral de 12dB
        std::string plot12db_filename = stem + "_12dB_zoom_plot.png";
        generate_debug_plot(plot12db_filename, "Zoom @ 12dB for " + stem, snr_db, signal_ev, result12db, 
                            std::optional<cv::Point2d>{cv::Point2d(7.0, 17.0)});
        log_stream << "  - Saved 12dB zoom plot: " << plot12db_filename << std::endl;

        // Gráfico 3: Zoom en el umbral de 0dB
        std::string plot0db_filename = stem + "_0dB_zoom_plot.png";
        generate_debug_plot(plot0db_filename, "Zoom @ 0dB for " + stem, snr_db, signal_ev, result0db,
                            std::optional<cv::Point2d>{cv::Point2d(-5.0, 5.0)});
        log_stream << "  - Saved 0dB zoom plot: " << plot0db_filename << std::endl;
    }

    log_stream << "\n--- Dynamic Range Results ---\n";
    log_stream << std::left << std::setw(35) << "RAW File"
              << std::setw(15) << "DR (12dB)" << std::setw(15) << "DR (0dB)"
              << "Patches" << std::endl;
    log_stream << std::string(80, '-') << std::endl;
    for (const auto& res : all_results) {
        log_stream << std::left << std::setw(35) << fs::path(res.filename).filename().string()
                  << std::fixed << std::setprecision(4) << std::setw(15) << res.dr_12db
                  << std::fixed << std::setprecision(4) << std::setw(15) << res.dr_0db
                  << res.patches_used << std::endl;
    }
    std::ofstream csv_file(opts.output_filename);
    csv_file << "raw_file,DR_EV_12dB,DR_EV_0dB,patches_used\n";
    for (const auto& res : all_results) {
        csv_file << fs::path(res.filename).filename().string() << "," << res.dr_12db << "," << res.dr_0db << "," << res.patches_used << "\n";
    }
    csv_file.close();
    log_stream << "\nResults saved to " << opts.output_filename << std::endl;

    return true;
}