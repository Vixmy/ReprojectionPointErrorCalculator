# include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#define PAT_ROW     7                   // チェスボードパターン列
#define PAT_COL     10                  // チェスボードパターン行
#define PAT_SIZE    PAT_ROW * PAT_COL   // チェスボードパターンサイズ
#define CAM_W       720                 // カメラ画像幅
#define CAM_H       540                 // カメラ画像高さ


int main() {
    // 画像読み込み
    cv::Mat white, black0, black1, binary0, binary1;
    std::ostringstream file_name;
    file_name << "../img/";
    white = cv::imread(file_name.str() + "proj_white.bmp");
    black0 = cv::imread(file_name.str() + "proj0_black.bmp", 0);
    black1 = cv::imread(file_name.str() + "proj1_black.bmp", 0);

    // 格子点検出
    std::vector<cv::Point2f> corners0, corners;
    cv::Size pattern_size = cv::Size2i(PAT_COL, PAT_ROW);
    bool found = cv::findChessboardCorners(white, pattern_size, corners);

    // 二値化処理
    int threshold = 30;
    cv::threshold(black0, binary0, threshold, 255, 0);
    cv::threshold(black1, binary1, threshold, 255, 0);

    // 輪郭抽出
    std::vector<std::vector<cv::Point>> contours0, contours1;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary0, contours0, hierarchy, 0, 1);
    cv::findContours(binary1, contours1, hierarchy, 0, 1);

    // 輪郭内部塗りつぶし
    std::vector<cv::Point2f> lights0, lights1; // 投影された光の点の座標
    for (int i = 0; i < PAT_SIZE; i++) {
        cv::Mat filled0 = cv::Mat(CAM_H, CAM_W, CV_8UC1, cv::Scalar::all(0));
        cv::Mat filled1 = cv::Mat(CAM_H, CAM_W, CV_8UC1, cv::Scalar::all(0));
        cv::drawContours(filled0, contours0, i, 255, -1);
        cv::drawContours(filled1, contours1, i, 255, -1);

        // 重心計算
        cv::Moments mu0 = moments(filled0, false);
        cv::Moments mu1 = moments(filled1, false);
        cv::Point2f mc0 = cv::Point2f(mu0.m10 / mu0.m00, mu0.m01 / mu0.m00);
        cv::Point2f mc1 = cv::Point2f(mu1.m10 / mu1.m00, mu1.m01 / mu1.m00);

        lights0.push_back(mc0);
        lights1.push_back(mc1);
    }



    // result image
    int w = 640;
    int h = 640;
    cv::Mat result_pix0(cv::Size(w, h), CV_8UC3, cv::Scalar::all(255));
    cv::Mat result_pix1(cv::Size(w, h), CV_8UC3, cv::Scalar::all(255));
    cv::Point2f center = cv::Point2f(w / 2, h / 2);
    for (int i = 0; i < w / 2; i += 50) {
        cv::Scalar color = (i % 250 == 0) ? cv::Scalar(20, 20, 20) : cv::Scalar(150, 150, 150);
        cv::line(result_pix0, cv::Point(w / 2 + i, 0), cv::Point(w / 2 + i, h), color, 1, 4);
        cv::line(result_pix0, cv::Point(w / 2 - i, 0), cv::Point(w / 2 - i, h), color, 1, 4);
        cv::line(result_pix1, cv::Point(w / 2 + i, 0), cv::Point(w / 2 + i, h), color, 1, 4);
        cv::line(result_pix1, cv::Point(w / 2 - i, 0), cv::Point(w / 2 - i, h), color, 1, 4);
    }
    for (int i = 0; i < h / 2; i += 50) {
        cv::Scalar color = (i % 250 == 0) ? cv::Scalar(20, 20, 20) : cv::Scalar(150, 150, 150);
        cv::line(result_pix0, cv::Point(0, h / 2 + i), cv::Point(w, h / 2 + i), color, 1, 4);
        cv::line(result_pix0, cv::Point(0, h / 2 - i), cv::Point(w, h / 2 - i), color, 1, 4);
        cv::line(result_pix1, cv::Point(0, h / 2 + i), cv::Point(w, h / 2 + i), color, 1, 4);
        cv::line(result_pix1, cv::Point(0, h / 2 - i), cv::Point(w, h / 2 - i), color, 1, 4);
    }
    cv::putText(result_pix0, "scale : ?.? pixel", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 50, 200), 2);
    cv::putText(result_pix1, "scale : ?.? pixel", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 50, 200), 2);
    // 誤差計算[pix]
    std::vector<cv::Point2f> error_pix0, error_pix1;
    for (int i = 0; i < PAT_SIZE; i++) {
        error_pix0.push_back(lights0[i] - corners[i]);
        error_pix1.push_back(lights1[i] - corners[i]);
    }
    double err_pix0 = 0.0, err_pix1 = 0.0;
    for (auto p : error_pix0) {
        double norm = cv::norm(p);
        err_pix0 += norm;

        cv::Mat c(1, 1, CV_8UC3, cv::Scalar(norm * 50, 255, 255));
        cv::cvtColor(c, c, cv::COLOR_HSV2RGB);
        cv::Scalar color = cv::Scalar(c.data[0], c.data[1], c.data[2]);
        cv::circle(result_pix0, center + p * 25, 1, color, 3, 4);	// 再投影点
    }
    for (auto p : error_pix1) {
        double norm = cv::norm(p);
        err_pix1 += norm;

        cv::Mat c(1, 1, CV_8UC3, cv::Scalar(norm * 50, 255, 255));
        cv::cvtColor(c, c, cv::COLOR_HSV2RGB);
        cv::Scalar color = cv::Scalar(c.data[0], c.data[1], c.data[2]);
        cv::circle(result_pix1, center + p * 25, 1, color, 3, 4);	// 再投影点
    }
    err_pix0 /= error_pix0.size();
    err_pix1 /= error_pix1.size();
    std::ostringstream str_pix0;
    str_pix0 << " : reprojection error [pixel/point] = " << err_pix0;
    cv::putText(result_pix0, str_pix0.str(), cv::Point(10, 600), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 50, 200), 2);
    cv::imwrite(file_name.str() + "result_pix0.bmp", result_pix0);
    std::ostringstream str_pix1;
    str_pix1 << " : reprojection error [pixel/point] = " << err_pix1;
    cv::putText(result_pix1, str_pix1.str(), cv::Point(10, 600), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 50, 200), 2);
    cv::imwrite(file_name.str() + "result_pix1.bmp", result_pix1);



    // ホモグラフィ計算
    std::vector<cv::Point2f> obj_points;
    for (int i = 0; i < PAT_ROW; i++) {
        for (int j = 0; j < PAT_COL; j++) {
            cv::Point2f p((PAT_COL - j) * 20.0, (PAT_ROW - i) * 20.0);
            obj_points.push_back(p);
        }
    }
    cv::Mat H = cv::findHomography(corners, obj_points);

    // 座標の射影
    std::vector<cv::Point2f> dst_points0, dst_points1;
    cv::perspectiveTransform(lights0, dst_points0, H);
    cv::perspectiveTransform(lights1, dst_points1, H);


    // 誤差計算[mm]
    cv::Mat result_mm0(cv::Size(w, h), CV_8UC3, cv::Scalar::all(255));
    cv::Mat result_mm1(cv::Size(w, h), CV_8UC3, cv::Scalar::all(255));
    for (int i = 0; i < w / 2; i += 50) {
        cv::Scalar color = (i % 250 == 0) ? cv::Scalar(20, 20, 20) : cv::Scalar(150, 150, 150);
        cv::line(result_mm0, cv::Point(w / 2 + i, 0), cv::Point(w / 2 + i, h), color, 1, 4);
        cv::line(result_mm0, cv::Point(w / 2 - i, 0), cv::Point(w / 2 - i, h), color, 1, 4);
        cv::line(result_mm1, cv::Point(w / 2 + i, 0), cv::Point(w / 2 + i, h), color, 1, 4);
        cv::line(result_mm1, cv::Point(w / 2 - i, 0), cv::Point(w / 2 - i, h), color, 1, 4);
    }
    for (int i = 0; i < h / 2; i += 50) {
        cv::Scalar color = (i % 250 == 0) ? cv::Scalar(20, 20, 20) : cv::Scalar(150, 150, 150);
        cv::line(result_mm0, cv::Point(0, h / 2 + i), cv::Point(w, h / 2 + i), color, 1, 4);
        cv::line(result_mm0, cv::Point(0, h / 2 - i), cv::Point(w, h / 2 - i), color, 1, 4);
        cv::line(result_mm1, cv::Point(0, h / 2 + i), cv::Point(w, h / 2 + i), color, 1, 4);
        cv::line(result_mm1, cv::Point(0, h / 2 - i), cv::Point(w, h / 2 - i), color, 1, 4);
    }
    cv::putText(result_mm0, "scale : ?.? mm", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 50, 200), 2);
    cv::putText(result_mm1, "scale : ?.? mm", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 50, 200), 2);
    std::vector<cv::Point2f> error_mm0, error_mm1;
    for (int i = 0; i < PAT_SIZE; i++) {
        error_mm0.push_back(dst_points0[i] - obj_points[i]);
        error_mm1.push_back(dst_points1[i] - obj_points[i]);
    }
    double err_mm0 = 0.0;
    for (auto p : error_mm0) {
        double norm = cv::norm(p);
        err_mm0 += norm;

        cv::Mat c(1, 1, CV_8UC3, cv::Scalar(norm * 50, 255, 255));
        cv::cvtColor(c, c, cv::COLOR_HSV2RGB);
        cv::Scalar color = cv::Scalar(c.data[0], c.data[1], c.data[2]);
        cv::circle(result_mm0, center + p * 50, 1, color, 3, 4);	// 再投影点
    }
    double err_mm1 = 0.0;
    for (auto p : error_mm1) {
        double norm = cv::norm(p);
        err_mm1 += norm;

        cv::Mat c(1, 1, CV_8UC3, cv::Scalar(norm * 50, 255, 255));
        cv::cvtColor(c, c, cv::COLOR_HSV2RGB);
        cv::Scalar color = cv::Scalar(c.data[0], c.data[1], c.data[2]);
        cv::circle(result_mm1, center + p * 50, 1, color, 3, 4);	// 再投影点
    }
    err_mm0 /= error_mm0.size();
    err_mm1 /= error_mm1.size();
    std::ostringstream str_mm0;
    str_mm0 << " : reprojection error [mm/point] = " << err_mm0;
    cv::putText(result_mm0, str_mm0.str(), cv::Point(10, 600), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 50, 200), 2);
    cv::imwrite(file_name.str() + "result_mm0.bmp", result_mm0);
    std::ostringstream str_mm1;
    str_mm1 << " : reprojection error [mm/point] = " << err_mm1;
    cv::putText(result_mm1, str_mm1.str(), cv::Point(10, 600), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 50, 200), 2);
    cv::imwrite(file_name.str() + "result_mm1.bmp", result_mm1);

    // artifact
    /*cv::Mat board_img;
    board_img = cv::imread("C:/kagami/data/rect283-5.png");
    cv::warpPerspective(board_img, board_img, H, cv::Size(1024, 768), CV_WARP_INVERSE_MAP);*/

    // プロジェクタの投影点間の距離
    cv::Mat relative_pix(cv::Size(w, h), CV_8UC3, cv::Scalar::all(255));
    cv::Mat relative_mm(cv::Size(w, h), CV_8UC3, cv::Scalar::all(255));
    for (int i = 0; i < w / 2; i += 50) {
        cv::Scalar color = (i % 250 == 0) ? cv::Scalar(20, 20, 20) : cv::Scalar(150, 150, 150);
        cv::line(relative_pix, cv::Point(w / 2 + i, 0), cv::Point(w / 2 + i, h), color, 1, 4);
        cv::line(relative_pix, cv::Point(w / 2 - i, 0), cv::Point(w / 2 - i, h), color, 1, 4);
        cv::line(relative_mm, cv::Point(w / 2 + i, 0), cv::Point(w / 2 + i, h), color, 1, 4);
        cv::line(relative_mm, cv::Point(w / 2 - i, 0), cv::Point(w / 2 - i, h), color, 1, 4);
    }
    for (int i = 0; i < h / 2; i += 50) {
        cv::Scalar color = (i % 250 == 0) ? cv::Scalar(20, 20, 20) : cv::Scalar(150, 150, 150);
        cv::line(relative_pix, cv::Point(0, h / 2 + i), cv::Point(w, h / 2 + i), color, 1, 4);
        cv::line(relative_pix, cv::Point(0, h / 2 - i), cv::Point(w, h / 2 - i), color, 1, 4);
        cv::line(relative_mm, cv::Point(0, h / 2 + i), cv::Point(w, h / 2 + i), color, 1, 4);
        cv::line(relative_mm, cv::Point(0, h / 2 - i), cv::Point(w, h / 2 - i), color, 1, 4);
    }
    cv::putText(relative_pix, "scale : ?.? pix", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 50, 200), 2);
    cv::putText(relative_mm, "scale : ?.? mm", cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 50, 200), 2);
    std::vector<cv::Point2f> relative_error_pix, relative_error_mm;
    for (int i = 0; i < PAT_SIZE; i++) {
        relative_error_pix.push_back(lights1[i] - lights0[i]);
        relative_error_mm.push_back(dst_points1[i] - dst_points0[i]);
    }
    double rel_err_pix = 0.0, rel_err_mm = 0.0;
    for (auto p : relative_error_pix) {
        double norm = cv::norm(p);
        rel_err_pix += norm;

        cv::Mat c(1, 1, CV_8UC3, cv::Scalar(norm * 50, 255, 255));
        cv::cvtColor(c, c, cv::COLOR_HSV2RGB);
        cv::Scalar color = cv::Scalar(c.data[0], c.data[1], c.data[2]);
        cv::circle(relative_pix, center + p * 50, 1, color, 3, 4);	// 再投影点
    }
    for (auto p : relative_error_mm) {
        double norm = cv::norm(p);
        rel_err_mm += norm;

        cv::Mat c(1, 1, CV_8UC3, cv::Scalar(norm * 50, 255, 255));
        cv::cvtColor(c, c, cv::COLOR_HSV2RGB);
        cv::Scalar color = cv::Scalar(c.data[0], c.data[1], c.data[2]);
        cv::circle(relative_mm, center + p * 50, 1, color, 3, 4);	// 再投影点
    }
    rel_err_pix /= relative_error_pix.size();
    rel_err_mm /= relative_error_mm.size();
    std::ostringstream rel_str_pix, rel_str_mm;
    rel_str_pix << " : relative error [pix/point] = " << rel_err_pix;
    rel_str_mm << " : relative error [mm/point] = " << rel_err_mm;
    cv::putText(relative_pix, rel_str_pix.str(), cv::Point(10, 600), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 50, 200), 2);
    cv::putText(relative_mm, rel_str_mm.str(), cv::Point(10, 600), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 50, 200), 2);
    cv::imwrite(file_name.str() + "relative_pix.bmp", relative_pix);
    cv::imwrite(file_name.str() + "relative_mm.bmp", relative_mm);

    cv::waitKey();

    // xmlファイル出力
    cv::FileStorage fs("rel_err_mm.xml", cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cout << "File can not be opened." << std::endl;
        return -1;
    }

    fs << "err_vec" << "[";
    for (int i = 0; i < (int)relative_error_mm.size(); i++) {
        fs << "{";
        fs << "x" << relative_error_mm[i].x;
        fs << "y" << relative_error_mm[i].y;
        fs << "}";
    }
    fs << "]";

    fs.release();

    return 0;
}