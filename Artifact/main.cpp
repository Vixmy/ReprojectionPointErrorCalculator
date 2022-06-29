#include <iostream>
#include <opencv2/opencv.hpp>

#define USE_BASLER	1
#define USE_DYNA_V1	1
#define USE_DYNA_V3 0

#if USE_DYNA_V1 || USE_DYNA_V1
#include <HighSpeedProjector.h>
#include <ProjectorUtility.h>
#endif

#if USE_BASLER
#include <HSC/baslerClass.hpp>
#pragma comment( lib, "BaslerLib")
#endif

#define CWIDTH		720
#define CHEIGHT		540
#define PWIDTH		1024
#define PHEIGHT		768
#define CFPS		300

#define PAT_ROW		7
#define PAT_COL		10
#define PAT_SIZE	20.0

int main() {
	bool flag = true;
	int captureFrame = 0;

	cv::Mat capImg = cv::Mat(cv::Size(CWIDTH, CHEIGHT), CV_8UC1, cv::Scalar::all(255));
	cv::Mat cornersImg = cv::Mat(cv::Size(CWIDTH, CHEIGHT), CV_8UC3, cv::Scalar::all(0));
	cv::Mat reprojectImgCam = cv::Mat(cv::Size(CWIDTH, CHEIGHT), CV_8UC3, cv::Scalar::all(0));
	cv::Mat output0 = cv::Mat(cv::Size(PWIDTH, PHEIGHT), CV_8UC3, cv::Scalar::all(0));
	cv::Mat output1 = cv::Mat(cv::Size(PWIDTH, PHEIGHT), CV_8UC3, cv::Scalar::all(0));

	cv::Mat Intrin_Cam = cv::Mat::eye(4, 4, CV_64F), Extrin_Cam = cv::Mat::eye(4, 4, CV_64F), distCoef_Cam = cv::Mat::eye(1, 5, CV_64F);
	{
		cv::FileStorage fs("../calib/_cam.xml", cv::FileStorage::READ);
		if (fs.isOpened()) {
			fs["intrinsic"] >> Intrin_Cam;
			fs["distCoefs"] >> distCoef_Cam;
			fs["extrinsic"] >> Extrin_Cam;
			fs.release();
		}
		else {
			std::cout << "cam.xml can not be opened." << std::endl;
		}
	}

	cv::Mat Intrin_Pro0 = cv::Mat::eye(4, 4, CV_64F), Extrin_Pro0 = cv::Mat::eye(4, 4, CV_64F), distCoef_Pro = cv::Mat::eye(1, 5, CV_64F);
	{
		cv::FileStorage fs("../calib/_pro1.xml", cv::FileStorage::READ);
		if (fs.isOpened()) {
			fs["intrinsic"] >> Intrin_Pro0;
			fs["distCoefs"] >> distCoef_Pro;
			fs["extrinsicInv"] >> Extrin_Pro0;
			fs.release();
		}
		else {
			std::cout << "cam.xml can not be opened." << std::endl;
		}
	}

	cv::Mat Intrin_Pro1 = cv::Mat::eye(4, 4, CV_64F), Extrin_Pro1 = cv::Mat::eye(4, 4, CV_64F);
	{
		cv::FileStorage fs("../calib/_pro0.xml", cv::FileStorage::READ);
		if (fs.isOpened()) {
			fs["intrinsic"] >> Intrin_Pro1;
			fs["distCoefs"] >> distCoef_Pro;
			fs["extrinsicInv"] >> Extrin_Pro1;
			fs.release();
		}
		else {
			std::cout << "cam.xml can not be opened." << std::endl;
		}
	}

	std::thread thrRender([&] {

		cv::Mat gray = cv::Mat(cv::Size(CWIDTH, CHEIGHT), CV_8UC1, cv::Scalar::all(100));
		cv::Size patternSize = cv::Size2i(PAT_COL, PAT_ROW);

		while (flag) {

			std::vector<cv::Point2f> cornersCam;

			bool found = cv::findChessboardCorners(capImg, patternSize, cornersCam, cv::CALIB_CB_FAST_CHECK);

			if (found) {

				// draw corners
				{
					cv::cvtColor(capImg, cornersImg, cv::COLOR_GRAY2BGR);
					cv::drawChessboardCorners(cornersImg, patternSize, cornersCam, found);
				}

				// setting 3d points
				std::vector<cv::Point3d> points;
				for (int j = 0; j < PAT_ROW; j++) {
					for (int k = 0; k < PAT_COL; k++) {
						cv::Point3d p(k * PAT_SIZE - PAT_COL * PAT_SIZE / 2, j * PAT_SIZE - PAT_ROW * PAT_SIZE / 2, 0.0);
						points.push_back(p);
					}
				}

				cv::Mat w2pRt0;
				cv::Mat w2pRt1;

				// draw reprojection points from camera
				{
					std::vector<cv::Point2d> reprojectionPoints;
					cv::Mat rvec, tvec;

					cv::Mat camMat = Intrin_Cam(cv::Range(0, 3), cv::Range(0, 3));
					cv::solvePnP(points, cornersCam, camMat, distCoef_Cam, rvec, tvec);

					cv::cvtColor(capImg, reprojectImgCam, cv::COLOR_GRAY2BGR);
					cv::projectPoints(points, rvec, tvec, camMat, distCoef_Cam, reprojectionPoints);

					for (int c = 0; c < PAT_COL; c++) {
						for (int r = 0; r < PAT_ROW; r++) {
							cv::circle(reprojectImgCam, reprojectionPoints[c * PAT_ROW + r], 2, cv::Scalar::all(255), -1, cv::LINE_AA);
						}
					}

					cv::Mat R;
					cv::Rodrigues(rvec, R);

					cv::Mat Rt = (cv::Mat_<double>(4, 4) <<
						R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), tvec.at<double>(0, 0),
						R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), tvec.at<double>(1, 0),
						R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), tvec.at<double>(2, 0),
						0.0, 0.0, 0.0, 1.0
						);

					w2pRt0 = Extrin_Pro0 * Rt;
					w2pRt1 = Extrin_Pro1 * Rt;

				}

				// draw inversed chessboard from projector
				{

					cv::Mat w2pR0 = (cv::Mat_<double>(3, 3) <<
						w2pRt0.at<double>(0, 0), w2pRt0.at<double>(0, 1), w2pRt0.at<double>(0, 2),
						w2pRt0.at<double>(1, 0), w2pRt0.at<double>(1, 1), w2pRt0.at<double>(1, 2),
						w2pRt0.at<double>(2, 0), w2pRt0.at<double>(2, 1), w2pRt0.at<double>(2, 2)
						);

					cv::Mat w2pR1 = (cv::Mat_<double>(3, 3) <<
						w2pRt1.at<double>(0, 0), w2pRt1.at<double>(0, 1), w2pRt1.at<double>(0, 2),
						w2pRt1.at<double>(1, 0), w2pRt1.at<double>(1, 1), w2pRt1.at<double>(1, 2),
						w2pRt1.at<double>(2, 0), w2pRt1.at<double>(2, 1), w2pRt1.at<double>(2, 2)
						);

					cv::Mat w2prvec0;
					cv::Rodrigues(w2pR0, w2prvec0);

					cv::Mat w2prvec1;
					cv::Rodrigues(w2pR1, w2prvec1);

					cv::Mat w2ptvec0 = (cv::Mat_<double>(3, 1) <<
						w2pRt0.at<double>(0, 3),
						w2pRt0.at<double>(1, 3),
						w2pRt0.at<double>(2, 3)
						);

					cv::Mat w2ptvec1 = (cv::Mat_<double>(3, 1) <<
						w2pRt1.at<double>(0, 3),
						w2pRt1.at<double>(1, 3),
						w2pRt1.at<double>(2, 3)
						);

					std::vector<cv::Point2d> points_2d;
					for (int j = 0; j < PAT_ROW; j++) {
						for (int k = 0; k < PAT_COL; k++) {
							cv::Point2d p((PAT_COL - k) * 20.0, (PAT_ROW - j) * 20.0);
							points_2d.push_back(p);
						}
					}
					std::vector<cv::Point2d> reprojectionPoints0, reprojectionPoints1;
					cv::Mat camMat0 = Intrin_Pro0(cv::Range(0, 3), cv::Range(0, 3));
					cv::Mat camMat1 = Intrin_Pro1(cv::Range(0, 3), cv::Range(0, 3));
					cv::projectPoints(points, w2prvec0, w2ptvec0, camMat0, distCoef_Pro, reprojectionPoints0);
					cv::projectPoints(points, w2prvec1, w2ptvec1, camMat1, distCoef_Pro, reprojectionPoints1);
					cv::Mat H0 = cv::findHomography(points_2d, reprojectionPoints0);
					cv::Mat H1 = cv::findHomography(points_2d, reprojectionPoints1);

					cv::Mat board_img;
					board_img = cv::imread("../img/inv_board.png");
					cv::warpPerspective(board_img, output0, H0, cv::Size(1024, 768));
					cv::warpPerspective(board_img, output1, H1, cv::Size(1024, 768));
				}

			}

		}
	});

	std::thread thrCap([&] {
#if USE_BASLER
		basler cam;
		float gain = 0.0f;

		cam.connect(0);
		cam.setParam(paramTypeCamera::paramInt::WIDTH, CWIDTH);
		cam.setParam(paramTypeCamera::paramInt::HEIGHT, CHEIGHT);
		cam.setParam(paramTypeCamera::paramFloat::FPS, CFPS);
		cam.setParam(paramTypeCamera::paramFloat::GAIN, gain);
		cam.setParam(paramTypeBasler::Param::ExposureTime, 1000000.0 / CFPS - 100.0);

		//cam.setParam(paramTypeBasler::FastMode::SensorReadoutModeFast);
		cam.setParam(paramTypeBasler::GrabStrategy::OneByOne);
		//cam.setParam(paramTypeBasler::CaptureType::BayerBGGrab);
		// cam.setParam(paramTypeBasler::CaptureType::ColorBGRGrab);
		cam.setParam(paramTypeBasler::CaptureType::MonocroGrab);
		cam.parameter_all_print();

		cv::Mat tmp = cv::Mat(cv::Size(CWIDTH, CHEIGHT), CV_8UC1, cv::Scalar::all(100));

		cam.start();

		while (flag) {
			if (cam.captureFrame(tmp.data) > 0) captureFrame++;
			std::memcpy(capImg.data, tmp.data, sizeof(unsigned char) * CWIDTH * CHEIGHT);
		}

		cam.stop();
		cam.disconnect();
#endif
	});

	std::thread thrPro([&] {
#if USE_DYNA_V1
		try {
			Sleep(2000);
			HighSpeedProjector proj0_V1;
			HighSpeedProjector proj1_V1;
			DYNAFLASH_PARAM param = getDefaultDynaParamGray();
			param.dFrameRate = CFPS;
			param.nMirrorMode = 1;
			printDynaParam(param);

			proj0_V1.connect(0);
			proj1_V1.connect(1);
			proj0_V1.setParam(param);
			proj1_V1.setParam(param);
			proj0_V1.start();
			proj1_V1.start();
			cv::Mat cdata0;
			cv::Mat cdata1;

			cv::Mat white = cv::Mat(cv::Size(PWIDTH, PHEIGHT), CV_8UC3, cv::Scalar::all(100));
			int count = 0;
			while (flag) {
				cv::cvtColor(output0, cdata0, cv::COLOR_RGB2GRAY);
				cv::cvtColor(output1, cdata1, cv::COLOR_RGB2GRAY);
				proj0_V1.sendImage(cdata0.data);
				proj1_V1.sendImage(cdata1.data);
			}
			proj0_V1.stop();
			proj1_V1.stop();
			proj0_V1.disconnect();
			proj1_V1.disconnect();
		}
		catch (std::exception& e) {
			std::cout << "\033[41m ERROR \033[49m\033[31m thrProj : " << e.what() << "\033[39m" << std::endl;
		}
#endif
	});

	while (flag) {
		cv::imshow("camera", capImg);
		cv::imshow("corners", cornersImg);
		cv::imshow("reprojection (cam)", reprojectImgCam);
		cv::imshow("inv_board0", output0);
		cv::imshow("inv_board1", output1);

		int key = cv::waitKey(1);

		switch (key) {
			case 'q':
				flag = false;
			case 's':
				cv::imwrite("../img/artifact.bmp", capImg);
		}
	}

	thrRender.join();
	thrCap.join();
	thrPro.join();

	return 0;
}