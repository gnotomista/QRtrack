#include <stdlib.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

#define MARKER_LENGTH 0.026
#define AXIS_LENGTH MARKER_LENGTH*0.75f

using namespace std;
using namespace cv;

struct cameraParameters{
  int image_width;
  int image_height;
  cv::Mat camera_matrix;
  cv::Mat distortion_coefficients;
};

// forward declarations
bool find_homography_to_reference_markers_image_plane(
  VideoCapture cam,
  Ptr<aruco::Dictionary> dictionary,
  Ptr<aruco::DetectorParameters> detectorParams,
  const vector< int > REFERENCE_MARKER_IDS,
  const vector< Point2f > reference_markers_image_plane_WORLD_PLANE,
  Mat& H);
static bool read_detector_parameters(
  std::string filename,
  cv::Ptr<cv::aruco::DetectorParameters> &params);
static bool read_camera_parameters(
  std::string filename,
  cameraParameters& camParams
);
cv::Mat point2f_to_homogeneous_mat_point(
  cv::Point2f in
);
cv::Point2f mat_point_to_homogeneous_point2f(
  cv::Mat in
);
Point2f find_marker_center(
  vector< Point2f > corners
);
Point2f map_marker_from_image_to_world(
  vector< Point2f > marker,
  Mat H
);
Point2f map_point_from_image_to_world(
  Point2f marker,
  Mat H
);
float get_marker_orientation(
  vector< Point2f > marker,
  Mat H
);
Point3f get_robot_pose(
  vector< Point2f > marker,
  Mat H
);

// main
int main(int argc, char** argv) {
  const vector< Point2f > reference_markers_image_plane_WORLD_PLANE = {
    Point2f(0,0),
    Point2f(1,0),
    Point2f(1,1),
    Point2f(0,1)};
  const vector< int > REFERENCE_MARKER_IDS = {0, 1, 2, 3};
  vector< Point2f > reference_markers_image_plane_image_plane;

  // camera parameters
  cameraParameters camParams;
  read_camera_parameters("../config/camera_parameters_lifecam3.yaml", camParams);
  cv::Mat camMatrix = camParams.camera_matrix;
  cv::Mat camDistCoeffs = camParams.distortion_coefficients;
  int image_width = camParams.image_width;
  int image_height = camParams.image_height;

  // video variables
  Mat img;
  VideoCapture cam(1);
  cam.set(CV_CAP_PROP_FRAME_WIDTH, image_width);
  cam.set(CV_CAP_PROP_FRAME_HEIGHT, image_height);

  // marker detector parameters
  Ptr<aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(aruco::DICT_4X4_50);
  Ptr<aruco::DetectorParameters> detectorParams = cv::aruco::DetectorParameters::create();
  read_detector_parameters("../config/detector_parameters.yml", detectorParams);
  vector< vector< Point2f > > corners, rejected;
  std::vector< int > ids;

  Mat H;
  find_homography_to_reference_markers_image_plane(cam, dictionary, detectorParams, REFERENCE_MARKER_IDS, reference_markers_image_plane_WORLD_PLANE, H);

  while(waitKey(30) != 27){

    cam >> img;

    aruco::detectMarkers(img, dictionary, corners, ids, detectorParams, rejected);

    for (int i = 0; i < ids.size(); i++)
      cout << ids[i] << " - " << get_robot_pose(corners[i], H) << endl;


    imshow("out", img);
  }

  return 0;
}

bool find_homography_to_reference_markers_image_plane(
  VideoCapture cam,
  Ptr<aruco::Dictionary> dictionary,
  Ptr<aruco::DetectorParameters> detectorParams,
  const vector< int > REFERENCE_MARKER_IDS,
  const vector< Point2f > reference_markers_image_plane_WORLD_PLANE,
  Mat& H) {

  Mat img;
  vector< vector< Point2f > > corners, rejected, reference_markers_image_plane;
  std::vector< int > ids;

  while(waitKey(30) != 27){

    cam >> img;

    aruco::detectMarkers(img, dictionary, corners, ids, detectorParams, rejected);

    if (ids.size() > 0) {

      reference_markers_image_plane.clear();

      for (int i = 0; i < REFERENCE_MARKER_IDS.size(); i++){
        vector< int >::iterator iter = find(ids.begin(), ids.end(), REFERENCE_MARKER_IDS[i]);
        if (iter != ids.end()){
          int idx = distance(ids.begin(), iter);
          reference_markers_image_plane.push_back(corners[idx]);
        }
      }

      if (reference_markers_image_plane.size() == REFERENCE_MARKER_IDS.size()){
        cout << "found them" << endl;
        vector< Point2f > image_points, world_points;
        for (int i = 0; i < REFERENCE_MARKER_IDS.size(); i++){
          image_points.push_back(
            Point2f(
              0.25*(reference_markers_image_plane[i][0].x+reference_markers_image_plane[i][1].x+reference_markers_image_plane[i][2].x+reference_markers_image_plane[i][3].x),
              0.25*(reference_markers_image_plane[i][0].y+reference_markers_image_plane[i][1].y+reference_markers_image_plane[i][2].y+reference_markers_image_plane[i][3].y)
            )
          );
          world_points.push_back(reference_markers_image_plane_WORLD_PLANE[i]);
        }
        cout << "Computing homography between the image point:\n" << image_points << "\nand the world points:\n" << world_points << endl << "...";

        H = findHomography(image_points, world_points);

        cout << "homography:\n" << H << endl;

        return true;
      }

      cv::aruco::drawDetectedMarkers(img, corners, ids);
    }

    imshow("out", img);
  }
}

static bool read_detector_parameters(std::string filename, cv::Ptr<cv::aruco::DetectorParameters> &params) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params->minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
    //fs["cornerRefinementMethod"] >> params->cornerRefinementMethod;
    fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params->markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params->minOtsuStdDev;
    fs["errorCorrectionRate"] >> params->errorCorrectionRate;
    return true;
}

static bool read_camera_parameters(std::string filename, cameraParameters& camParams) {
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  if (!fs.isOpened())
    return false;
  fs["camera_matrix"] >> camParams.camera_matrix;
  fs["distortion_coefficients"] >> camParams.distortion_coefficients;
  fs["image_width"] >> camParams.image_width;
  fs["image_height"] >> camParams.image_height;
}

cv::Mat point2f_to_homogeneous_mat_point(cv::Point2f in)
{
    cv::Mat out(3,1, CV_64FC1);
    out.at<double>(0,0) = in.x;
    out.at<double>(1,0) = in.y;
    out.at<double>(2,0) = 1.0;
    return out;
}

cv::Point2f mat_point_to_homogeneous_point2f(cv::Mat in)
{
    cv::Point2f out;
    out.x = in.at<double>(0,0) / in.at<double>(2,0);
    out.y = in.at<double>(1,0) / in.at<double>(2,0);
    return out;
}


Point2f find_marker_center(vector< Point2f > corners) {
  return Point2f(
    0.25*(corners[0].x+corners[1].x+corners[2].x+corners[3].x),
    0.25*(corners[0].y+corners[1].y+corners[2].y+corners[3].y)
  );
}

Point2f map_marker_from_image_to_world(vector< Point2f > marker, Mat H) {
  Mat homog_world_point, homog_image_point;
  Point2f marker_center;

  // TODO: undistort before homography

  marker_center = find_marker_center(marker);

  homog_image_point = point2f_to_homogeneous_mat_point(marker_center);

  homog_world_point = H*homog_image_point;

  return mat_point_to_homogeneous_point2f(homog_world_point);
}

Point2f map_point_from_image_to_world(Point2f marker, Mat H) {
  Mat homog_world_point, homog_image_point;

  // TODO: undistort before homography

  homog_image_point = point2f_to_homogeneous_mat_point(marker);

  homog_world_point = H*homog_image_point;

  return mat_point_to_homogeneous_point2f(homog_world_point);
}

float get_marker_orientation(vector< Point2f > marker, Mat H){
  vector< Point2f > center_world;
  for (int i = 0; i < 4; i++)
    center_world.push_back(map_point_from_image_to_world(marker[i], H));
  Point2f forward_vector = (center_world[1]+center_world[2]-center_world[0]-center_world[3])/2;
  return atan2(forward_vector.y, forward_vector.x);
}

Point3f get_robot_pose(vector< Point2f > marker, Mat H) {
  Point2f position = map_marker_from_image_to_world(marker, H);
  float orientation = get_marker_orientation(marker, H);
  return Point3f(position.x, position.y, orientation);
}
