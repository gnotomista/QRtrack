#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

#define MARKER_LENGTH 0.026
#define AXIS_LENGTH MARKER_LENGTH*0.75f
#define REFERENCE_MARKER_ID 22

struct cameraParameters{
  int image_width;
  int image_height;
  cv::Mat camera_matrix;
  cv::Mat distortion_coefficients;
};

bool calculatePoseWrtReferenceMarker(
  std::vector< int > ids,
  std::vector< cv::Vec3d > rvecs,
  std::vector< cv::Vec3d > tvecs,
  std::vector< cv::Vec3d >& rvecs_wrt_ref,
  std::vector< cv::Vec3d >& tvecs_wrt_ref
);
void printStuffs(
  std::vector< int > ids,
  std::vector< cv::Vec3d > rvecs,
  std::vector< cv::Vec3d > tvecs
);
void printStuffsWithRef(
  std::vector< int > ids,
  std::vector< cv::Vec3d > rvecs,
  std::vector< cv::Vec3d > tvecs,
  std::vector< cv::Vec3d > rvecs_wrt_ref,
  std::vector< cv::Vec3d > tvecs_wrt_ref
);
void printStuffsWithRefNice(
  std::vector< int > ids,
  std::vector< cv::Vec3d > rvecs,
  std::vector< cv::Vec3d > tvecs,
  std::vector< cv::Vec3d > rvecs_wrt_ref,
  std::vector< cv::Vec3d > tvecs_wrt_ref
);
static bool readCameraParameters(
  std::string filename,
  cameraParameters& camParams
);
static bool readDetectorParameters(
  std::string filename,
  cv::Ptr<cv::aruco::DetectorParameters> &params
);
cv::Mat vec3dToMat(
  cv::Vec3d in
);
std::vector<int> sortVector(
  std::vector<int> in, std::vector<int>& out
);
std::vector<int> sortVector(
  std::vector<int> in
);

// -----------------------------------------------------------------------------
// fixed transformation to the origin
// -----------------------------------------------------------------------------
// pose of the 'reference marker reference system' in the 'origin reference system'
// i.e. if:   r = reference marker reference system
//            o = origin reference system
//      then: r = R0vec * o + t0
//            (notice that this means translate first of t0 and then rotate of R0vec)
//                        ---------------------------
// Assuming that we want the following origin reference system:
//    - x origin parallel to x camera
//    - y origin antiparallel to y camera
//    - z origin == -z camera
// R0vec and t0 can be easily autodetected from the pose of the reference marker
cv::Mat R0vec = (cv::Mat_<double>(3,1) << 0, 0, 0);
cv::Mat t0 = (cv::Mat_<double>(3,1) << 0.65, -0.33, 0);
// -----------------------------------------------------------------------------

cv::Mat tinv_cv, Rinvvec_cv;
bool recorded_reference_marker_pose = false;
   
int main(int argc, char** argv) {

    // camera parameters
    cameraParameters camParams;
    readCameraParameters("../config/camera_parameters_lifecam1.yaml", camParams);
    cv::Mat camMatrix = camParams.camera_matrix;
    cv::Mat camDistCoeffs = camParams.distortion_coefficients;
    int image_width = camParams.image_width;
    int image_height = camParams.image_height;
    //std::cout << camMatrix << camDistCoeffs << image_height << image_width << std::endl;

    // video variables
    cv::VideoCapture cam;
    cv::Mat img;

    // marker detector parameters
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    cv::Ptr<cv::aruco::DetectorParameters> detectorParams = cv::aruco::DetectorParameters::create();
    readDetectorParameters("../config/detector_parameters.yml", detectorParams);

    // markers parameters
    std::vector< cv::Vec3d > rvecs, tvecs, rvecs_wrt_ref, tvecs_wrt_ref;
    std::vector< int > ids;
    std::vector< std::vector< cv::Point2f > > corners, rejected;

    // open camera and setup parameters
    cam.open(1);
    cam.set(CV_CAP_PROP_FRAME_WIDTH, image_width);
    cam.set(CV_CAP_PROP_FRAME_HEIGHT, image_height);
    //cam.set(cv::CAP_PROP_AUTOFOCUS, true);
    //cam.set(cv::CAP_PROP_AUTO_EXPOSURE, true);
    //cam.set(cv::CAP_PROP_BRIGHTNESS, -100);
    //std::cout << cam.get(cv::CAP_PROP_AUTOFOCUS);

    while (true) {

        cam >> img;

        cv::aruco::detectMarkers(img, dictionary, corners, ids, detectorParams, rejected);

        if (ids.size() > 0) {
          cv::aruco::estimatePoseSingleMarkers(corners, MARKER_LENGTH, camMatrix, camDistCoeffs, rvecs, tvecs);
          cv::aruco::drawDetectedMarkers(img, corners, ids);
          for (int i = 0; i < ids.size(); i++) {
            cv::aruco::drawAxis(img, camMatrix, camDistCoeffs, rvecs[i], tvecs[i], AXIS_LENGTH);
          }

          if (calculatePoseWrtReferenceMarker(ids, rvecs, tvecs, rvecs_wrt_ref, tvecs_wrt_ref)) {
            //printStuffsWithRef(ids, rvecs, tvecs, rvecs_wrt_ref, tvecs_wrt_ref);
            printStuffsWithRefNice(ids, rvecs, tvecs, rvecs_wrt_ref, tvecs_wrt_ref);
          } else {
            printStuffs(ids, rvecs, tvecs);
          }
        }

        cv::imshow("img", img);

        if (cv::waitKey(100) == 27) {
          cam.release();
          break;
        }

    }

}

bool calculatePoseWrtReferenceMarker(
                              std::vector< int > ids,
                              std::vector< cv::Vec3d > rvecs,
                              std::vector< cv::Vec3d > tvecs,
                              std::vector< cv::Vec3d >& rvecs_wrt_ref,
                              std::vector< cv::Vec3d >& tvecs_wrt_ref) {
  bool foundReferenceMarker = false;
  int idReferenceMarker;
  for (int i = 0; i < ids.size(); i++) {
    if (ids[i] == REFERENCE_MARKER_ID){
      foundReferenceMarker = true;
      idReferenceMarker = i;
      break;
    }
  }
  if (foundReferenceMarker) {
    rvecs_wrt_ref.clear();
    tvecs_wrt_ref.clear();

    for (int i = 0; i < ids.size(); i++) {
      if (!recorded_reference_marker_pose) {
          cv::Mat R_cv;
          cv::Rodrigues(rvecs[idReferenceMarker], R_cv);
          cv::Mat t_cv;
          t_cv = vec3dToMat(tvecs[idReferenceMarker]);
          cv::Mat Rinv_cv;
          cv::transpose(R_cv, Rinv_cv);
          //cv::Mat tinv_cv;
          tinv_cv = -Rinv_cv*t_cv;
          //cv::Mat Rinvvec_cv;
          cv::Rodrigues(Rinv_cv, Rinvvec_cv);    
          recorded_reference_marker_pose = true;
      }
      //cv::Mat R_cv;
      //cv::Rodrigues(rvecs[idReferenceMarker], R_cv);
      //cv::Mat t_cv;
      //t_cv = vec3dToMat(tvecs[idReferenceMarker]);
      //cv::Mat Rinv_cv;
      //cv::transpose(R_cv, Rinv_cv);
      //cv::Mat tinv_cv;
      //tinv_cv = -Rinv_cv*t_cv;
      //cv::Mat Rinvvec_cv;
      //cv::Rodrigues(Rinv_cv, Rinvvec_cv);

      cv::Mat rvec_ref, tvec_ref;
      cv::composeRT(rvecs[i], tvecs[i], Rinvvec_cv, tinv_cv, rvec_ref, tvec_ref);

      // same thing as before but for the reference-origin transformation
      // cv::Rodrigues(R0vec, R_cv);
      // t_cv = t0;
      // cv::transpose(R_cv, Rinv_cv);
      // tinv_cv = -Rinv_cv*t_cv;
      // cv::Rodrigues(Rinv_cv, Rinvvec_cv);

      cv::Mat o_R_m, o_t_m;
      // cv::composeRT(Rinvvec_cv, tinv_cv, rvec_ref, tvec_ref, o_R_m, o_t_m);
      cv::composeRT(rvec_ref, tvec_ref, R0vec, t0, o_R_m, o_t_m);

      rvecs_wrt_ref.push_back(o_R_m);
      tvecs_wrt_ref.push_back(o_t_m);
    }
  }
  return foundReferenceMarker;
}

void printStuffsWithRefNice(
                  std::vector< int > ids,
                  std::vector< cv::Vec3d > rvecs,
                  std::vector< cv::Vec3d > tvecs,
                  std::vector< cv::Vec3d > rvecs_wrt_ref,
                  std::vector< cv::Vec3d > tvecs_wrt_ref
                ) {

  //std::vector<int> ids_sorted;
  //std::vector<int> idx = sortVector(ids, ids_sorted);
  std::vector<int> idx_sorted = sortVector(ids);

  std::cout << "Found the following markers:" << std::endl;
  std::cout << "ID\tx[m]\ty[m]\ttheta[deg]" << std::endl;
  for (int i = 0; i < idx_sorted.size(); i++) {
    int j = idx_sorted[i];
    //std::cout << ids[i] << "\t" << rvecs[i] << "\t" << tvecs[i] << "\t" << rvecs_wrt_ref[i] << "\t" << tvecs_wrt_ref[i] << std::endl;
    if (ids[j] != REFERENCE_MARKER_ID)
        std::cout << std::setprecision(3) << std::fixed << ids[j] << "\t" << tvecs_wrt_ref[j][0] << "\t" << tvecs_wrt_ref[j][1] << "\t" << rvecs_wrt_ref[j][2]*180.0/CV_PI << std::endl;
  }
  std::cout << std::endl;
}

void printStuffsWithRef(
                  std::vector< int > ids,
                  std::vector< cv::Vec3d > rvecs,
                  std::vector< cv::Vec3d > tvecs,
                  std::vector< cv::Vec3d > rvecs_wrt_ref,
                  std::vector< cv::Vec3d > tvecs_wrt_ref
                ) {
  std::cout << "Found the following markers:" << std::endl;
  std::cout << "ID\trvec\ttvec\trvecwrtref\ttvecwrtref" << std::endl;
  for (int i = 0; i < ids.size(); i++) {
    std::cout << ids[i] << "\t" << rvecs[i] << "\t" << tvecs[i] << "\t" << rvecs_wrt_ref[i] << "\t" << tvecs_wrt_ref[i] << std::endl;
    //std::cout << ids[i] << "\t" << rvecs_wrt_ref[i] << "\t" << tvecs_wrt_ref[i] << std::endl;
  }
  std::cout << std::endl;
}

void printStuffs(
                  std::vector< int > ids,
                  std::vector< cv::Vec3d > rvecs,
                  std::vector< cv::Vec3d > tvecs
                ) {
  std::cout << "Found the following markers:" << std::endl;
  std::cout << "ID\trvec\ttvec" << std::endl;
  for (int i = 0; i < ids.size(); i++) {
    std::cout << ids[i] << "\t" << rvecs[i] << "\t" << tvecs[i] << std::endl;
  }
  std::cout << std::endl;
}

static bool readCameraParameters(std::string filename, cameraParameters& camParams) {
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  if (!fs.isOpened())
    return false;
  fs["camera_matrix"] >> camParams.camera_matrix;
  fs["distortion_coefficients"] >> camParams.distortion_coefficients;
  fs["image_width"] >> camParams.image_width;
  fs["image_height"] >> camParams.image_height;
}

static bool readDetectorParameters(std::string filename, cv::Ptr<cv::aruco::DetectorParameters> &params) {
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

cv::Mat vec3dToMat(cv::Vec3d in)
{
    cv::Mat out(3,1, CV_64FC1);
    out.at<double>(0,0) = in[0];
    out.at<double>(1,0) = in[1];
    out.at<double>(2,0) = in[2];
    return out;
}

std::vector<int> sortVector(std::vector<int> in, std::vector<int>& in_sorted) {
    std::vector<int> idx;

    for (int i = 0; i < in.size(); i++)
        in_sorted.push_back(in[i]);

    std::sort(in_sorted.begin(), in_sorted.end());

    for (int i = 0; i < in_sorted.size(); i++) {
        idx.push_back(std::distance(in.begin(), std::find(in.begin(), in.end(), in_sorted[i])));
    }

    return idx;
}

std::vector<int> sortVector(std::vector<int> in) {
    std::vector<int> idx, in_sorted;

    for (int i = 0; i < in.size(); i++)
        in_sorted.push_back(in[i]);

    std::sort(in_sorted.begin(), in_sorted.end());

    for (int i = 0; i < in_sorted.size(); i++) {
        idx.push_back(std::distance(in.begin(), std::find(in.begin(), in.end(), in_sorted[i])));
    }

    return idx;
    
}

