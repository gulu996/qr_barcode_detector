#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/String.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>

class QRDetector
{
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    ros::Publisher text_pub_;

    cv::QRCodeDetector qr_detector;

    static bool isValidQuad(const std::vector<cv::Point2f>& points, const cv::Size& size)
    {
        if (points.size() != 4)
            return false;

        const double area = std::abs(cv::contourArea(points));
        const double min_area = 0.0003 * static_cast<double>(size.area());
        if (area < min_area)
            return false;

        if (!cv::isContourConvex(points))
            return false;

        double min_side = std::numeric_limits<double>::max();
        double max_side = 0.0;
        for (int i = 0; i < 4; ++i)
        {
            const cv::Point2f a = points[i];
            const cv::Point2f b = points[(i + 1) % 4];
            const double len = cv::norm(a - b);
            min_side = std::min(min_side, len);
            max_side = std::max(max_side, len);
        }

        if (min_side < 5.0)
            return false;

        if (max_side / min_side > 6.0)
            return false;

        return true;
    }

    static cv::Point clampPoint(const cv::Point2f& point, const cv::Size& size)
    {
        const int x = std::max(0, std::min(size.width - 1, static_cast<int>(std::round(point.x))));
        const int y = std::max(0, std::min(size.height - 1, static_cast<int>(std::round(point.y))));
        return cv::Point(x, y);
    }

public:
    QRDetector() : it_(nh_)
    {
        image_sub_ = it_.subscribe("/left_camera/image", 1, &QRDetector::imageCallback, this);
        image_pub_ = it_.advertise("/qr_detected_image", 1);
        text_pub_ = nh_.advertise<std_msgs::String>("/qr_decoded_text", 1);
        ROS_INFO("QR Detector Node Started (Decode + Timing Mode).");
    }

    void imageCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        const auto start_time = std::chrono::steady_clock::now();

        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat frame = cv_ptr->image;
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> points;
        const std::string decoded_text = qr_detector.detectAndDecode(gray, points);
        const bool has_valid_quad = isValidQuad(points, frame.size());
        const bool decoded = !decoded_text.empty();

        if (decoded && has_valid_quad)
        {
            std::array<cv::Point, 4> quad;
            for (int i = 0; i < 4; ++i)
                quad[i] = clampPoint(points[i], frame.size());

            for (int i = 0; i < 4; ++i)
                cv::line(frame, quad[i], quad[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);

            int min_x = frame.cols - 1;
            int min_y = frame.rows - 1;
            for (const auto& p : quad)
            {
                min_x = std::min(min_x, p.x);
                min_y = std::min(min_y, p.y);
            }

            const cv::Point label_pos(min_x, std::max(0, min_y - 10));
            const std::string label = "QR: " + decoded_text;
            cv::putText(frame, label, label_pos, cv::FONT_HERSHEY_SIMPLEX,
                        0.6, cv::Scalar(0, 255, 0), 2);

            std_msgs::String msg_out;
            msg_out.data = decoded_text;
            text_pub_.publish(msg_out);

            ROS_INFO_THROTTLE(1.0, "QR decoded: %s", decoded_text.c_str());
        }

        const auto end_time = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        const double fps = ms > 0.0 ? 1000.0 / ms : 0.0;
        const std::string timing_text = cv::format("Proc: %.2f ms (%.1f FPS)", ms, fps);
        cv::putText(frame, timing_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(0, 255, 255), 2);

        ROS_INFO_THROTTLE(1.0, "Frame time: %.2f ms (%.1f FPS)", ms, fps);

        image_pub_.publish(cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg());
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "qr_detector_node");
    QRDetector detector;
    ros::spin();
    return 0;
}
