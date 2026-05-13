#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <vector>

class QRDetector
{
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;

    cv::QRCodeDetector qr_detector;
    const std::array<double, 3> detect_scales_{{1.0, 2.0, 3.0}};

    // ================= 统计信息 =================
    struct TimingStats
    {
        size_t frame_count = 0;
        size_t detected_count = 0;

        double total_time_ms = 0.0;
        double total_detected_time_ms = 0.0;

        double avg_time_ms() const
        {
            return frame_count > 0 ? total_time_ms / frame_count : 0.0;
        }

        double avg_detected_time_ms() const
        {
            return detected_count > 0 ? total_detected_time_ms / detected_count : 0.0;
        }
    } stats_;

    // ================= 工具函数 =================
    static std::array<cv::Point2f, 4> points_to_ordered(const std::vector<cv::Point2f>& points)
    {
        std::array<cv::Point2f, 4> ordered;

        const auto sum_cmp = [](const cv::Point2f& a, const cv::Point2f& b)
        {
            return (a.x + a.y) < (b.x + b.y);
        };
        const auto diff_cmp = [](const cv::Point2f& a, const cv::Point2f& b)
        {
            return (a.x - a.y) < (b.x - b.y);
        };

        ordered[0] = *std::min_element(points.begin(), points.end(), sum_cmp);
        ordered[2] = *std::max_element(points.begin(), points.end(), sum_cmp);
        ordered[1] = *std::min_element(points.begin(), points.end(), diff_cmp);
        ordered[3] = *std::max_element(points.begin(), points.end(), diff_cmp);

        return ordered;
    }

    static bool isBetterQuad(const std::array<cv::Point2f, 4>& candidate,
                             const std::array<cv::Point2f, 4>& current)
    {
        const auto quadArea = [](const std::array<cv::Point2f, 4>& quad)
        {
            return std::abs(cv::contourArea(std::vector<cv::Point2f>(quad.begin(), quad.end())));
        };

        return quadArea(candidate) > quadArea(current);
    }

    static bool normalizeQuad(const std::vector<cv::Point2f>& points,
                              const cv::Size& size,
                              std::array<cv::Point2f, 4>& ordered)
    {
        if (points.size() != 4)
            return false;

        ordered = points_to_ordered(points);
        const std::vector<cv::Point2f> ordered_vec(ordered.begin(), ordered.end());

        const double area = std::abs(cv::contourArea(ordered_vec));
        const double min_area = 0.00005 * static_cast<double>(size.area());
        if (area < min_area)
            return false;

        if (!cv::isContourConvex(ordered_vec))
            return false;

        double min_side = std::numeric_limits<double>::max();
        double max_side = 0.0;
        for (int i = 0; i < 4; ++i)
        {
            const cv::Point2f a = ordered[i];
            const cv::Point2f b = ordered[(i + 1) % 4];
            const double len = cv::norm(a - b);
            min_side = std::min(min_side, len);
            max_side = std::max(max_side, len);
        }

        if (min_side < 4.0)
            return false;

        if (max_side / min_side > 6.0)
            return false;

        const auto normalizedDot = [](const cv::Point2f& a, const cv::Point2f& b)
        {
            const double denom = cv::norm(a) * cv::norm(b);
            if (denom <= 1e-6)
                return 1.0;
            return std::abs((a.x * b.x + a.y * b.y) / denom);
        };

        for (int i = 0; i < 4; ++i)
        {
            const cv::Point2f edge1 = ordered[(i + 1) % 4] - ordered[i];
            const cv::Point2f edge2 = ordered[(i + 3) % 4] - ordered[i];
            if (normalizedDot(edge1, edge2) > 0.98)
                return false;
        }

        return true;
    }

    bool detectAndValidate(const cv::Mat& gray,
                           double scale,
                           std::array<cv::Point2f, 4>& ordered)
    {
        cv::Mat scaled_gray;
        if (std::abs(scale - 1.0) < 1e-6)
        {
            scaled_gray = gray;
        }
        else
        {
            cv::resize(gray, scaled_gray, cv::Size(), scale, scale, cv::INTER_AREA);
        }

        std::vector<cv::Point2f> points;
        if (!qr_detector.detect(scaled_gray, points))
            return false;

        return normalizeQuad(points, scaled_gray.size(), ordered);
    }

    static cv::Point clampPoint(const cv::Point2f& point, const cv::Size& size)
    {
        const int x = std::max(0, std::min(size.width - 1, static_cast<int>(std::round(point.x))));
        const int y = std::max(0, std::min(size.height - 1, static_cast<int>(std::round(point.y))));
        return cv::Point(x, y);
    }

    void printTimingTable(double frame_time_ms, bool detected)
    {
        const char* status_color = detected ? "\033[1;32m" : "\033[1;31m";
        const char* reset = "\033[0m";

        printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
        printf("\033[1;34m|                      QR DETECTION TIMING                    |\033[0m\n");
        printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
        printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Total Frames", stats_.frame_count);
        printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Detected Frames", stats_.detected_count);
        printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
        printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Current Frame", "Time (ms)");
        printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
        printf("%s| %-29s | %-27.2f |\033[0m\n",
               status_color,
               detected ? "DETECTED" : "NOT DETECTED",
               frame_time_ms);
        printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
        printf("\033[1;32m| %-29s | %-27.2f |\033[0m\n", "Avg All Frames", stats_.avg_time_ms());
        printf("\033[1;32m| %-29s | %-27.2f |\033[0m\n", "Avg When Detected", stats_.avg_detected_time_ms());
        printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n\n");
    }

public:
    QRDetector() : it_(nh_)
    {
        image_sub_ = it_.subscribe("/left_camera/image", 1, &QRDetector::imageCallback, this);
        image_pub_ = it_.advertise("/qr_detected_image", 1);
        ROS_INFO("QR Detector Node Started (Balanced Detection Mode).");
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

        std::array<cv::Point2f, 4> ordered;
        bool has_valid_quad = false;
        double used_scale = 1.0;

        for (const double scale : detect_scales_)
        {
            if (detectAndValidate(gray, scale, ordered))
            {
                has_valid_quad = true;
                used_scale = scale;
                break;
            }
        }

        if (has_valid_quad)
        {
            std::array<cv::Point, 4> quad;
            for (int i = 0; i < 4; ++i)
            {
                const cv::Point2f scaled_point = ordered[i] * static_cast<float>(1.0 / used_scale);
                quad[i] = clampPoint(scaled_point, frame.size());
            }

            // ✅ 确保绘制绿色方框
            for (int i = 0; i < 4; ++i)
                cv::line(frame, quad[i], quad[(i + 1) % 4], cv::Scalar(0, 255, 0), 3);  // 线宽增加到3

            cv::putText(frame, "QR Detected", quad[0], cv::FONT_HERSHEY_SIMPLEX,
                        0.7, cv::Scalar(0, 255, 0), 2);
        }

        const auto end_time = std::chrono::steady_clock::now();
        const double frame_time_ms =
            std::chrono::duration<double, std::milli>(end_time - start_time).count();

        // 更新统计
        stats_.frame_count++;
        stats_.total_time_ms += frame_time_ms;
        if (has_valid_quad)
        {
            stats_.detected_count++;
            stats_.total_detected_time_ms += frame_time_ms;
        }

        // 打印表格
        printTimingTable(frame_time_ms, has_valid_quad);

        // 显示到图像
        const std::string timing_text = cv::format("%.2f ms (%.1f FPS)",
                                                  frame_time_ms,
                                                  1000.0 / frame_time_ms);
        cv::putText(frame, timing_text, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);

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