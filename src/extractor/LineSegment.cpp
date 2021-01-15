#include "LineSegment.h"

#include <QtMath>

class LineSegmentData : public QSharedData
{
public:
	LineSegmentData(Eigen::Vector3f _start = Eigen::Vector3f(0, 0, 0), Eigen::Vector3f _end = Eigen::Vector3f(0, 0, 0), int _segmentNo = -1)
		: start(_start)
		, end(_end)
		, secondaryDir(Eigen::Vector3f::Zero())
		, segmentNo(_segmentNo)
		, start2d(0, 0)
		, end2d(0, 0)
		, red(0)
		, green(0)
		, blue(0)
		, index(-1)
		, longDescriptor(40)
	{
		shortDescriptor.resize(13);
		std::fill(shortDescriptor.begin(), shortDescriptor.end(), 0);

		localDescriptor.resize(5);
		std::fill(localDescriptor.begin(), localDescriptor.end(), 0);
	}

	Eigen::Vector3f start;
	Eigen::Vector3f end;
	Eigen::Vector3f secondaryDir;
	Eigen::Vector3f center;
	cv::Point2d start2d;
	cv::Point2d end2d;
	double red;
	double green;
	double blue;
	int segmentNo;
	int index;

	std::vector<float> shortDescriptor;
	std::vector<float> longDescriptor;
	std::vector<float> localDescriptor;
};

LineSegment::LineSegment(const Eigen::Vector3f &start, const Eigen::Vector3f &end, int segmentNo, QObject *parent)
	: QObject(parent)
	, data(new LineSegmentData(start, end, segmentNo))
{

}

LineSegment::LineSegment(const LineSegment &rhs) : data(rhs.data)
{

}

LineSegment &LineSegment::operator=(const LineSegment &rhs)
{
	if (this != &rhs)
		data.operator=(rhs.data);
	return *this;
}

LineSegment::~LineSegment()
{

}

Eigen::Vector3f LineSegment::start() const
{
	return data->start;
}

void LineSegment::setStart(const Eigen::Vector3f &pt)
{
	data->start = pt;
}

Eigen::Vector3f LineSegment::end() const
{
	return data->end;
}

void LineSegment::setEnd(const Eigen::Vector3f &pt)
{
	data->end = pt;
}

Eigen::Vector3f LineSegment::middle() const
{
	return (start() + end()) / 2;
}

Eigen::Vector3f LineSegment::center() const
{
	return data->center;
}

void LineSegment::setCenter(const Eigen::Vector3f& center)
{
	data->center = center;
}

cv::Point2f LineSegment::start2d() const
{
	return data->start2d;
}

Eigen::Vector3f LineSegment::secondaryDir() const
{
	return data->secondaryDir;
}

void LineSegment::setSecondaryDir(const Eigen::Vector3f & dir)
{
	data->secondaryDir = dir;
}

void LineSegment::setStart2d(const cv::Point2f& _value)
{
	data->start2d = _value;
}

cv::Point2f LineSegment::end2d() const
{
	return data->end2d;
}

void LineSegment::setEnd2d(const cv::Point2f& _value)
{
	data->end2d = _value;
}

float LineSegment::length() const
{
	return (end() - start()).norm();
}

Eigen::Vector3f LineSegment::direction() const
{
	return end() - start();
}

Eigen::Vector3f LineSegment::normalizedDir() const
{
	return direction().normalized();
}

int LineSegment::index() const
{
	return data->index;
}

void LineSegment::setIndex(int index)
{
	data->index = index;
}

void LineSegment::reproject(const Eigen::Matrix3f& rot, const Eigen::Vector3f& trans)
{
	data->start = rot * data->start + trans;
	data->end = rot * data->end + trans;
	Eigen::Vector3f middle = rot * this->middle() + trans;
	data->center = rot * LineSegment::center() + trans;

	Eigen::Vector3f dir = direction().normalized();
	float length = (data->start - data->end).norm();
	Eigen::Vector3f projPt = data->start - dir * data->start.dot(dir);

	data->shortDescriptor[0] = projPt.x();
	data->shortDescriptor[1] = projPt.y();
	data->shortDescriptor[2] = projPt.z();
	data->shortDescriptor[3] = dir.x();
	data->shortDescriptor[4] = dir.y();
	data->shortDescriptor[5] = dir.z();
	data->shortDescriptor[6] = data->center.x();
	data->shortDescriptor[7] = data->center.y();
	data->shortDescriptor[8] = data->center.z();
	data->shortDescriptor[9] = data->red / 255;
	data->shortDescriptor[10] = data->green / 255;
	data->shortDescriptor[11] = data->blue / 255;
	data->shortDescriptor[12] = length;
	Eigen::Matrix<float, 1, 13>::Map(data->shortDescriptor.data(), 1, 13).normalize();
	//std::cout << data->shortDescriptor << std::endl;
}

int LineSegment::shortDescriptorSize()
{
	return 13;
}

int LineSegment::longDescriptorSize()
{
	return 5 * 8;
}

const std::vector<float>& LineSegment::longDescriptor() const
{
	return data->longDescriptor;
}

void LineSegment::setLongDescriptor(const std::vector<float>& desc)
{
	data->longDescriptor = desc;
}



void LineSegment::calculateColorAvg(const cv::Mat& mat)
{
	cv::Point2f dir = end2d() - start2d();
	int length = roundf(cv::norm(dir));
	dir /= length;

	//std::cout << "    color mat: " << mat.cols << ", " << mat.rows << ", start: [" << start2d().x << ", " << start2d().y << "], end: [" << end2d().x << ", " << end2d().y << "], length: " << length << std::endl;
	cv::Vec3f cvColor(0, 0, 0);
	for (int i = 0; i < length; i++)
	{
		cv::Point2f pt = start2d() + dir * i;
		if (pt.x < 0 || pt.x >= mat.cols || pt.y < 0 || pt.y >= mat.rows)
			continue;

		cvColor += mat.at<cv::Vec3b>(pt.y, pt.x);
	}
	cvColor /= length;

	data->red = cvColor[0];
	data->green = cvColor[1];
	data->blue = cvColor[2];

	//std::cout << "    color: " << data->red << ", " << data->green << ", " << data->blue << std::endl;
}

void LineSegment::drawColorLine(cv::Mat& mat)
{
	cv::Point2f dir = end2d() - start2d();
	int length = roundf(cv::norm(dir));
	dir /= length;

	cv::Vec3f cvColor(0, 0, 0);
	for (int i = 0; i < length; i++)
	{
		cv::Point2f pt = start2d() + dir * i;
		if (pt.x < 0 || pt.x >= mat.cols || pt.y < 0 || pt.y >= mat.rows)
			continue;

		mat.at<cv::Vec3b>(pt.y, pt.x) = cv::Vec3b(data->red, data->green, data->blue);
	}
}

int LineSegment::segmentNo() const
{
	return data->segmentNo;
}

void LineSegment::setSegmentNo(int segmentNo)
{
	data->segmentNo = segmentNo;
}

bool LineSegment::available() const
{
	return length() > 0;
}

void LineSegment::reverse()
{
	Eigen::Vector3f tmp = data->start;
	data->start = data->end;
	data->end = tmp;
}

bool LineSegment::similarDirection(const LineSegment &other, float &angle, float threshold)
{
	angle = angleToAnotherLine(other);
	if (angle < threshold)
	{
		return true;
	}
	else if (angle > (M_PI - threshold))
	{
		angle = M_PI - angle;
		return true;
	}

	return false;
}

bool LineSegment::similarDirection(const LineSegment &other, float threshold)
{
	float angle = angleToAnotherLine(other);
	if (angle < threshold || angle >(M_PI - threshold))
		return true;
	return false;
}

void LineSegment::applyAnotherLineDirection(const LineSegment& other)
{
	if (direction().dot(other.direction()) < 0)
	{
		reverse();
	}
}

float LineSegment::angleToAnotherLine(const LineSegment &other)
{
	return qAcos(direction().normalized().dot(other.direction().normalized()));
}

std::vector<float> LineSegment::shortDescriptor() const
{
	return data->shortDescriptor;
}

std::vector<float> LineSegment::localDescriptor() const
{
	return data->localDescriptor;
}

void LineSegment::generateLocalDescriptor()
{
	Eigen::Vector3f point = closedPointOnLine(Eigen::Vector3f::Zero());
	Eigen::Vector2f angles = calculateAngles();
	data->localDescriptor[0] = angles.x();
	data->localDescriptor[1] = angles.y();
	data->localDescriptor[2] = point.x();
	data->localDescriptor[3] = point.y();
	data->localDescriptor[4] = point.z();
	Eigen::Matrix<float, 1, 5>::Map(data->localDescriptor.data(), 1, 5).normalize();
}

void LineSegment::showLocalDescriptor()
{
	std::cout << data->segmentNo <<  " [" << data->start.transpose() << "]: ";
	for (int i = 0; i < data->localDescriptor.size(); i++)
	{
		std::cout << data->localDescriptor[i] << " ";
	}
	std::cout << std::endl;
}

//Eigen::VectorXf LineSegment::longDescriptor() const
//{
//    return data->longDescriptor;
//}

float LineSegment::averageDistance(const LineSegment &other)
{
	float distM = pointDistance(other.middle());
	return distM;
}

float LineSegment::pointDistance(const Eigen::Vector3f &point)
{
	Eigen::Vector3f line = point - data->start;
	Eigen::Vector3f dir = direction().normalized();
	Eigen::Vector3f pointProj = data->start + dir * (line.dot(dir));
	float distance = qAbs((point - pointProj).norm());
	return distance;
}

Eigen::Vector3f LineSegment::closedPointOnLine(const Eigen::Vector3f &point)
{
	Eigen::Vector3f dir = direction().normalized();
	Eigen::Vector3f ev = point - data->start;
	Eigen::Vector3f closedPoint = data->start + dir * (ev.dot(dir));
	return closedPoint;
}

Eigen::Vector2f LineSegment::calculateAngles()
{
	Eigen::Vector3f dir = direction().normalized();
	float xProj = dir.dot(Eigen::Vector3f(1.0f, 0.0f, 0.0f));
	float yProj = dir.dot(Eigen::Vector3f(0.0f, 1.0f, 0.0f));
	float zProj = dir.dot(Eigen::Vector3f(0.0f, 0.0f, 1.0f));
	float xzProj = (dir - Eigen::Vector3f(0.0f, 1.0f, 0.0f) * yProj).norm();
	float angleH = qAtan2(zProj, xProj);
	float angleV = qAtan2(yProj, xzProj);
	return Eigen::Vector2f(angleH, angleV);
}

double LineSegment::red() const
{
	return data->red;
}

double LineSegment::green() const
{
	return data->green;
}

double LineSegment::blue() const
{
	return data->blue;
}

Eigen::Matrix3f LineSegment::localRotaion() const
{
	Eigen::Vector3f xAxis = direction().normalized();
	Eigen::Vector3f yAxis = xAxis.cross(secondaryDir()).normalized();
	Eigen::Vector3f zAxis = xAxis.cross(yAxis).normalized();
	Eigen::Matrix3f matrix = (Eigen::AngleAxisf(0, xAxis) * Eigen::AngleAxisf(0, yAxis) * Eigen::AngleAxisf(0, zAxis)).toRotationMatrix();
	return matrix;
}

LineSegment LineSegment::clone()
{
	LineSegment line(data->start, data->end, data->segmentNo);
	line.data->start = data->start;
	line.data->end = data->end;
	line.data->secondaryDir = data->secondaryDir;
	line.data->center = data->center;
	line.data->start2d = data->start2d;
	line.data->end2d = data->end2d;
	line.data->red = data->red;
	line.data->green = data->green;
	line.data->blue = data->blue;
	line.data->segmentNo = data->segmentNo;
	line.data->index = data->index;
	return line;
}

